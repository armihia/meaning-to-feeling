import json
from pathlib import Path

import numpy as np

from .core import text_row, validate_profile
from .prompts import DYNAMIC_PARAMETERS, DYNAMIC_PROFILE_KIND, EMOTIONS


class GainCalibration:
    def __init__(self, path, history_head, composition_basis):
        saved = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(saved, dict):
            raise ValueError("正文增益参数必须是对象")
        if (
            saved.get("emotion_order") != EMOTIONS
            or saved.get("history_head_name") != history_head
        ):
            raise ValueError("正文增益参数的情绪顺序或历史语义头与当前模型不同")
        variance = saved.get("gain_variance")
        if (
            type(variance) not in (int, float)
            or not np.isfinite(variance)
            or variance < 0
        ):
            raise ValueError("正文增益先验方差必须是有限非负数")
        self.variance = float(variance)
        try:
            self.noise = np.asarray(saved.get("noise"), float)
        except (TypeError, ValueError) as exc:
            raise ValueError("正文增益噪声需要9维正定协方差") from exc
        if (
            self.noise.shape != (9, 9)
            or not np.isfinite(self.noise).all()
            or not np.allclose(self.noise, self.noise.T, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("正文增益噪声需要9维对称正定协方差")
        try:
            np.linalg.cholesky(self.noise)
        except np.linalg.LinAlgError as exc:
            raise ValueError("正文增益噪声需要9维正定协方差") from exc
        with np.errstate(over="ignore", invalid="ignore"):
            self.precision = np.linalg.inv(self.noise)
        if not np.isfinite(self.precision).all():
            raise ValueError("正文增益噪声的精度超出有限数值范围")
        self.composition_basis = composition_basis

    def infer(self, differences):
        information = score = 0.0
        for dq, dy in differences:
            dx = np.asarray(dq, float) @ self.composition_basis
            target = np.asarray(dy, float) @ self.composition_basis
            if (
                dx.shape != (9,)
                or target.shape != (9,)
                or not np.isfinite(dx).all()
                or not np.isfinite(target).all()
            ):
                raise ValueError("正文增益只接受完整有限的实际相邻变化")
            with np.errstate(over="ignore", invalid="ignore"):
                information += float(dx @ self.precision @ dx)
                score += float(dx @ self.precision @ (target - dx))
        denominator = 1.0 + self.variance * information
        if not np.isfinite([information, score, denominator]).all() or information < 0:
            raise ValueError("正文增益历史信息超出有限数值范围")
        posterior_variance = self.variance / denominator
        mean = 1.0 + posterior_variance * score
        if not np.isfinite([posterior_variance, mean]).all():
            raise ValueError("正文增益后验超出有限数值范围")
        gain = float(np.clip(mean, 0.0, 2.0))
        return gain, posterior_variance, information


class ReliabilityCalibration:
    def __init__(self, path, history_head, *, gain_prior=None):
        with np.load(Path(path), allow_pickle=False) as saved:
            self.mean = saved["mean"].astype(float)
            self.between = saved["between"].astype(float)
            self.noise = saved["noise"].astype(float)
            self.composition_basis = saved["composition_basis"].astype(float)
            order = saved["emotion_order"].tolist()
            head = saved["history_head_name"].item()
        if order != EMOTIONS or head != history_head:
            raise ValueError("可靠性参数的情绪顺序或历史语义头与当前模型不同")
        self.history_head_name = history_head
        if (
            self.mean.shape != (9,)
            or self.between.shape != (9, 9)
            or self.noise.shape != (9, 9)
            or self.composition_basis.shape != (10, 9)
        ):
            raise ValueError("可靠性参数维度错误")
        if not all(
            np.isfinite(a).all()
            for a in [self.mean, self.between, self.noise, self.composition_basis]
        ):
            raise ValueError("可靠性参数含无效数值")
        np.linalg.cholesky(self.between)
        np.linalg.cholesky(self.noise)
        self.prior_precision = np.linalg.inv(self.between)
        self.observation_precision = np.linalg.inv(self.noise)
        self.gain = (
            GainCalibration(gain_prior, history_head, self.composition_basis)
            if gain_prior is not None
            else None
        )

    def infer(self, residuals):
        values = np.asarray(residuals, float).reshape(-1, 10)
        if not np.isfinite(values).all():
            raise ValueError("历史残差含无效数值，缺失观测不能补零")
        r = values @ self.composition_basis
        covariance = np.linalg.inv(
            self.prior_precision + len(r) * self.observation_precision
        )
        coefficient = covariance @ self.observation_precision @ (r - self.mean).sum(0)
        offset = self.composition_basis @ (self.mean + coefficient)
        variance = np.diag(
            self.composition_basis @ covariance @ self.composition_basis.T
        )
        return offset, variance

    def generate_profile(self, engine, value):
        if self.gain is not None:
            return self._generate_gain_profile(engine, value)
        result = engine.generate_profile(value)
        residuals = []
        used_points = 0
        for item in value["history"]:
            points = [
                p for p in item.get("emotion_observations", []) if p["end_char"] > 0
            ]
            if not points:
                continue
            ends = [p["end_char"] for p in points]
            _, semantic, _ = engine.states(text_row(item), ends, np.zeros(12))
            q = np.asarray(
                [engine.shared_state(semantic[e], history=True) for e in ends]
            )
            y = np.asarray([p["values"] for p in points])
            residuals.append((y - q).mean(0))
            used_points += len(points)
        offset, variance = self.infer(residuals)
        result["vector"][:10] = offset.tolist()
        result["generator"] = "按历史可靠性加权的情绪校准G"
        result["emotion_history_trials"] = len(residuals)
        result["reliability"] = dict(
            method="gaussian_composition_contrasts",
            history_trials=len(residuals),
            body_observation_points=used_points,
            emotion_order=EMOTIONS,
            posterior_variance=variance.tolist(),
            interpretation="各情绪在独立历史正文中的校准倾向；开始前保留作背景记录，无正文评分时使用训练群体先验。",
        )
        return result

    def _generate_gain_profile(self, engine, value):
        if engine.head is None or engine.history_head is None:
            raise ValueError("正文增益校准需要配套的query与history数值头")
        if engine.history_head["name"] != self.history_head_name:
            raise ValueError("正文增益先验与实际历史数值头不同")
        collected = []
        result = engine.generate_profile(value, _emotion_observations=collected)
        pairs, differences = [], []
        used_points = 0
        for ends, q, y in collected:
            indices = sorted(
                (j for j, end in enumerate(ends) if end > 0), key=ends.__getitem__
            )
            if not indices:
                continue
            q, y = np.asarray(q)[indices], np.asarray(y)[indices]
            pairs.append((q, y))
            used_points += len(indices)
            if len(indices) > 1:
                count = len(indices) - 1
                differences.append(((q[-1] - q[0]) / count, (y[-1] - y[0]) / count))
        gain, gain_variance, information = self.gain.infer(differences)
        initial, initial_variance = self.infer([(y - q).mean(0) for q, y in pairs])
        history_prior = np.asarray(engine.history_head["prior"], float)
        if gain == 1.0:
            body, body_variance = initial.copy(), initial_variance.copy()
        else:
            body_residuals = [
                (y - (history_prior + gain * (q - history_prior))).mean(0)
                for q, y in pairs
            ]
            body, body_variance = self.infer(body_residuals)
        result["vector"] = [
            *body.tolist(),
            *initial.tolist(),
            *result["vector"][10:12],
            gain,
        ]
        result["profile_kind"] = DYNAMIC_PROFILE_KIND
        result["parameter_order"] = DYNAMIC_PARAMETERS.copy()
        result["generator"] = "按正文可靠性校准与变化噪声收缩的初始参考/正文增益G"
        result["emotion_history_trials"] = len(pairs)
        result["reliability"] = dict(
            method="gaussian_composition_contrasts_with_body_gain",
            history_trials=len(pairs),
            body_observation_points=used_points,
            emotion_order=EMOTIONS,
            posterior_variance=body_variance.tolist(),
            posterior_variance_scope="给定已估计gain的正文偏移条件方差，不包含gain不确定性的传播",
            initial_reference_posterior_variance=initial_variance.tolist(),
            body_gain=dict(
                method="noise_aware_scalar_posterior",
                prior_mean=1.0,
                prior_variance=self.gain.variance,
                posterior_variance=gain_variance,
                posterior_variance_scope="gain裁剪到0至2之前的高斯后验方差",
                information=information,
                history_trials=len(differences),
                observation="每段实际正文首末点的平均相邻变化；每段最多一个观测，不包含开始前评分",
                clip_range=[0.0, 2.0],
            ),
            interpretation="初始参考沿用gain=1的正文可靠性偏移；正文偏移在给定gain后重新估计，无正文评分时使用训练群体先验。",
        )
        result["history_coverage"]["body_emotion_trials"] = len(pairs)
        result["history_coverage"]["body_emotion_points"] = used_points
        result["history_coverage"]["body_change_trials"] = len(differences)
        result["scope"] = (
            "独立历史正文中的行为响应校准与变化增益参数，不是已验证的心理或临床人格维度；"
            "初始参考为gain=1的校准，未用开始前评分拟合；G不接收待预测query"
        )
        validate_profile(result, engine.model, engine.head_name)
        return result
