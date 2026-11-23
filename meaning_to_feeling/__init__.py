import json
from pathlib import Path

from .backend import LocalStore, Runtime
from .core import CompositeResponseModel, text_row
from .history_scope import select_completed_history
from .prompts import EMOTIONS

__all__ = ["Model", "EMOTIONS"]

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "model_weights/meaning_to_feeling_9b"


class Model:
    def __init__(
        self,
        weights,
        work_dir=".runtime",
        port=11436,
        *,
        calibration_prior=None,
        gain_prior=None,
    ):
        if gain_prior is not None and calibration_prior is None:
            raise ValueError("正文增益先验需要同时提供正文可靠性calibration_prior")
        self.weights = Path(weights).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.parameters = json.loads(
            (self.weights / "model.json").read_text(encoding="utf-8-sig")
        )
        self.runtime = Runtime(self.weights, self.work_dir, port)
        self.engine = None
        self.calibration = None
        if calibration_prior is not None:
            from .reliability import ReliabilityCalibration

            self.calibration = ReliabilityCalibration(
                calibration_prior,
                self.parameters["heads"]["history"]["name"],
                gain_prior=gain_prior,
            )

    def __enter__(self):
        self.runtime.start()
        try:
            store = LocalStore(self.runtime, self.work_dir / "responses.sqlite")
            try:
                self.engine = CompositeResponseModel(
                    store,
                    self.parameters["heads"]["query"],
                    self.parameters["heads"]["history"],
                    self.parameters["model"],
                )
            except BaseException:
                store.close()
                raise
            return self
        except BaseException:
            self.runtime.close()
            raise

    def __exit__(self, *_):
        try:
            if self.engine is not None:
                self.engine.close()
                self.engine = None
        finally:
            self.runtime.close()

    def generate_profile(self, history, *, strategy=None, as_of=None):
        if self.engine is None:
            raise RuntimeError("请在with Model(...)内调用")
        if strategy not in (None, "original", "history_mean"):
            raise ValueError("不支持这个个人档案生成方式")
        history, scope = select_completed_history(history, as_of)
        if strategy == "history_mean":
            from .history_mean import generate_profile

            profile = generate_profile(self.engine, history)
        elif self.calibration is not None:
            profile = self.calibration.generate_profile(self.engine, history)
        else:
            profile = self.engine.generate_profile(history)
        if scope is not None:
            profile["history_time_scope"] = scope
        return profile

    def predict(self, value, profile=None, *, segmentation_policy="detailed"):
        if self.engine is None:
            raise RuntimeError("请在with Model(...)内调用")
        if (
            profile is None
            and self.calibration is not None
            and "personality_vector" not in value
        ):
            profile = self.generate_profile(dict(history=[], emotion_order=EMOTIONS))
        return self.engine.predict(
            value, profile, segmentation_policy=segmentation_policy
        )

    def observe(self, value, end_char):
        if self.engine is None:
            raise RuntimeError("请在with Model(...)内调用")
        row = text_row(value)
        if type(end_char) is not int or not 0 <= end_char <= len(row["text"]):
            raise ValueError("前缀位置超出正文")
        answer, _ = self.engine.store.one(
            dict(
                character=row["character"],
                context=row["context"],
                prefix=row["text"][:end_char],
            )
        )
        return self.engine.state_readout(answer).tolist()
