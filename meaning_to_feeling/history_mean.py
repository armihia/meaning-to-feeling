import copy

import numpy as np

from .prompts import EMOTIONS, HISTORY_MEAN_PARAMETERS, HISTORY_MEAN_PROFILE_KIND


def validate_history_mean(profile):
    if (
        profile.get("emotion_order") != EMOTIONS
        or profile.get("parameter_order") != HISTORY_MEAN_PARAMETERS
    ):
        raise ValueError("历史均值档案需要完整的情绪及参数顺序")
    counts = profile.get("phase_history_trials")
    if (
        not isinstance(counts, list)
        or len(counts) != 2
        or any(type(n) is not int or n < 0 for n in counts)
    ):
        raise ValueError("历史均值档案需要开始前、正文各自的有效历史段数")
    values = profile.get("vector")
    if (
        not isinstance(values, (list, tuple, np.ndarray))
        or len(values) != 22
        or any(isinstance(v, (bool, str)) or not isinstance(v, (int, float, np.number)) for v in values)
    ):
        raise ValueError("历史均值向量需要22个有限数值及完整档案")
    vector = np.asarray(values, float)
    sums = vector[:20].reshape(2, 10)
    if (
        not np.isfinite(vector).all()
        or np.any(sums < 0)
        or np.any(sums > np.asarray(counts)[:, None] + 1e-6)
        or not np.allclose(sums.sum(1), counts, atol=1e-6, rtol=0)
        or np.any(np.abs(vector[20:]) > 1 + 1e-6)
    ):
        raise ValueError("历史感受合计与有效段数不一致，或切分修正超出范围")
    return vector


def from_statistics(base_profile, sums, counts):
    result = copy.deepcopy(base_profile)
    result.update(
        profile_kind=HISTORY_MEAN_PROFILE_KIND,
        parameter_order=HISTORY_MEAN_PARAMETERS.copy(),
        phase_history_trials=list(counts),
        vector=[*np.asarray(sums, float).reshape(-1).tolist(), *base_profile["vector"][-2:]],
        generator="本人独立历史感受均值融合G",
        scope="开始前与正文分别按历史段等权融合；表示任务反应倾向，未验证为稳定心理人格。",
    )
    validate_history_mean(result)
    return result


def generate_profile(engine, value):
    observations = []
    base = engine.generate_profile(value, _emotion_observations=observations)
    sums, counts = np.zeros((2, 10)), [0, 0]
    for ends, _, targets in observations:
        phases = (np.asarray(ends) > 0).astype(int)
        for phase in (0, 1):
            selected = phases == phase
            if selected.any():
                sums[phase] += np.asarray(targets)[selected].mean(0)
                counts[phase] += 1
    return from_statistics(base, sums, counts)


def calibrated_state(shared, vector, counts, end_char):
    phase = int(end_char > 0)
    if counts[phase] == 0:
        return shared.copy()
    total = vector[10 * phase:10 * (phase + 1)]
    return (total + 3.0 * shared) / (counts[phase] + 3.0)
