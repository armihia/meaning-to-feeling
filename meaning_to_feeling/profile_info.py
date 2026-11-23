import numpy as np

from .prompts import EMOTIONS


def history_coverage(emotion_pairs, boundary_counts):
    result = dict(
        emotion_trials=len(emotion_pairs),
        emotion_points=sum(len(y) for _, y in emotion_pairs),
        nonpunctuation_points=int(boundary_counts[0]),
        punctuation_points=int(boundary_counts[1]),
        emotions=[],
    )
    if emotion_pairs:
        expected = np.concatenate([q for q, _ in emotion_pairs])
        observed = np.concatenate([y for _, y in emotion_pairs])
        result['emotions'] = [
            dict(key=key, reference_min=float(expected[:, j].min()),
                 reference_max=float(expected[:, j].max()),
                 observed_min=float(observed[:, j].min()),
                 observed_max=float(observed[:, j].max()))
            for j, key in enumerate(EMOTIONS)
        ]
    return result
