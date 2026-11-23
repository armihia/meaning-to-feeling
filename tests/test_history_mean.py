import copy
import json
import unittest

import numpy as np

from meaning_to_feeling import Model
from meaning_to_feeling.core import validate_profile
from meaning_to_feeling.history_mean import from_statistics, generate_profile
from meaning_to_feeling.prompts import EMOTIONS
from test_inference import InferenceTests


def emotion(joy):
    return [joy, *[0.0] * 8, 1 - joy]


class HistoryMeanTests(unittest.TestCase):
    def test_cutoff_uses_completion_time_and_keeps_the_equal_instant(self):
        engine, store = InferenceTests().engine()
        model = Model.__new__(Model)
        model.engine, model.calibration = engine, None
        cutoff = "2026-11-19T09:00:00.000001+08:00"
        earlier = dict(
            text="我回家了。", sample_id="earlier", finished_at="2026-11-19T00:50:00Z",
            emotion_observations=[dict(end_char=5, values=emotion(.6))],
        )
        equal = dict(
            text="我放下书。", sample_id="equal",
            finished_at="2026-11-19T01:00:00.000001Z",
            emotion_observations=[dict(end_char=5, values=emotion(.2))],
        )
        future = dict(
            earlier, text="我走进屋。", sample_id="future",
            started_at="2026-11-19T00:55:00Z", finished_at="2026-11-19T01:00:00.000002Z",
        )
        unknown = dict(future, text="我完成了。", sample_id="unknown", finished_at=None)
        naive = dict(future, text="我走出门。", sample_id="naive", finished_at="2026-11-19T00:50:00")
        history = dict(emotion_order=EMOTIONS, history=[future, earlier, unknown, equal, naive])
        before = copy.deepcopy(history)
        scoped = model.generate_profile(history, strategy="history_mean", as_of=cutoff)
        self.assertEqual({row["prefix"] for row in store.visible}, {earlier["text"], equal["text"]})
        self.assertEqual(history, before)
        self.assertEqual(scoped.pop("history_time_scope"), dict(
            as_of="2026-11-19T01:00:00.000001+00:00",
            provided_trials=5, eligible_trials=2, after_cutoff_trials=1,
            unverified_completion_trials=2,
        ))
        manual = model.generate_profile(dict(history, history=[earlier, equal]), strategy="history_mean")
        self.assertEqual(scoped, manual)
        original = model.generate_profile(history, strategy="history_mean")
        self.assertEqual(original["history_trials"], 5)
        self.assertNotIn("history_time_scope", original)

    def test_trial_equal_weight_and_separate_initial_body(self):
        engine, _ = InferenceTests().engine()
        history = dict(emotion_order=EMOTIONS, history=[
            dict(text="我回家。", emotion_observations=[
                dict(end_char=0, values=emotion(1)),
                dict(end_char=2, values=emotion(.2)),
                dict(end_char=4, values=emotion(.4)),
            ]),
            dict(text="我到家。", emotion_observations=[dict(end_char=4, values=emotion(.9))]),
        ])
        before = copy.deepcopy(history)
        profile = generate_profile(engine, history)
        self.assertEqual(history, before)
        self.assertEqual(profile["phase_history_trials"], [1, 2])
        np.testing.assert_allclose(profile["vector"][:10], emotion(1))
        np.testing.assert_allclose(profile["vector"][10:20], 2 * np.asarray(emotion(.6)))
        profile = json.loads(json.dumps(profile))
        engine.generate_profile = lambda *_args, **_kwargs: self.fail("F不应重建G")
        result = engine.predict(dict(text="我吃饭。"), profile)
        np.testing.assert_allclose(result["initial_emotion"], emotion(.25), atol=1e-15)
        np.testing.assert_allclose(result["emotion_matrix"][-1], emotion(.36), atol=1e-15)
        self.assertEqual(result["emotion_calibration"]["history_weights"], [.25, .4])
        self.assertEqual(result["model"]["profile_dimensions"], 22)
        states = np.asarray([result["initial_emotion"], *result["emotion_matrix"]])
        np.testing.assert_allclose(result["delta_matrix"], np.diff(states, axis=0))

    def test_empty_history_exactly_retains_shared_prediction_and_boundaries(self):
        engine, _ = InferenceTests().engine()
        profile = generate_profile(engine, dict(history=[], emotion_order=EMOTIONS))
        value = dict(text="我回家了，终于能休息。")
        old = engine.predict(value, [0.0] * 12)
        new = engine.predict(value, profile)
        for field in ["units", "initial_emotion", "emotion_matrix", "delta_matrix"]:
            self.assertEqual(old[field], new[field], field)

    def test_missing_initial_is_not_a_neutral_observation(self):
        engine, _ = InferenceTests().engine()
        base = engine.generate_profile(dict(history=[], emotion_order=EMOTIONS))
        profile = from_statistics(base, [np.zeros(10), emotion(.8)], [0, 1])
        result = engine.predict(dict(text="我回家。"), profile)
        self.assertEqual(result["initial_emotion"], emotion(0))
        np.testing.assert_allclose(result["emotion_matrix"][-1], emotion(.35))

    def test_malformed_profile_fails_before_reading_any_text(self):
        engine, store = InferenceTests().engine()
        good = generate_profile(engine, dict(history=[], emotion_order=EMOTIONS))
        bad = [good["vector"], dict(good, phase_history_trials=[1, 0]), dict(good, phase_history_trials=[False, 0])]
        for field in ["profile_kind", "parameter_order", "emotion_order", "phase_history_trials"]:
            item = copy.deepcopy(good)
            item.pop(field)
            bad.append(item)
        for item in bad:
            with self.assertRaises(ValueError):
                validate_profile(item, engine.model, "test")
        self.assertEqual(store.visible, [])


if __name__ == "__main__":
    unittest.main()
