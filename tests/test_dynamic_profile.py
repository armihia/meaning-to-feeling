import copy
import unittest

import numpy as np

from meaning_to_feeling.core import simplex, text_row, validate_profile
from meaning_to_feeling.prompts import (
    DYNAMIC_PARAMETERS,
    DYNAMIC_PROFILE_KIND,
    EMOTIONS,
)
import test_inference


class DynamicProfileTests(unittest.TestCase):
    def engine(self):
        return test_inference.InferenceTests().engine()

    def profile(self, gain=0.4, rates=(0.0, 0.0)):
        return dict(
            vector=[0.01]
            + [0.0] * 8
            + [-0.01]
            + [0.02]
            + [0.0] * 8
            + [-0.02]
            + list(rates)
            + [gain],
            emotion_order=EMOTIONS,
            parameter_order=DYNAMIC_PARAMETERS,
            profile_kind=DYNAMIC_PROFILE_KIND,
            semantic_head="test",
        )

    def test_gain1_reproduces_old_F_including_cutting(self):
        for policy in ("legacy", "detailed"):
            for rates in ((0.0, 0.0), (-0.5, 0.4), (0.2, -0.1)):
                engine, _ = self.engine()
                p = self.profile(1.0, rates)
                p["vector"][10:20] = p["vector"][:10]
                old = p["vector"][:10] + list(rates)
                value = dict(
                    text="我到家了，终于能休息。", candidate_ends=[2, 4, 6, 8, 12]
                )
                value["candidate_ends"][-1] = len(value["text"])
                a = engine.predict(value, old, segmentation_policy=policy)
                b = engine.predict(value, p, segmentation_policy=policy)
                for key in (
                    "units",
                    "initial_emotion",
                    "emotion_matrix",
                    "delta_matrix",
                    "segmentation",
                ):
                    self.assertEqual(a[key], b[key], key)
                self.assertEqual(b["model"]["profile_dimensions"], 23)

    def test_body_uses_actual_query_center_and_preserves_initial(self):
        engine, _ = self.engine()
        engine.head = dict(engine.head, prior=[0.3] + [0.0] * 8 + [0.7])
        engine.history_head = dict(engine.history_head, prior=[0.7] + [0.0] * 8 + [0.3])
        row = text_row(dict(text="我回家。", context_text="背景。"))
        p = self.profile()
        v = validate_profile(p, engine.model, "test")
        states, semantic, _ = engine.states(row, [0, 2, 4], v)
        expected_initial = simplex((engine.shared_state(semantic[0]) + v[10:20])[None])[
            0
        ]
        np.testing.assert_array_equal(states[0], expected_initial)
        for e in (2, 4):
            q = engine.shared_state(semantic[e])
            prior = np.asarray(engine.head["prior"])
            expected = simplex((prior + v[22] * (q - prior) + v[:10])[None])[0]
            np.testing.assert_allclose(states[e], expected, atol=1e-15)
            wrong = np.asarray(engine.history_head["prior"])
            wrong = simplex((wrong + v[22] * (q - wrong) + v[:10])[None])[0]
            self.assertGreater(np.max(abs(states[e] - wrong)), 0.1)
        low, high = copy.deepcopy(p), copy.deepcopy(p)
        low["vector"][22], high["vector"][22] = 0.0, 2.0
        a = engine.predict(dict(text="我回家。"), low)
        b = engine.predict(dict(text="我回家。"), high)
        self.assertEqual(a["initial_emotion"], b["initial_emotion"])
        self.assertNotEqual(a["emotion_matrix"], b["emotion_matrix"])

    def test_layout_and_range_prevent_silent_old23_reuse(self):
        p = self.profile(2.0)
        validate_profile(p, "qwen3.5:9b", "test")
        invalid = [p["vector"], dict(p, profile_kind="old_research23")]
        for field in ("profile_kind", "emotion_order", "parameter_order"):
            q = copy.deepcopy(p)
            q.pop(field)
            invalid.append(q)
        for index, value in ((22, 2.01), (22, -0.01), (10, 0.5), (20, 1.1), (22, True)):
            q = copy.deepcopy(p)
            q["vector"][index] = value
            invalid.append(q)
        for q in invalid:
            with self.assertRaises(ValueError):
                validate_profile(q, "qwen3.5:9b", "test")

    def test_H_collector_reuses_actual_observations_once(self):
        engine, store = self.engine()
        history = dict(
            emotion_order=EMOTIONS,
            history=[
                dict(
                    text="我回家。",
                    emotion_observations=[
                        dict(end_char=4, values=[0.2] + [0.0] * 8 + [0.8]),
                        dict(end_char=0, values=[0.0] * 9 + [1.0]),
                    ],
                )
            ],
        )
        collected = []
        a = engine.generate_profile(history, _emotion_observations=collected)
        self.assertEqual(len(store.visible), 2)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0][0], [4, 0])
        b = engine.generate_profile(history)
        self.assertEqual(a, b)
        self.assertEqual(len(store.visible), 4)


if __name__ == "__main__":
    unittest.main()
