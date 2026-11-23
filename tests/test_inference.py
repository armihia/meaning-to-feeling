import copy
import json
import unittest

import numpy as np

from meaning_to_feeling.backend import activation_payload, grade_payload
from meaning_to_feeling.core import (
    CompositeResponseModel,
    candidate_ends,
    text_row,
    validate_profile,
)
from meaning_to_feeling.prompts import EMOTIONS


class Store:
    semantic_interface = "nine-emotion-activation-composition-v1"

    def __init__(self):
        self.visible = []

    def request(self, request):
        return dict(message=dict(content=json.dumps(dict(major=[1], minor=[2])))), 0.0

    def one(self, row):
        self.visible.append(dict(row))
        vector = np.zeros(10)
        vector[0] = 0.2 if row["prefix"] else 0.0
        vector[-1] = 1.0 - vector[0]
        return dict(
            emotions={e: "absent" for e in EMOTIONS[:-1]},
            factor_semantic=vector.tolist(),
        ), dict(cache_hit=False)

    def state_readout(self, value):
        return np.asarray(value["factor_semantic"])

    def close(self):
        pass


class InferenceTests(unittest.TestCase):
    def engine(self):
        prior = [0.0] * 9 + [1.0]
        head = dict(
            type="trace_affine",
            semantic_model="qwen3.5:9b",
            semantic_interface=Store.semantic_interface,
            emotion_order=EMOTIONS,
            prior=prior,
            semantic_mean=prior,
            gain=1.0,
            name="test",
        )
        store = Store()
        return CompositeResponseModel(store, head, head), store

    def test_exact_text_and_prefix_visibility(self):
        engine, store = self.engine()
        text = "我🙂，\n终于到家。"
        value = dict(
            text=text,
            context_text="等待。",
            target_character="我",
            candidate_ends=[3, 6, len(text)],
        )
        profile = dict(
            vector=[0.05] + [0.0] * 8 + [-0.05, 0.0, 0.0], semantic_head="test"
        )
        old = copy.deepcopy(profile)
        result = engine.predict(value, profile)
        self.assertEqual("".join(u["text"] for u in result["units"]), text)
        self.assertEqual(profile, old)
        for row in store.visible:
            self.assertTrue(text.startswith(row["prefix"]))
            self.assertEqual(
                set(row), {"character", "perspective", "context", "prefix"}
            )
        states = np.asarray([result["initial_emotion"], *result["emotion_matrix"]])
        np.testing.assert_allclose(states.sum(1), 1)
        self.assertGreaterEqual(float(states.min()), 0.0)
        np.testing.assert_allclose(result["delta_matrix"], np.diff(states, axis=0))

    def test_profile_rejects_wrong_head_and_nonfinite(self):
        with self.assertRaises(ValueError):
            validate_profile(
                dict(vector=[0.0] * 12, semantic_head="other"), "qwen3.5:9b", "test"
            )
        with self.assertRaises(ValueError):
            validate_profile([float("nan")] + [0.0] * 11, "qwen3.5:9b")

    def test_candidates_reject_uncovered_or_duplicate_positions(self):
        row = text_row(dict(text="我的句子。"))
        for ends in [[1, 1, 5], [1, 2], [False, 5]]:
            with self.assertRaises(ValueError):
                candidate_ends(row, ends)

    def test_backend_sends_visible_material_only(self):
        row = dict(
            character="我",
            context="背景",
            prefix="我正在",
            person="private-id",
            labels=[1],
            questionnaire=[2],
        )
        grade = json.loads(grade_payload(row)["messages"][1]["content"])
        self.assertEqual(
            grade, dict(character="我", background="背景", visible_text="我正在")
        )
        activation = json.loads(
            activation_payload(row, {e: "absent" for e in EMOTIONS[:-1]})["messages"][
                1
            ]["content"]
        )
        self.assertEqual(
            set(activation), {"character", "background", "visible_text", "emotions"}
        )


if __name__ == "__main__":
    unittest.main()
