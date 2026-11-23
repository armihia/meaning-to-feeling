import json
import unittest

import numpy as np

from meaning_to_feeling.core import (
    CompositeResponseModel,
    detailed_candidates,
    punctuation_boundaries,
    text_row,
)
from meaning_to_feeling.prompts import EMOTIONS


class SelectedStore:
    semantic_interface = "nine-emotion-activation-composition-v1"

    def __init__(self, major=(), minor=()):
        self.major, self.minor = list(major), list(minor)
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        return dict(
            message=dict(content=json.dumps(dict(major=self.major, minor=self.minor)))
        ), 0.0

    def one(self, row):
        return dict(
            emotions={e: "absent" for e in EMOTIONS[:-1]},
            factor_semantic=[0.0] * 9 + [1.0],
        ), dict(cache_hit=True)

    def state_readout(self, answer):
        return np.asarray(answer["factor_semantic"])


class SegmentationPolicyTests(unittest.TestCase):
    def engine(self, major=(), minor=()):
        store = SelectedStore(major, minor)
        head = dict(
            name="test",
            type="trace_affine",
            semantic_model="qwen3.5:9b",
            semantic_interface=store.semantic_interface,
            emotion_order=EMOTIONS,
            prior=[0.0] * 9 + [1.0],
            semantic_mean=[0.0] * 9 + [1.0],
            gain=1.0,
        )
        return CompositeResponseModel(store, head, head), store

    def test_punctuation_survives_empty_model_nomination(self):
        engine, _ = self.engine()
        text = "我到家了，才发现灯还亮着；我松了口气。"
        result = engine.predict(dict(text=text))
        expected = [text.index("，") + 1, text.index("；") + 1]
        self.assertEqual(result["segmentation"]["fixed_punctuation_ends"], expected)
        self.assertEqual(
            [u["end_char"] for u in result["units"]], expected + [len(text)]
        )
        self.assertEqual("".join(u["text"] for u in result["units"]), text)
        self.assertTrue(
            all(
                any(not c.isspace() and c not in "，；。" for c in u["text"])
                for u in result["units"]
            )
        )

    def test_all_semantic_minor_boundaries_survive_without_count_cap(self):
        engine, _ = self.engine([2], [1, 3, 4])
        text = "我想休息却还得继续工作"
        value = dict(text=text, candidate_ends=[2, 4, 6, 8, len(text)])
        vector = [0.0] * 10 + [-0.9, -0.9]
        result = engine.predict(value, vector)
        self.assertEqual(
            [u["end_char"] for u in result["units"]], value["candidate_ends"]
        )
        self.assertIsNone(result["segmentation"]["requested_internal_boundaries"])
        self.assertFalse(result["segmentation"]["count_limit_applied"])

    def test_legacy_retains_original_count_control(self):
        engine, _ = self.engine([2], [1, 3, 4])
        text = "我想休息却还得继续工作"
        value = dict(text=text, candidate_ends=[2, 4, 6, 8, len(text)])
        result = engine.predict(
            value, [0.0] * 10 + [-0.9, -0.9], segmentation_policy="legacy"
        )
        self.assertEqual(len(result["units"]), 1)
        self.assertTrue(result["segmentation"]["count_limit_applied"])

    def test_reader_condition_is_sent_to_specific_boundary_request(self):
        engine, _ = self.engine()
        row = text_row(dict(text="我终于回家了。"))
        first = np.zeros(12)
        second = first.copy()
        second[10] = 0.2
        before = engine.segmentation_request(
            row, [1, 3, len(row["text"])], reader=first
        )
        after = engine.segmentation_request(
            row, [1, 3, len(row["text"])], reader=second
        )
        self.assertNotEqual(
            before["messages"][1]["content"], after["messages"][1]["content"]
        )
        value = json.loads(after["messages"][1]["content"])
        self.assertEqual(
            value["reader_boundary_profile"]["nonpunctuation_rate_shift"], 0.2
        )
        self.assertNotIn("needed", value)
        self.assertNotIn("block_count", value)

    def test_emotion_only_difference_is_not_invented_as_boundary_information(self):
        engine, store = self.engine([1])
        value = dict(text="我终于回家了。", candidate_ends=[1, 3, 7])
        a = engine.predict(value, [0.0] * 12)
        b = engine.predict(value, [0.1] + [0.0] * 8 + [-0.1, 0.0, 0.0])
        self.assertEqual(a["units"], b["units"])
        self.assertEqual(store.requests[0]["messages"], store.requests[1]["messages"])
        self.assertNotEqual(a["emotion_matrix"], b["emotion_matrix"])

    def test_numbers_and_punctuation_runs_are_not_broken_into_symbols(self):
        text = "我在12:30付了3.14元，随后回家了。"
        self.assertEqual(punctuation_boundaries(text), [text.index("，") + 1])
        quoted = "我问：“真的？！”  然后笑了。"
        self.assertEqual(
            punctuation_boundaries(quoted),
            [quoted.index("：") + 1, quoted.index("然后")],
        )

    def test_punctuation_only_candidate_span_attaches_to_previous_word(self):
        text = "我🙂，\n终于到家。"
        fixed = punctuation_boundaries(text)
        ends = detailed_candidates(text, [3, 6, len(text)], fixed)
        self.assertEqual(ends, [4, 6, len(text)])
        self.assertTrue(set(fixed) <= set(ends))


if __name__ == "__main__":
    unittest.main()
