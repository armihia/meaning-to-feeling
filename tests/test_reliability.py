import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from meaning_to_feeling import Model
from meaning_to_feeling.core import CompositeResponseModel
from meaning_to_feeling.prompts import (
    DYNAMIC_PARAMETERS,
    DYNAMIC_PROFILE_KIND,
    EMOTIONS,
)
from meaning_to_feeling.reliability import GainCalibration, ReliabilityCalibration


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "prior.npz"
        self.basis = np.linalg.qr(np.eye(10)[:, :9] - 0.1)[0]
        self.mean = np.linspace(-0.01, 0.01, 9)
        self.between = np.eye(9) * 0.03 + np.ones((9, 9)) * 0.001
        self.noise = np.eye(9) * 0.04
        np.savez(
            self.path,
            mean=self.mean,
            between=self.between,
            noise=self.noise,
            composition_basis=self.basis,
            emotion_order=np.asarray(EMOTIONS),
            history_head_name=np.asarray("test"),
        )
        self.calibration = ReliabilityCalibration(self.path, "test")

    def test_posterior_equals_dense_conditioning_and_sums_to_zero(self):
        residual = np.asarray(
            [[0.1] + [0.0] * 8 + [-0.1], [-0.1, 0.2] + [0.0] * 7 + [-0.1]]
        )
        latent = residual @ self.basis
        h = np.tile(np.eye(9), (2, 1))
        cov = h @ self.between @ h.T + np.kron(np.eye(2), self.noise)
        delta = (
            self.between @ h.T @ np.linalg.solve(cov, (latent - self.mean).reshape(-1))
        )
        expected = self.basis @ (self.mean + delta)
        actual, uncertainty = self.calibration.infer(residual)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertLess(abs(actual.sum()), 1e-14)
        self.assertTrue((uncertainty >= 0).all())

    def test_no_history_uses_population_prior_and_wrong_head_is_rejected(self):
        offset, variance = self.calibration.infer([])
        np.testing.assert_allclose(offset, self.basis @ self.mean, atol=1e-14)
        np.testing.assert_allclose(
            variance, np.diag(self.basis @ self.between @ self.basis.T), atol=1e-14
        )
        with self.assertRaises(ValueError):
            ReliabilityCalibration(self.path, "different")

    def test_inline_vector_keeps_existing_precedence(self):
        model = Model.__new__(Model)
        model.calibration = self.calibration
        model.engine = SimpleNamespace(
            predict=lambda value, profile, **kwargs: dict(value=value, profile=profile)
        )
        result = model.predict(dict(text="我回家了。", personality_vector=[0.0] * 12))
        self.assertIsNone(result["profile"])
        self.assertEqual(result["value"]["personality_vector"], [0.0] * 12)

    def test_G_keeps_boundary_coordinates_and_ignores_query_field(self):
        q = np.asarray([0.1] + [0.0] * 8 + [0.9])
        engine = SimpleNamespace(
            generate_profile=lambda value: dict(vector=[0.0] * 10 + [0.2, -0.1]),
            states=lambda row, ends, vector: ({}, {e: q for e in ends}, {}),
            shared_state=lambda semantic, history: semantic,
        )
        history = [
            dict(
                text="我回家。",
                context_text="",
                target_character="我",
                emotion_observations=[
                    dict(end_char=2, values=[0.3] + [0.0] * 8 + [0.7])
                ],
            )
        ]
        a = self.calibration.generate_profile(engine, dict(history=history))
        b = self.calibration.generate_profile(
            engine, dict(history=history, query_answer=[1.0] * 10)
        )
        self.assertEqual(a["vector"][10:], [0.2, -0.1])
        self.assertEqual(a["vector"], b["vector"])

    def test_background_only_history_is_not_a_body_observation(self):
        engine = SimpleNamespace(
            generate_profile=lambda value: dict(
                vector=[0.0] * 12, emotion_history_trials=1
            ),
            states=lambda *args: (_ for _ in ()).throw(
                AssertionError("开始前评分不进入正文校准")
            ),
        )
        value = dict(
            history=[
                dict(
                    text="我回家。",
                    context_text="",
                    target_character="我",
                    emotion_observations=[dict(end_char=0, values=[1.0] + [0.0] * 9)],
                )
            ]
        )
        result = self.calibration.generate_profile(engine, value)
        np.testing.assert_allclose(
            result["vector"][:10], self.basis @ self.mean, atol=1e-14
        )
        self.assertEqual(result["emotion_history_trials"], 0)
        self.assertEqual(result["reliability"]["body_observation_points"], 0)


class DynamicReliabilityTests(unittest.TestCase):
    def setUp(self):
        ReliabilityTests.setUp(self)
        self.gain_path = Path(self.temporary.name) / "gain.json"
        self.gain_noise = np.eye(9) * 0.02 + np.ones((9, 9)) * 0.003
        self.save_gain()
        self.dynamic = ReliabilityCalibration(
            self.path, "test", gain_prior=self.gain_path
        )
        self.prior = np.asarray([0.2] + [0.0] * 8 + [0.8])

    def save_gain(self, **changes):
        saved = dict(
            gain_variance=0.7,
            noise=self.gain_noise.tolist(),
            emotion_order=EMOTIONS,
            history_head_name="test",
        )
        saved.update(changes)
        self.gain_path.write_text(json.dumps(saved), encoding="utf-8")

    def engine(self, states):
        class Store:
            semantic_interface = "nine-emotion-activation-composition-v1"

            def __init__(self):
                self.visible = []

            def one(self, row):
                self.visible.append(dict(row))
                return dict(
                    state=states[len(row["prefix"])],
                    emotions={emotion: "absent" for emotion in EMOTIONS[:-1]},
                ), dict(cache_hit=False)

            def state_readout(self, answer):
                return np.asarray(answer["state"])

        store = Store()
        head = dict(
            type="trace_affine",
            semantic_model="qwen3.5:9b",
            semantic_interface=store.semantic_interface,
            emotion_order=EMOTIONS,
            prior=self.prior.tolist(),
            semantic_mean=self.prior.tolist(),
            gain=1.0,
            name="test",
        )
        return CompositeResponseModel(store, head, head), store

    def history(self, points):
        return dict(
            emotion_order=EMOTIONS,
            history=[
                dict(
                    text="我回家了。",
                    emotion_observations=[
                        dict(end_char=end, values=y) for end, y in points
                    ],
                )
            ],
        )

    def test_gain_posterior_matches_dense_scalar_conditioning(self):
        dq = np.asarray([[0.1, -0.1] + [0.0] * 8, [-0.03, 0.06, -0.03] + [0.0] * 7])
        dy = dq * np.asarray([[1.4], [0.9]])
        dx, target = dq @ self.basis, dy @ self.basis
        h = dx.reshape(-1)
        noise = np.kron(np.eye(2), self.gain_noise)
        marginal = noise + 0.7 * np.outer(h, h)
        expected = 1.0 + 0.7 * h @ np.linalg.solve(marginal, (target - dx).reshape(-1))
        expected_variance = 0.7 - 0.7**2 * h @ np.linalg.solve(marginal, h)
        gain, variance, information = self.dynamic.gain.infer(zip(dq, dy))
        self.assertAlmostEqual(gain, expected, places=14)
        self.assertAlmostEqual(variance, expected_variance, places=14)
        self.assertGreater(information, 0.0)

    def test_gain_is_clipped_to_zero_through_two(self):
        dq = np.asarray([0.1, -0.1] + [0.0] * 8)
        self.assertEqual(self.dynamic.gain.infer([(dq, 100.0 * dq)])[0], 2.0)
        self.assertEqual(self.dynamic.gain.infer([(dq, -100.0 * dq)])[0], 0.0)

    def test_empty_differences_and_zero_prior_variance_give_exact_gain_one(self):
        self.assertEqual(self.dynamic.gain.infer([]), (1.0, 0.7, 0.0))
        self.save_gain(gain_variance=0.0)
        gain_model = GainCalibration(self.gain_path, "test", self.basis)
        differences = [(np.asarray([0.1, -0.1] + [0.0] * 8), np.zeros(10))]
        gain, variance, information = gain_model.infer(differences)
        self.assertEqual((gain, variance), (1.0, 0.0))
        self.assertGreater(information, 0.0)

    def test_no_history_and_background_only_do_not_fit_body_gain(self):
        engine, store = self.engine({0: self.prior})
        for value, point_count in [
            (dict(history=[], emotion_order=EMOTIONS), 0),
            (self.history([(0, [0.9] + [0.0] * 8 + [0.1])]), 1),
        ]:
            with self.subTest(point_count=point_count):
                result = self.dynamic.generate_profile(engine, value)
                self.assertEqual(result["vector"][-1], 1.0)
                self.assertEqual(result["vector"][:10], result["vector"][10:20])
                np.testing.assert_array_equal(
                    result["vector"][:10], self.basis @ self.mean
                )
                self.assertEqual(
                    result["history_coverage"]["emotion_points"], point_count
                )
                self.assertEqual(result["history_coverage"]["body_emotion_points"], 0)
                self.assertEqual(result["history_coverage"]["body_change_trials"], 0)
        self.assertEqual(len(store.visible), 1)

    def test_initial_reference_and_body_offsets_use_their_own_calibration(self):
        q1 = [0.1] + [0.0] * 8 + [0.9]
        q2 = [0.3] + [0.0] * 8 + [0.7]
        y1 = [0.15] + [0.0] * 8 + [0.85]
        y2 = [0.45] + [0.0] * 8 + [0.55]
        engine, store = self.engine({1: q1, 4: q2})
        value = self.history([(4, y2), (1, y1)])
        original = copy.deepcopy(value)
        result = self.dynamic.generate_profile(engine, value)
        q = np.asarray(
            [
                engine.shared_state(np.asarray(q1), history=True),
                engine.shared_state(np.asarray(q2), history=True),
            ]
        )
        y = np.asarray([y1, y2])
        gain = result["vector"][-1]
        expected_initial = self.calibration.infer([(y - q).mean(0)])[0]
        expected_body = self.calibration.infer(
            [(y - (self.prior + gain * (q - self.prior))).mean(0)]
        )[0]
        self.assertNotEqual(gain, 1.0)
        np.testing.assert_array_equal(result["vector"][:10], expected_body)
        np.testing.assert_array_equal(result["vector"][10:20], expected_initial)
        self.assertFalse(np.array_equal(expected_body, expected_initial))
        self.assertEqual(result["vector"][20:22], [0.0, 0.0])
        self.assertEqual(result["parameter_order"], DYNAMIC_PARAMETERS)
        self.assertEqual(result["profile_kind"], DYNAMIC_PROFILE_KIND)
        self.assertEqual(result["history_coverage"]["body_change_trials"], 1)
        self.assertEqual(len(store.visible), 2)
        self.assertEqual(value, original)

    def test_one_average_adjacent_change_per_history_excludes_background(self):
        states = {
            end: [0.05 * end] + [0.0] * 8 + [1.0 - 0.05 * end] for end in [0, 1, 3, 5]
        }
        points = [
            (end, [0.08 * end] + [0.0] * 8 + [1.0 - 0.08 * end]) for end in [5, 0, 1, 3]
        ]
        engine, store = self.engine(states)
        result = self.dynamic.generate_profile(engine, self.history(points))
        first = engine.shared_state(np.asarray(states[1]), history=True)
        last = engine.shared_state(np.asarray(states[5]), history=True)
        expected = self.dynamic.gain.infer(
            [((last - first) / 2, (np.asarray(points[0][1]) - points[2][1]) / 2)]
        )
        self.assertEqual(result["vector"][-1], expected[0])
        self.assertEqual(result["reliability"]["body_gain"]["history_trials"], 1)
        self.assertEqual(result["history_coverage"]["body_emotion_points"], 3)
        self.assertEqual(len(store.visible), 4)

    def test_zero_variance_dynamic_offsets_equal_existing_twelve_vector(self):
        self.save_gain(gain_variance=0.0)
        dynamic = ReliabilityCalibration(self.path, "test", gain_prior=self.gain_path)
        q = {1: [0.1] + [0.0] * 8 + [0.9], 4: [0.3] + [0.0] * 8 + [0.7]}
        value = self.history(
            [(1, [0.2] + [0.0] * 8 + [0.8]), (4, [0.5] + [0.0] * 8 + [0.5])]
        )
        engine, _ = self.engine(q)
        old = self.calibration.generate_profile(engine, value)
        new = dynamic.generate_profile(engine, value)
        self.assertEqual(len(old["vector"]), 12)
        self.assertEqual(new["vector"][:10], old["vector"][:10])
        self.assertEqual(new["vector"][10:20], old["vector"][:10])
        self.assertEqual(new["vector"][20:22], old["vector"][10:12])
        self.assertEqual(new["vector"][-1], 1.0)
        self.assertNotIn("profile_kind", old)

    def test_gain_prior_validates_identity_variance_and_positive_definite_noise(self):
        invalid = [
            dict(gain_variance=-0.1),
            dict(gain_variance=True),
            dict(gain_variance=float("nan")),
            dict(history_head_name="query"),
            dict(emotion_order=EMOTIONS[::-1]),
            dict(noise=np.eye(10).tolist()),
            dict(noise=np.zeros((9, 9)).tolist()),
            dict(noise=(np.eye(9) + np.triu(np.ones((9, 9)), 1)).tolist()),
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.save_gain(**changes)
                with self.assertRaises(ValueError):
                    ReliabilityCalibration(self.path, "test", gain_prior=self.gain_path)

    def test_model_requires_body_prior_and_preserves_saved_or_inline_vector(self):
        with self.assertRaises(ValueError):
            Model("missing_weights", gain_prior=self.gain_path)
        weights = Path(self.temporary.name) / "weights"
        weights.mkdir()
        (weights / "model.json").write_text(
            json.dumps(dict(heads=dict(history=dict(name="test")))), encoding="utf-8"
        )
        with patch("meaning_to_feeling.Runtime"):
            model = Model(
                weights, calibration_prior=self.path, gain_prior=self.gain_path
            )
            self.assertIsNotNone(model.calibration.gain)
            saved_only_model = Model(weights)
        saved = dict(vector=[0.0] * 22 + [1.2], profile_kind=DYNAMIC_PROFILE_KIND)
        saved_only_model.engine = SimpleNamespace(
            predict=lambda value, profile, **kwargs: profile
        )
        self.assertIs(saved_only_model.predict(dict(text="我回家了。"), saved), saved)
        model.engine = SimpleNamespace(predict=lambda value, profile, **kwargs: profile)
        self.assertIsNone(
            model.predict(dict(text="我回家了。", personality_vector=saved["vector"]))
        )

    def test_model_without_profile_generates_empty_history_gain_one(self):
        model = Model.__new__(Model)
        model.calibration = self.dynamic
        engine, store = self.engine({})
        engine.predict = lambda value, profile, **kwargs: profile
        model.engine = engine
        result = model.predict(dict(text="我回家了。"))
        self.assertEqual(result["profile_kind"], DYNAMIC_PROFILE_KIND)
        self.assertEqual(result["vector"][-1], 1.0)
        self.assertEqual(result["vector"][:10], result["vector"][10:20])
        self.assertEqual(store.visible, [])

    def test_gain_branch_preserves_core_input_validation_and_requires_heads(self):
        engine, store = self.engine({})
        with self.assertRaises(ValueError):
            self.dynamic.generate_profile(engine, dict(history=[]))
        for head in ["head", "history_head"]:
            original = getattr(engine, head)
            setattr(engine, head, None)
            with self.subTest(head=head), self.assertRaises(ValueError):
                self.dynamic.generate_profile(
                    engine, dict(history=[], emotion_order=EMOTIONS)
                )
            setattr(engine, head, original)
        self.assertEqual(store.visible, [])

    def test_cli_passes_gain_and_body_priors_to_model(self):
        from meaning_to_feeling.__main__ import main

        input_path = Path(self.temporary.name) / "input.json"
        output_path = Path(self.temporary.name) / "profile.json"
        value = dict(history=[], emotion_order=EMOTIONS)
        input_path.write_text(json.dumps(value), encoding="utf-8")
        argv = [
            "meaning_to_feeling",
            "profile",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--weights",
            self.temporary.name,
            "--calibration-prior",
            str(self.path),
            "--gain-prior",
            str(self.gain_path),
        ]
        expected = dict(vector=[0.0] * 22 + [1.0])
        with (
            patch("sys.argv", argv),
            patch("meaning_to_feeling.__main__.load_dotenv"),
            patch("meaning_to_feeling.__main__.Model") as constructor,
            patch("builtins.print"),
        ):
            model = constructor.return_value.__enter__.return_value
            model.generate_profile.return_value = expected
            main()
        self.assertEqual(constructor.call_args.kwargs["gain_prior"], self.gain_path)
        self.assertEqual(constructor.call_args.kwargs["calibration_prior"], self.path)
        model.generate_profile.assert_called_once_with(value)
        self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), expected)


if __name__ == "__main__":
    unittest.main()
