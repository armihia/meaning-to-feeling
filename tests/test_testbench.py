import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest.mock import call, patch

import numpy as np

from meaning_to_feeling.core import CompositeResponseModel
from meaning_to_feeling.prompts import (
    DEFAULT_MODEL,
    DYNAMIC_PARAMETERS,
    DYNAMIC_PROFILE_KIND,
    EMOTIONS,
)
from meaning_to_feeling.testbench import Bench, BusyError, handler_for
from http.server import ThreadingHTTPServer


class Store:
    semantic_interface = "nine-emotion-activation-composition-v1"

    def request(self, value):
        return dict(message=dict(content=json.dumps(dict(major=[1], minor=[])))), 0.0

    constrained_request = request

    def one(self, row):
        values = [0.0] * 10
        values[0] = 0.2 if row["prefix"] else 0.0
        values[-1] = 1.0 - values[0]
        return dict(
            emotions={e: "absent" for e in EMOTIONS[:-1]}, factor_semantic=values
        ), dict(cache_hit=False)

    def state_readout(self, answer):
        return np.asarray(answer["factor_semantic"])

    def close(self):
        pass


class FakeModel:
    enters = 0
    exits = 0
    release = None
    fail = False

    def __init__(self, weights, work_dir, port):
        parameters = json.loads(
            (Path(weights) / "model.json").read_text(encoding="utf-8")
        )
        self.engine = CompositeResponseModel(
            Store(), parameters["heads"]["query"], parameters["heads"]["history"]
        )
        self.runtime = SimpleNamespace(process=None, close=lambda: None)

    def __enter__(self):
        type(self).enters += 1
        return self

    def __exit__(self, *_):
        type(self).exits += 1

    def predict(self, value, profile, *, segmentation_policy="detailed"):
        if type(self).release is not None:
            type(self).release.wait(5)
        if type(self).fail:
            raise RuntimeError("本次模型回复无效")
        return self.engine.predict(
            value, profile, segmentation_policy=segmentation_policy
        )

    def generate_profile(self, value, *, strategy=None):
        if strategy == "history_mean":
            from meaning_to_feeling.history_mean import generate_profile

            return generate_profile(self.engine, value)
        return self.engine.generate_profile(value)


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="meaning_testbench_")
        self.root = Path(self.temp.name)
        prior = [0.0] * 9 + [1.0]
        head = dict(
            name="test",
            type="trace_affine",
            semantic_model=DEFAULT_MODEL,
            semantic_interface=Store.semantic_interface,
            emotion_order=EMOTIONS,
            prior=prior,
            semantic_mean=prior,
            gain=1.0,
        )
        (self.root / "model.json").write_text(
            json.dumps(dict(model=DEFAULT_MODEL, heads=dict(query=head, history=head))),
            encoding="utf-8",
        )
        FakeModel.enters = FakeModel.exits = 0
        FakeModel.release = None
        FakeModel.fail = False
        self.bench = Bench(self.root, self.root / "runtime", model_factory=FakeModel)
        self.payload = dict(
            kind="predict",
            input=dict(
                text="我终于回家了。", context_text="工作结束。", target_character="我"
            ),
            profiles=[dict(preset_id="shared")],
        )

    def tearDown(self):
        if FakeModel.release is not None:
            FakeModel.release.set()
        self.bench.close()
        self.temp.cleanup()

    def finished(self, job):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            value = self.bench.job(job["id"])
            if value["state"] in ["complete", "failed"]:
                return value
            time.sleep(0.01)
        self.fail("任务未在预期时间内结束")

    def test_compare_reuses_loaded_model_and_profile_is_immutable(self):
        payload = copy.deepcopy(self.payload)
        payload["profiles"].append(
            dict(value=[0.1] + [0.0] * 8 + [-0.1, 0.0, 0.0], name="我的向量")
        )
        original = copy.deepcopy(payload)
        result = self.finished(self.bench.submit(payload))
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["result"]["runs"]), 2)
        self.assertEqual(payload, original)
        self.finished(self.bench.submit(self.payload))
        self.assertEqual(FakeModel.enters, 1)
        self.finished(self.bench.submit(dict(kind="unload")))
        self.assertFalse(self.bench.status()["loaded"])
        self.assertEqual(FakeModel.exits, 1)

    def test_invalid_vector_rejected_before_model_load(self):
        for vector in [[0.0] * 128, [float("nan")] + [0.0] * 11, [1.0] * 12]:
            payload = copy.deepcopy(self.payload)
            payload["profiles"] = [dict(value=vector)]
            with self.assertRaises(ValueError):
                self.bench.submit(payload)
        self.assertEqual(FakeModel.enters, 0)
        self.assertFalse(self.bench.status()["busy"])

    def test_bootstrap_keeps_original_dimension_and_announces_optional_format(self):
        data = self.bench.bootstrap()
        self.assertEqual(data["dimensions"], 12)
        self.assertEqual(data["supported_dimensions"], [12, 22, 23])
        self.assertIn(DYNAMIC_PROFILE_KIND, data["supported_profile_kinds"])

    def test_complete_dynamic_profile_can_compare_and_is_kept_for_export(self):
        profile = dict(
            profile_kind=DYNAMIC_PROFILE_KIND,
            semantic_head="test",
            emotion_order=EMOTIONS,
            parameter_order=DYNAMIC_PARAMETERS,
            vector=[0.1]
            + [0.0] * 8
            + [-0.1]
            + [0.7]
            + [0.0] * 8
            + [-0.7, -0.2, 0.3, 1.3],
        )
        before = copy.deepcopy(profile)
        payload = copy.deepcopy(self.payload)
        payload["profiles"].append(dict(value=profile, name="正文变化候选"))
        job = self.finished(self.bench.submit(payload))
        self.assertEqual(job["state"], "complete", job.get("error"))
        run = job["result"]["runs"][1]
        self.assertEqual(run["profile"], before)
        self.assertEqual(run["prediction"]["personality_vector"], profile["vector"])
        self.assertEqual(profile, before)

    def test_bare_dynamic_array_requires_complete_profile_before_loading(self):
        for value in [[0.0] * 22 + [1.0], dict(vector=[0.0] * 22 + [1.0])]:
            payload = copy.deepcopy(self.payload)
            payload["profiles"] = [dict(value=value)]
            with self.assertRaisesRegex(ValueError, "profile_kind|完整"):
                self.bench.submit(payload)
        self.assertEqual(FakeModel.enters, 0)

    def test_only_one_gpu_job_is_accepted(self):
        FakeModel.release = threading.Event()
        job = self.bench.submit(self.payload)
        with self.assertRaises(BusyError):
            self.bench.submit(self.payload)
        FakeModel.release.set()
        self.assertEqual(self.finished(job)["state"], "complete")

    def test_failure_clears_busy_state_and_next_request_can_run(self):
        FakeModel.fail = True
        failed = self.finished(self.bench.submit(self.payload))
        self.assertEqual(failed["state"], "failed")
        self.assertIn("回复无效", failed["error"])
        self.assertFalse(self.bench.status()["busy"])
        FakeModel.fail = False
        self.assertEqual(
            self.finished(self.bench.submit(self.payload))["state"], "complete"
        )

    def test_history_generation_returns_saved_vector(self):
        history = self.bench.bootstrap()["history_example"]
        result = self.finished(self.bench.submit(dict(kind="profile", history=history)))
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["result"]["profile"]["vector"]), 12)
        self.assertEqual(result["result"]["profile"]["semantic_head"], "test")
        self.assertEqual(result["result"]["profile"]["history_source"], "illustrative")
        self.assertEqual(result["result"]["profile_generator"], "original")

    def test_history_mean_can_generate_export_and_predict(self):
        generated = self.finished(self.bench.submit(dict(
            kind="profile", history=self.bench.bootstrap()["history_example"],
            profile_generator="history_mean",
        )))
        self.assertEqual(generated["state"], "complete", generated.get("error"))
        profile = generated["result"]["profile"]
        self.assertEqual(len(profile["vector"]), 22)
        payload = copy.deepcopy(self.payload)
        payload["profiles"] = [dict(value=profile)]
        predicted = self.finished(self.bench.submit(payload))
        self.assertEqual(predicted["state"], "complete", predicted.get("error"))
        result = predicted["result"]["runs"][0]["prediction"]
        self.assertEqual(result["personalization_mode"], "phase_history_mean")
        self.assertIn("phase_history_trials", result["emotion_calibration"])

    def test_cutoff_all_excluded_profile_exports_and_predicts(self):
        history = self.bench.bootstrap()["history_example"]
        completed_later = [
            dict(row, finished_at="2026-11-19T01:01:00Z") for row in history["history"]
        ]
        generated = self.finished(self.bench.submit(dict(
            kind="profile", history=dict(history, history=completed_later),
            profile_generator="history_mean", as_of="2026-11-19T09:00:00+08:00",
        )))
        self.assertEqual(generated["state"], "complete", generated.get("error"))
        saved = json.loads(json.dumps(generated["result"]["profile"]))
        self.assertEqual(saved["history_trials"], 0)
        self.assertEqual(saved["sources"], [])
        self.assertEqual(saved["phase_history_trials"], [0, 0])
        self.assertEqual(saved["history_time_scope"]["after_cutoff_trials"], len(completed_later))
        payload = dict(self.payload, profiles=[dict(value=saved), dict(preset_id="shared")])
        result = self.finished(self.bench.submit(payload))
        self.assertEqual(result["state"], "complete", result.get("error"))
        personal, shared = [row["prediction"] for row in result["result"]["runs"]]
        for field in ("units", "initial_emotion", "emotion_matrix", "delta_matrix"):
            self.assertEqual(personal[field], shared[field], field)

    def test_unconfigured_generator_is_visible_and_rejected_before_loading(self):
        generators = self.bench.bootstrap()["profile_generators"]
        self.assertEqual(
            [g["id"] for g in generators], ["history_mean", "original", "reliability", "body_gain"]
        )
        self.assertEqual([g["configured"] for g in generators], [True, True, False, False])
        history = self.bench.bootstrap()["history_example"]
        for generator in ["reliability", "body_gain"]:
            with self.assertRaisesRegex(ValueError, "未配置"):
                self.bench.submit(
                    dict(kind="profile", history=history, profile_generator=generator)
                )
        self.assertEqual(FakeModel.enters, 0)

    def test_selected_generator_reuses_model_and_original_stays_default(self):
        calls = []

        def generate(engine, history):
            calls.append(engine)
            return dict(engine.generate_profile(history), generator="测试候选")

        self.bench.profile_calibrations["body_gain"] = SimpleNamespace(
            generate_profile=generate
        )
        history = self.bench.bootstrap()["history_example"]
        candidate = self.finished(
            self.bench.submit(
                dict(kind="profile", history=history, profile_generator="body_gain")
            )
        )
        original = self.finished(
            self.bench.submit(dict(kind="profile", history=history))
        )
        self.assertEqual(candidate["state"], "complete", candidate.get("error"))
        self.assertEqual(candidate["result"]["profile"]["generator"], "测试候选")
        self.assertEqual(original["result"]["profile_generator"], "original")
        self.assertEqual(calls, [self.bench.model.engine])
        self.assertEqual(FakeModel.enters, 1)

    def test_gain_configuration_requires_body_prior(self):
        with self.assertRaisesRegex(ValueError, "calibration-prior"):
            Bench(self.root, self.root / "other", gain_prior=self.root / "gain.json")

    def test_configured_candidates_share_the_explicit_body_prior(self):
        body, gain = self.root / "body.npz", self.root / "gain.json"
        with patch(
            "meaning_to_feeling.reliability.ReliabilityCalibration"
        ) as calibration:
            bench = Bench(
                self.root,
                self.root / "configured",
                model_factory=FakeModel,
                calibration_prior=body,
                gain_prior=gain,
            )
            try:
                self.assertEqual(
                    calibration.call_args_list,
                    [call(body, "test"), call(body, "test", gain_prior=gain)],
                )
                self.assertTrue(
                    all(
                        g["configured"] for g in bench.bootstrap()["profile_generators"]
                    )
                )
                self.assertEqual(FakeModel.enters, 0)
            finally:
                bench.close()

    def test_local_profiles_must_match_the_numerical_head(self):
        path = self.root / "wrong.json"
        profile = dict(
            id="reader_a",
            name="A",
            source="human_calibrated",
            source_label="真人历史校准",
            description="测试",
            profile=dict(vector=[0.0] * 12, semantic_head="different"),
        )
        path.write_text(json.dumps([profile]), encoding="utf-8")
        with self.assertRaises(ValueError):
            Bench(self.root, self.root / "other", path, model_factory=FakeModel)

    def test_http_serves_only_ui_and_rejects_foreign_origin_and_nan(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.bench))
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
        )
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(url + "/api/bootstrap") as stream:
                data = json.load(stream)
            self.assertGreaterEqual(len(data["profiles"]), 3)
            self.assertTrue(
                all(p["source"] != "human_calibrated" for p in data["profiles"])
            )
            with opener.open(url + "/app.js") as stream:
                self.assertIn("renderMatrix", stream.read().decode("utf-8"))
            for path in ["/.env", "/model.gguf", "/../model.json"]:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    opener.open(url + path)
                self.assertEqual(raised.exception.code, 404)
            request = urllib.request.Request(
                url + "/api/jobs",
                data=json.dumps(self.payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://unrelated.example",
                },
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                opener.open(request)
            self.assertEqual(raised.exception.code, 403)
            request = urllib.request.Request(
                url + "/api/jobs",
                data=b'{"kind":"predict","value":NaN}',
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                opener.open(request)
            self.assertEqual(raised.exception.code, 400)
            self.assertTrue(data["supports_history_time_cutoff"])
            request = urllib.request.Request(
                url + "/api/jobs",
                data=json.dumps(dict(
                    kind="profile", history=data["history_example"], as_of="2026-11-19T09:00",
                )).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                opener.open(request)
            self.assertEqual(raised.exception.code, 400)
            self.assertIn("带时区", json.load(raised.exception)["error"])
            self.assertEqual(FakeModel.enters, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)


if __name__ == "__main__":
    unittest.main()
