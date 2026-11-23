import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from meaning_to_feeling.__main__ import main


class ConfigurationTests(unittest.TestCase):
    def invoke(self, action, environment, profile=None, options=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output_path = root / "input.json", root / "output.json"
            value = dict(history=[]) if action == "profile" else dict(text="我到家了。")
            input_path.write_text(json.dumps(value), encoding="utf-8")
            argv = [
                "mtf",
                action,
                "--weights",
                str(root),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                *options,
            ]
            if profile is not None:
                saved = root / "profile.json"
                saved.write_text(json.dumps(profile), encoding="utf-8")
                argv.extend(["--profile", str(saved)])
            with (
                patch.dict(os.environ, environment),
                patch("sys.argv", argv),
                patch("meaning_to_feeling.__main__.load_dotenv"),
                patch("meaning_to_feeling.__main__.Model") as constructor,
                patch("builtins.print"),
            ):
                model = constructor.return_value.__enter__.return_value
                model.generate_profile.return_value = dict(vector=[0.0] * 12)
                model.predict.return_value = dict(units=[])
                main()
                return constructor.call_args.kwargs, model

    def test_empty_optional_env_keeps_original_generator(self):
        kwargs, _ = self.invoke(
            "profile", dict(MTF_CALIBRATION_PRIOR="", MTF_GAIN_PRIOR="")
        )
        self.assertIsNone(kwargs["calibration_prior"])
        self.assertIsNone(kwargs["gain_prior"])

    def test_testbench_env_does_not_change_cli_default(self):
        kwargs, _ = self.invoke(
            "profile",
            dict(MTF_CALIBRATION_PRIOR="body.npz", MTF_GAIN_PRIOR="gain.json"),
        )
        self.assertIsNone(kwargs["calibration_prior"])
        self.assertIsNone(kwargs["gain_prior"])

    def test_saved_profile_predict_does_not_load_generator_priors(self):
        profile = dict(vector=[0.0] * 12)
        kwargs, model = self.invoke(
            "predict",
            dict(
                MTF_CALIBRATION_PRIOR="missing_body.npz",
                MTF_GAIN_PRIOR="missing_gain.json",
            ),
            profile,
        )
        self.assertIsNone(kwargs["calibration_prior"])
        self.assertIsNone(kwargs["gain_prior"])
        self.assertEqual(model.predict.call_args.args[1], profile)

    def test_profile_forwards_cutoff_and_strategy(self):
        cutoff = "2026-11-19T09:00:00+08:00"
        _, model = self.invoke(
            "profile", {}, options=("--as-of", cutoff, "--profile-strategy", "history_mean"),
        )
        model.generate_profile.assert_called_once_with(
            dict(history=[]), as_of=cutoff, strategy="history_mean",
        )


if __name__ == "__main__":
    unittest.main()
