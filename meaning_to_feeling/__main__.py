import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from . import DEFAULT_WEIGHTS, Model
from .history_scope import parse_cutoff


def main():
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["profile", "predict"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-strategy", choices=["original", "history_mean"])
    parser.add_argument("--as-of", help="生成档案时的开读时间，须带时区")
    parser.add_argument(
        "--weights", type=Path, default=os.getenv("MTF_WEIGHTS") or DEFAULT_WEIGHTS
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".runtime"))
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--calibration-prior", type=Path)
    parser.add_argument("--gain-prior", type=Path)
    parser.add_argument(
        "--segmentation-policy", choices=["detailed", "legacy"], default="detailed"
    )
    args = parser.parse_args()
    profile_options = {}
    if args.profile_strategy is not None:
        profile_options["strategy"] = args.profile_strategy
    if args.as_of is not None:
        if args.action != "profile":
            parser.error("--as-of用于生成档案的profile命令")
        try:
            parse_cutoff(args.as_of)
        except ValueError as exc:
            parser.error(str(exc))
        profile_options["as_of"] = args.as_of
    value = json.loads(args.input.read_text(encoding="utf-8-sig"))
    profile = (
        json.loads(args.profile.read_text(encoding="utf-8-sig"))
        if args.profile
        else None
    )
    if args.action == "predict" and (
        profile is not None or "personality_vector" in value
    ):
        args.calibration_prior = args.gain_prior = None
    if args.gain_prior is not None and args.calibration_prior is None:
        parser.error("--gain-prior需要同时提供--calibration-prior")
    with Model(
        args.weights,
        args.work_dir,
        args.port,
        calibration_prior=args.calibration_prior,
        gain_prior=args.gain_prior,
    ) as model:
        result = (
            model.generate_profile(value, **profile_options)
            if args.action == "profile"
            else model.predict(
                value, profile, segmentation_policy=args.segmentation_policy
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(dict(saved=str(args.output), action=args.action), ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
