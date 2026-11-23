import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HEADS = ROOT / "model_weights/shared"
STORY_INSTRUCTION = (
    "Represent the current understanding and feelings of the specified person, "
    "using only the given background and visible text. Preserve changes in meaning and affect."
)
ROCLING_INSTRUCTION = "Estimate the emotional valence and arousal expressed by a Chinese doctor's self-reflection."


def embed(texts, model_dir, device, precision):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="left", local_files_only=True
    )
    encoder = (
        AutoModel.from_pretrained(
            model_dir,
            torch_dtype=getattr(torch, precision),
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    input_ids = tokenizer(texts, truncation=False)["input_ids"]
    if max(map(len, input_ids)) > encoder.config.max_position_embeddings:
        raise ValueError("输入长度超过 Qwen3-Embedding-4B 的上下文限制")
    order = sorted(range(len(texts)), key=lambda i: len(input_ids[i]))
    features = np.empty((len(texts), encoder.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), 8):
            batch = order[start : start + 8]
            tokens = tokenizer.pad(
                [{"input_ids": input_ids[i]} for i in batch], return_tensors="pt"
            ).to(device)
            state = (
                encoder(**tokens, use_cache=False)
                .last_hidden_state[:, -1]
                .float()
                .cpu()
            )
            features[batch] = torch.nn.functional.normalize(state).numpy()
    return features


def story_prompt(row):
    return (
        f"Instruct: {STORY_INSTRUCTION}\nQuery: perspective=character\n"
        f"人物：{row['target_character']}。语境：{row.get('context_text', '')}\n"
        f"已读正文：{row['text']}"
    )


def rocling_prompt(text):
    return f"Instruct: {ROCLING_INSTRUCTION}\nQuery: {text}"


def story_scores(features, head_path=HEADS / "storycommonsense.npz"):
    with np.load(head_path, allow_pickle=False) as head:
        logits = features @ head["coef"].T + head["intercept"]
        probabilities = np.exp(-np.logaddexp(0, -logits))
        return (
            head["emotions"].tolist(),
            probabilities,
            probabilities >= head["threshold"],
        )


def rocling_scores(features, support_path, head_path=HEADS / "rocling.npz"):
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import normalize

    with (
        np.load(support_path, allow_pickle=False) as support,
        np.load(head_path, allow_pickle=False) as head,
    ):
        if not np.array_equal(head["ids"], support["ids"]):
            raise ValueError("开发文本 ID 或行顺序与 ROCLING 权重不一致")
        kernel = rbf_kernel(
            normalize(features.astype(float)),
            normalize(support["embedding"].astype(float)),
            gamma=float(head["gamma"]),
        )
        return np.clip(head["mean"] + kernel @ head["dual_coef"], 1, 9)


def input_rows(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = value if isinstance(value, list) else [value]
    if not rows or any(
        not isinstance(row, dict) or not row.get("text") for row in rows
    ):
        raise ValueError("输入须包含至少一条带 text 字段的 JSON 记录")
    return rows


def prepare_rocling(path, output, encode, head_path=HEADS / "rocling.npz"):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    ids = [row["ID"] for row in rows]
    with np.load(head_path, allow_pickle=False) as head:
        if ids != head["ids"].tolist():
            raise ValueError("请使用 README 指定版本的 DSAMST-ValidationSet.csv")
    features = encode([rocling_prompt(row["Text"]) for row in rows])
    with output.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), embedding=features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task", choices=["storycommonsense", "prepare-rocling", "rocling"]
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--encoder", type=Path, default=ROOT / "models/qwen3-embedding-4b"
    )
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--precision", choices=["float32", "bfloat16"])
    parser.add_argument(
        "--support", type=Path, default=ROOT / ".runtime/rocling_support.npz"
    )
    args = parser.parse_args()
    precision = args.precision or (
        "float32" if args.task == "storycommonsense" else "bfloat16"
    )

    def encode(prompts):
        return embed(prompts, args.encoder, args.device, precision)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "prepare-rocling":
        prepare_rocling(args.input, args.output, encode)
    else:
        rows = input_rows(args.input)
        if args.task == "storycommonsense":
            features = encode([story_prompt(row) for row in rows])
            names, probs, labels = story_scores(features)
            result = dict(
                emotion_order=names,
                probabilities=probs.tolist(),
                labels=labels.tolist(),
            )
        else:
            features = encode([rocling_prompt(row["text"]) for row in rows])
            scores = rocling_scores(features, args.support)
            result = dict(axes=["Valence", "Arousal"], scores=scores.tolist())
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(args.output)


if __name__ == "__main__":
    main()
