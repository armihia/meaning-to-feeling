import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import gguf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BLOB = "sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
DEFAULT_OUTPUT = ROOT / "model_weights/meaning_to_feeling_9b/model.gguf"


def default_source():
    models = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama/models")
    return models / "blobs" / SOURCE_BLOB


def normalized_fields(reader):
    architecture = gguf.Keys.General.ARCHITECTURE
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == architecture:
            continue
        value, kind = field.contents(), field.types[0]
        subtype = None
        if kind == gguf.GGUFValueType.ARRAY:
            if not value:
                if ".vision." in name:
                    continue
                raise ValueError(f"文本元数据含空数组：{name}")
            subtype = field.types[-1]
        if name == "qwen35.rope.dimension_sections":
            if len(value) == 3:
                value = [*value, 0]
        if name == "qwen35.attention.head_count_kv" and isinstance(value, list):
            positive = {count for count in value if count > 0}
            if len(positive) != 1:
                raise ValueError("全注意力层的 KV 头数不统一")
            value = positive.pop()
            kind, subtype = gguf.GGUFValueType.UINT32, None
        yield name, value, kind, subtype


def tensor_name(name):
    return f"{name}.bias" if name.endswith(".ssm_dt") else name


def write_text_model(reader, tensors, output):
    writer = gguf.GGUFWriter(output, "qwen35", endianess=reader.endianess)
    try:
        for name, value, kind, subtype in normalized_fields(reader):
            writer.add_key_value(name, value, kind, sub_type=subtype)
        for tensor in tensors:
            values = tensor.data
            writer.add_tensor_info(
                tensor_name(tensor.name),
                values.shape,
                values.dtype,
                values.nbytes,
                tensor.tensor_type,
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    finally:
        writer.close()


def verify_text_model(tensors, output):
    saved = gguf.GGUFReader(str(output))
    by_name = {tensor.name: tensor for tensor in saved.tensors}
    if len(by_name) != len(tensors):
        raise ValueError("保存后的文本张量数量有变")
    chunk_size = 8 * 1024 * 1024
    for tensor in tensors:
        reference = by_name[tensor_name(tensor.name)]
        if tensor.tensor_type != reference.tensor_type or not np.array_equal(
            tensor.shape, reference.shape
        ):
            raise ValueError(f"张量格式有变：{tensor.name}")
        current = tensor.data.view(np.uint8).reshape(-1)
        restored = reference.data.view(np.uint8).reshape(-1)
        if current.size != restored.size:
            raise ValueError(f"张量字节数有变：{tensor.name}")
        for start in range(0, current.size, chunk_size):
            stop = start + chunk_size
            if not np.array_equal(current[start:stop], restored[start:stop]):
                raise ValueError(f"张量字节有变：{tensor.name}")


def prepare(source, output):
    if output.exists():
        raise FileExistsError(f"输出文件已存在：{output}")
    reader = gguf.GGUFReader(str(source))
    architecture = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if architecture is None or architecture.contents() != "qwen35":
        raise ValueError("请选择 Qwen3.5 9B 的原始 Ollama GGUF")
    tensors = [
        tensor
        for tensor in reader.tensors
        if not tensor.name.startswith(("v.", "mtp."))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".gguf-") as folder:
        partial = Path(folder) / "model.gguf"
        write_text_model(reader, tensors, partial)
        verify_text_model(tensors, partial)
        partial.rename(output)
    return len(tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(
            f"找不到原始 Ollama GGUF：{args.source}。请按 README 下载或使用 --source。"
        )
    count = prepare(args.source, args.output)
    print(f"保存 {count} 个已逐字节校验的文本张量到 {args.output}")


if __name__ == "__main__":
    main()
