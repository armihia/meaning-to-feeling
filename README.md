# Meaning-to-Feeling

从新文本、语境和代入人物预测意元边界、逐意元十维情绪状态及相邻变化。个人档案由同一读者在其他材料上的分块和感受评分生成，可以保存后用于新的文本。

## 安装

完整推理支持 Windows x64、NVIDIA CUDA 和 Python 3.10+。模型服务需要 CUDA0 至少 **10,000 MiB** 空闲显存。以下命令在仓库根目录运行。

```powershell
git clone https://github.com/armihia/meaning-to-feeling.git
cd meaning-to-feeling
conda create -n meaning-to-feeling python=3.11 -y
conda activate meaning-to-feeling
python -m pip install -e .
python -m pip install gguf==0.19.0
```

## 模型与第三方依赖

项目训练的数值头和个人校准先验在 `model_weights/meaning_to_feeling_9b/`。完整推理另需 Qwen3.5 9B 底座和 Windows CUDA 运行时。

| 下载项 | 下载后的位置 |
| --- | --- |
| [实验使用的 Ollama Qwen3.5 9B Q4_K_M 原文件](https://registry.ollama.ai/v2/library/qwen3.5/blobs/sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c) | 保存原文件后运行下文的转换工具，生成 `model_weights/meaning_to_feeling_9b/model.gguf`。 |
| [llama.cpp b11669 Windows CUDA 13.4 x64](https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-win-cuda-13.4-x64.zip) | 将可执行文件和配套 DLL 解压到 `model_weights/meaning_to_feeling_9b/runtime/`。 |
| [llama.cpp b11669 CUDA 13.4 依赖 DLL](https://github.com/ggml-org/llama.cpp/releases/download/b11669/cudart-llama-bin-win-cuda-13.4-x64.zip) | 解压到同一个 `runtime/`，使 DLL 和 `llama-server.exe` 位于同一级。 |

在本地准备 Qwen3.5 9B：

```powershell
New-Item -ItemType Directory -Force .runtime | Out-Null
curl.exe --fail --location "https://registry.ollama.ai/v2/library/qwen3.5/blobs/sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c" --output .runtime/qwen3.5-9b-source.gguf
python tools/prepare_ollama.py --source .runtime/qwen3.5-9b-source.gguf
```

已有这份 [Ollama 模型](https://ollama.com/library/qwen3.5:9b) 时，可以直接运行 `python tools/prepare_ollama.py`。工具在 `OLLAMA_MODELS` 或默认 Ollama 模型目录查找固定版本，也可以用 `--source` 指定本地文件。转换保留文本张量字节并逐一校验，生成的 GGUF 约 5.15 GiB。

```text
meaning-to-feeling/
├── meaning_to_feeling/
├── examples/
├── tools/prepare_ollama.py
└── model_weights/
    ├── meaning_to_feeling_9b/
    │   ├── model.json
    │   ├── reliability_prior_candidate.npz
    │   ├── gain_prior_candidate.json
    │   ├── model.gguf
    │   └── runtime/
    │       ├── llama-server.exe
    │       └── 配套 DLL
    └── shared/
        ├── storycommonsense.npz
        └── rocling.npz
```

用 `--weights` 可选择其他目录。经常使用的本机路径也可写入 `.env`，模板见 [配置示例](.env.example)。

## 网页试用

双击 `启动试用台.cmd`，或运行下面的命令。打开 [本地试用台](http://127.0.0.1:3002/) 后，可填写正文、语境和人物，选择示例档案或导入自己的档案。历史生成页可以保存档案，分析页可以比较两个档案并导出分块和矩阵。

```powershell
python -m meaning_to_feeling.testbench --browser
```

模型在首次分析时加载。切换端口可加 `--port 13002 --model-port 12436`。退出使用 `python -m meaning_to_feeling.testbench --stop`，自定义端口时传入同一个 `--port`。

## 命令行推理

### 使用共享档案

```powershell
python -m meaning_to_feeling predict --input examples/input.json --output outputs/shared.json
```

### 从独立历史生成人的档案

```powershell
python -m meaning_to_feeling profile --profile-strategy history_mean --input examples/history.json --output outputs/profile.json
python -m meaning_to_feeling predict --input examples/input.json --profile outputs/profile.json --output outputs/personal.json
```

`examples/history.json` 用于演示格式。实际使用时填写此人在其他材料上的真实观察。从某次开读时间回放时，历史条目填写带时区的 `finished_at`，生成时传入 `--as-of`。可选的可靠性和正文变化先验由 `--calibration-prior` 和 `--gain-prior` 传入。用这些先验生成档案时沿用原始档案策略。

### 输入与输出

| 字段 | 内容 |
| --- | --- |
| `text` | 新正文。 |
| `context_text` | 正文出现前已知的语境。 |
| `target_character` | 代入的人物。 |
| `units` | 意元及其原字符区间。 |
| `initial_emotion` | 只读语境后的十维状态。 |
| `emotion_matrix` | 每个意元末端的累计感受。 |
| `delta_matrix` | 相邻状态的差，首行相对开始前状态。 |

十维顺序为高兴、悲伤、愤怒、恐惧、惊讶、焦虑、受挫、困惑、内疚、平淡。状态各项总和为 1。字符使用 Unicode 码点位置。切分可以用全文，感受读出使用语境和相应意元末端之前的正文。默认策略保留固定句读并按语义判断候选词尾。复现旧切分协议可指定 `--segmentation-policy legacy`。

## 公开基准的共享读出

仓库还提供两个项目拟合的预测头。二者复用冻结的 Qwen3-Embedding-4B。Story Commonsense 以指定人物和可见故事前缀预测八类情绪。ROCLING DSA-MST 预测反思文本的效价和唤醒度。它们是共享语义组件的研究读出，完整的个人轨迹使用上面的 9B 入口。

按 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/) 安装与 CUDA 相符的 PyTorch，再准备 [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)。

```powershell
python -m pip install "transformers>=4.51,<6" huggingface_hub scikit-learn safetensors
hf download Qwen/Qwen3-Embedding-4B --local-dir models/qwen3-embedding-4b
python -m meaning_to_feeling.shared storycommonsense --input examples/storycommonsense.json --output outputs/storycommonsense.json
```

ROCLING 的核回归需要在本机编码 [原竞赛开发文本](https://github.com/NYCU-NLP/ROCLING-2025-ST-DSA-MST/blob/ee92b8e4eebdc78f8f6f63d5bf29b5d27bb33b95/Dataset/DSAMST-ValidationSet.csv)。 `prepare-rocling` 只需要开发输入 CSV。仓库里的 `rocling.npz` 提供已拟合的系数。

```powershell
New-Item -ItemType Directory -Force .runtime | Out-Null
curl.exe --fail --location "https://raw.githubusercontent.com/NYCU-NLP/ROCLING-2025-ST-DSA-MST/ee92b8e4eebdc78f8f6f63d5bf29b5d27bb33b95/Dataset/DSAMST-ValidationSet.csv" --output .runtime/DSAMST-ValidationSet.csv
python -m meaning_to_feeling.shared prepare-rocling --input .runtime/DSAMST-ValidationSet.csv --output .runtime/rocling_support.npz
python -m meaning_to_feeling.shared rocling --input examples/input.json --output outputs/rocling.json
```

原评测的 4B 读出使用 CUDA。Story Commonsense 使用 FP32，ROCLING 使用 BF16。单机试用也可选 `--device cpu`，ROCLING 查询和开发输入需使用一致的编码器及精度。

| 测试协议 | 共享读出成绩 |
| --- | --- |
| [Story Commonsense 作者的 14,216 条测试](https://aclanthology.org/2020.acl-main.426/) | 八类 micro-F1 67.54%。 |
| [ROCLING 2025 DSA-MST 的 1,541 条测试](https://aclanthology.org/2025.rocling-main.41/) | 效价 PCC 0.81528，唤醒度 PCC 0.63771。 |

## 检查

```powershell
python -m unittest discover -s tests
node --test tests/test_profile_details.cjs
```

[第三方来源和许可证](THIRD_PARTY_NOTICES.md)
