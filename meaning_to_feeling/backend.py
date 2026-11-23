import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
import psutil

from .core import parse_emotions, semantic_state
from .prompts import (
    ACTIVATION,
    ACTIVATION_SYSTEM,
    DEFAULT_MODEL,
    EMOTION_SCHEMA,
    EMOTION_SYSTEM,
    SCHEMA,
)


class Runtime:
    def __init__(self, weights, work_dir, port=11436, context_size=2048):
        self.weights = Path(weights).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.port = port
        self.context_size = context_size
        self.url = f"http://127.0.0.1:{port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.process = None
        self.identity = None

    def post(self, endpoint, body=None, timeout=120):
        if not endpoint.startswith("/") or endpoint.startswith("//"):
            raise ValueError("只允许本机API路径")
        request = urllib.request.Request(
            self.url + endpoint,
            None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def _server_arguments(self, binary, model):
        return [
            str(binary),
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--alias",
            DEFAULT_MODEL,
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
            "--pooling",
            "none",
            "--embd-normalize",
            "-1",
            "--ctx-size",
            str(self.context_size),
            "--batch-size",
            "1024",
            "--ubatch-size",
            "1024",
            "--parallel",
            "1",
            "--threads",
            "4",
            "--threads-batch",
            "4",
            "--gpu-layers",
            "999",
            "--cache-ram",
            "0",
            "--no-cache-prompt",
            "--reasoning",
            "off",
            "--no-mmproj",
            "--offline",
            "--embedding",
            "--device",
            "CUDA0",
            "--flash-attn",
            "on",
        ]

    def start(self):
        binary = self.weights / "runtime/llama-server.exe"
        model = self.weights / "model.gguf"
        if not binary.is_file() or not model.is_file():
            raise FileNotFoundError("权重目录需要model.gguf和runtime/llama-server.exe")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"{self.port}端口已被占用，请选择其他端口")
        available = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=8,
        )
        if float(available.splitlines()[0]) < 10000:
            raise RuntimeError("CUDA0空闲显存不足10000MiB，请等其他任务结束后再启动")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        arguments = self._server_arguments(binary, model)
        with (
            (self.work_dir / "server.stdout.log").open("w", encoding="utf-8") as out,
            (self.work_dir / "server.stderr.log").open("w", encoding="utf-8") as err,
        ):
            self.process = subprocess.Popen(
                arguments,
                cwd=binary.parent,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        self.identity = (
            self.process.pid,
            psutil.Process(self.process.pid).create_time(),
            binary.resolve(),
        )
        (self.work_dir / "service.json").write_text(
            json.dumps(
                dict(
                    pid=self.identity[0],
                    created_at=self.identity[1],
                    binary=str(binary),
                    arguments=arguments,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("模型服务启动失败，请检查server.stderr.log")
                try:
                    if self.post("/health", timeout=1).get("status") == "ok":
                        return self
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            raise TimeoutError("模型服务启动超时，请检查server.stderr.log")
        except BaseException:
            self.close()
            raise

    def check_tokens(self, body):
        prompt = self.post("/apply-template", {"messages": body["messages"]})["prompt"]
        count = len(
            self.post(
                "/tokenize",
                {"content": prompt, "add_special": False, "parse_special": True},
            )["tokens"]
        )
        if count + body["max_tokens"] > self.context_size:
            raise ValueError(f"提示与输出预算超过{self.context_size} token，请缩短正文或语境")
        return count

    def close(self):
        if self.process is None or self.process.poll() is not None:
            return
        pid, created, binary = self.identity
        try:
            live = psutil.Process(pid)
            if (
                abs(live.create_time() - created) > 0.1
                or Path(live.exe()).resolve() != binary
            ):
                return
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass


def grade_payload(row):
    visible = dict(
        character=row["character"],
        background=row["context"],
        visible_text=row["prefix"],
    )
    return dict(
        model=DEFAULT_MODEL,
        messages=[
            dict(role="system", content=EMOTION_SYSTEM),
            dict(role="user", content=json.dumps(visible, ensure_ascii=False)),
        ],
        temperature=0,
        seed=17,
        max_tokens=256,
        stream=False,
        response_format=dict(
            type="json_schema",
            json_schema=dict(name="EmotionGrades", strict=True, schema=EMOTION_SCHEMA),
        ),
    )


def activation_payload(row, emotions):
    visible = dict(
        character=row["character"],
        background=row["context"],
        visible_text=row["prefix"],
        emotions=emotions,
    )
    return dict(
        model=DEFAULT_MODEL,
        messages=[
            dict(role="system", content=ACTIVATION_SYSTEM),
            dict(role="user", content=json.dumps(visible, ensure_ascii=False)),
        ],
        temperature=0,
        seed=17,
        max_tokens=64,
        stream=False,
        response_format=dict(
            type="json_schema",
            json_schema=dict(name="NonNeutralMass", strict=True, schema=SCHEMA),
        ),
    )


def boundary_payload(request):
    options = request["options"]
    return dict(
        model=DEFAULT_MODEL,
        messages=request["messages"],
        temperature=options.get("temperature", 0),
        seed=options.get("seed", 17),
        max_tokens=options["num_predict"],
        stream=False,
        response_format=dict(
            type="json_schema",
            json_schema=dict(
                name="MeaningBoundaries", strict=True, schema=request["format"]
            ),
        ),
    )


class LocalStore:
    semantic_interface = "nine-emotion-activation-composition-v1"

    def __init__(self, runtime, path):
        self.runtime = runtime
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS responses(request TEXT PRIMARY KEY,response TEXT NOT NULL,seconds REAL NOT NULL)"
        )
        self.connection.commit()
        model = runtime.weights / "model.gguf"
        stat = model.stat()
        self.context = dict(
            model=str(model),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            context_size=getattr(runtime, "context_size", 2048),
            micro_batch_size=1024,
            pooling="none",
            mode="embedding",
            device="cuda",
        )
        self.requests = 0

    def close(self):
        self.connection.close()

    def request_body(self, body):
        key = json.dumps(
            dict(runtime=self.context, endpoint="/v1/chat/completions", request=body),
            ensure_ascii=False,
            sort_keys=True,
        )
        saved = self.connection.execute(
            "SELECT response,seconds FROM responses WHERE request=?", (key,)
        ).fetchone()
        if saved is None:
            self.runtime.check_tokens(body)
            start = time.perf_counter()
            result = self.runtime.post("/v1/chat/completions", body)
            seconds = time.perf_counter() - start
            self.requests += 1
            self.connection.execute(
                "INSERT INTO responses VALUES (?,?,?)",
                (key, json.dumps(result, ensure_ascii=False), seconds),
            )
            self.connection.commit()
        else:
            result, seconds = json.loads(saved[0]), saved[1]
        if (
            result.get("model") != DEFAULT_MODEL
            or result["choices"][0].get("finish_reason") != "stop"
        ):
            raise ValueError("模型不符或输出未完整结束，请检查服务日志")
        return result, seconds

    def request(self, request):
        result, seconds = self.request_body(boundary_payload(request))
        return dict(
            model=DEFAULT_MODEL,
            message=result["choices"][0]["message"],
            transport="native_schema",
            usage=result.get("usage"),
        ), seconds

    constrained_request = request

    def one(self, row):
        if row.get("profile_prior"):
            raise ValueError("当前接口不接收语言画像条件")
        body = grade_payload(row)
        start = time.perf_counter()
        before = self.requests
        seconds = 0.0
        for attempt in range(2):
            response, extra = self.request_body(body)
            seconds += extra
            try:
                answer = parse_emotions(response["choices"][0]["message"]["content"])
                break
            except (ValueError, TypeError, KeyError) as exc:
                if attempt:
                    raise
                body["messages"] += [
                    dict(
                        role="assistant",
                        content=response["choices"][0]["message"].get("content", ""),
                    ),
                    dict(
                        role="user",
                        content=f"格式错误：{exc}。只返回emotions及完整九个情绪键，值为absent/low/medium/high。",
                    ),
                ]
        old = semantic_state(answer)
        if old[-1] == 1.0:
            semantic = old
        else:
            response, extra = self.request_body(
                activation_payload(row, answer["emotions"])
            )
            seconds += extra
            value = json.loads(response["choices"][0]["message"]["content"])
            if (
                not isinstance(value, dict)
                or set(value) != {"activation"}
                or not isinstance(value["activation"], str)
                or value["activation"] not in ACTIVATION
            ):
                raise ValueError("需要activation字段及规定的总权重词")
            mass = ACTIVATION[value["activation"]]
            composition = old[:-1] / old[:-1].sum()
            semantic = np.r_[mass * composition, 1.0 - mass]
        return dict(
            emotions=answer["emotions"], factor_semantic=semantic.tolist()
        ), dict(
            seconds=seconds,
            wall_seconds=time.perf_counter() - start,
            cache_hit=before == self.requests,
            format_repair=bool(attempt),
            semantic_interface=self.semantic_interface,
        )

    def state_readout(self, answer):
        return np.asarray(answer["factor_semantic"], float)
