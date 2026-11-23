import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import queue
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import uuid
import webbrowser

from dotenv import load_dotenv

from . import DEFAULT_WEIGHTS, Model
from .core import text_row, validate_profile
from .demo_data import COLORS, EXAMPLES, HISTORY_EXAMPLE, LABELS, PROFILES
from .history_scope import select_completed_history
from .prompts import DEFAULT_MODEL, DYNAMIC_PROFILE_KIND, EMOTIONS, HISTORY_MEAN_PROFILE_KIND

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("web")
APPLICATION = "meaning-to-feeling-testbench"
RELEASE = "2026.11.05"


class BusyError(ValueError):
    pass


def load_profiles(path, parameters):
    result = copy.deepcopy(PROFILES)
    if path and Path(path).is_file():
        local = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(local, list):
            raise ValueError("本地示例向量文件需要数组")
        seen = {r["id"] for r in result}
        for row in local:
            if (
                not isinstance(row, dict)
                or row.get("id") in seen
                or not isinstance(row.get("id"), str)
            ):
                raise ValueError("本地示例向量需要独立的id")
            validate_profile(
                row["profile"],
                parameters["model"],
                parameters["heads"]["query"]["name"],
            )
            result.append(
                {
                    key: row[key]
                    for key in [
                        "id",
                        "name",
                        "source",
                        "source_label",
                        "description",
                        "profile",
                    ]
                }
            )
            seen.add(row["id"])
    return result


class ReportingStore:
    def __init__(self, store, notify):
        self.store = store
        self.notify = notify

    def __getattr__(self, name):
        return getattr(self.store, name)

    def one(self, row):
        end = len(row["prefix"])
        self.notify(
            "reading", "记录开始前的感受" if end == 0 else f"读取前{end}个字的感受"
        )
        return self.store.one(row)

    def request(self, value):
        self.notify("segmenting", "划分句子中的意义单元")
        return self.store.request(value)

    def constrained_request(self, value):
        self.notify("segmenting", "补充当前粒度下的切分位置")
        return self.store.constrained_request(value)


class Bench:
    def __init__(
        self,
        weights,
        work_dir,
        profiles_path=None,
        model_port=11436,
        model_factory=Model,
        *,
        calibration_prior=None,
        gain_prior=None,
    ):
        self.weights = Path(weights).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.model_port = model_port
        self.parameters = json.loads(
            (self.weights / "model.json").read_text(encoding="utf-8-sig")
        )
        self.profiles = load_profiles(profiles_path, self.parameters)
        self.profile_calibrations = {}
        if gain_prior is not None and calibration_prior is None:
            raise ValueError("正文变化候选还需要配套的calibration-prior")
        if calibration_prior is not None:
            from .reliability import ReliabilityCalibration

            history_head = self.parameters["heads"]["history"]["name"]
            self.profile_calibrations["reliability"] = ReliabilityCalibration(
                calibration_prior, history_head
            )
            if gain_prior is not None:
                self.profile_calibrations["body_gain"] = ReliabilityCalibration(
                    calibration_prior, history_head, gain_prior=gain_prior
                )
        self.model_factory = model_factory
        self.model = None
        self.loaded = False
        self.loading = False
        self.jobs = OrderedDict()
        self.active = None
        self.lock = threading.RLock()
        self.tasks = queue.Queue()
        self.closing = False
        self.worker = threading.Thread(
            target=self._run, name="meaning-to-feeling-worker", daemon=True
        )
        self.worker.start()

    def bootstrap(self):
        return dict(
            application=APPLICATION,
            model=self.parameters["model"],
            release=RELEASE,
            dimensions=12,
            supported_dimensions=[12, 22, 23],
            supported_profile_kinds=[DYNAMIC_PROFILE_KIND, HISTORY_MEAN_PROFILE_KIND],
            default_profile_generator="history_mean",
            supports_history_time_cutoff=True,
            profile_generators=[
                dict(
                    id=key,
                    name=name,
                    configured=key in ("original", "history_mean") or key in self.profile_calibrations,
                )
                for key, name in [
                    ("history_mean", "本人历史均值融合 · 新版"),
                    ("original", "原版校准 · 12维"),
                    ("reliability", "可靠性校准 · 12维"),
                    ("body_gain", "初始参考与正文变化 · 23维"),
                ]
            ],
            profiles=self.profiles,
            examples=EXAMPLES,
            history_example=HISTORY_EXAMPLE,
            emotions=[
                dict(key=key, label=label, color=color)
                for key, label, color in zip(EMOTIONS, LABELS, COLORS)
            ],
        )

    def status(self):
        with self.lock:
            return dict(
                application=APPLICATION,
                release=RELEASE,
                loaded=self.loaded,
                loading=self.loading,
                busy=self.active is not None,
                active_job=self.active,
                model=DEFAULT_MODEL,
                weights=str(self.weights),
                work_dir=str(self.work_dir),
                model_port=self.model_port,
            )

    def _profile(self, value):
        if not isinstance(value, dict):
            raise ValueError("请选择或填写个人向量")
        if "preset_id" in value:
            preset = next(
                (p for p in self.profiles if p["id"] == value["preset_id"]), None
            )
            if preset is None:
                raise ValueError("没有找到这个示例向量，请重新选择")
            return copy.deepcopy(preset)
        profile = value.get("value")
        if isinstance(profile, dict) and set(profile) == {"vector"}:
            profile = profile["vector"]
        if profile is None:
            raise ValueError("请粘贴完整profile.json，或12个数的数组")
        validate_profile(
            profile, self.parameters["model"], self.parameters["heads"]["query"]["name"]
        )
        name = value.get("name", "我的向量")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ValueError("请填写不超过60字的向量名称")
        return dict(
            id="custom",
            name=name,
            source="custom",
            source_label="自填向量",
            profile=copy.deepcopy(profile),
        )

    def prepare(self, value):
        if not isinstance(value, dict):
            raise ValueError("请求需要JSON对象")
        kind = value.get("kind", "predict")
        if kind == "predict":
            supplied = value.get("input")
            row = text_row(supplied)
            if (
                len(row["text"]) > 2000
                or len(row["context"]) > 4000
                or len(row["character"]) > 100
            ):
                raise ValueError("请先用一两句话试用，并缩短过长的语境或人物名称")
            profiles = value.get("profiles")
            policy = value.get("segmentation_policy", "detailed")
            if policy not in ["detailed", "legacy"]:
                raise ValueError("不支持这个切分策略")
            if not isinstance(profiles, list) or not 1 <= len(profiles) <= 2:
                raise ValueError("每次请选择一个向量，或两个向量进行对比")
            return dict(
                kind=kind,
                input={
                    k: supplied.get(k, default)
                    for k, default in [
                        ("text", ""),
                        ("context_text", ""),
                        ("target_character", "我"),
                    ]
                },
                profiles=[self._profile(p) for p in profiles],
                segmentation_policy=policy,
            )
        if kind == "profile":
            generator = value.get("profile_generator", "original")
            if generator not in ["original", "history_mean", "reliability", "body_gain"]:
                raise ValueError("请选择已有的向量生成方式")
            if generator not in ("original", "history_mean") and generator not in self.profile_calibrations:
                raise ValueError("这个生成方式尚未配置，请先选择原版校准")
            history = value.get("history")
            if (
                not isinstance(history, dict)
                or history.get("emotion_order") != EMOTIONS
                or not isinstance(history.get("history"), list)
                or not 1 <= len(history["history"]) <= 30
            ):
                raise ValueError(
                    "历史JSON需要十维emotion_order和1至30段history；可先载入格式示例"
                )
            history, scope = select_completed_history(history, value.get("as_of"))
            for row in history["history"]:
                text_row(row)
            return dict(
                kind=kind, history=copy.deepcopy(history), profile_generator=generator,
                history_time_scope=scope,
            )
        if kind in ["load", "unload"]:
            return dict(kind=kind)
        raise ValueError("不支持这个操作")

    def submit(self, value):
        task = self.prepare(value)
        with self.lock:
            if self.closing:
                raise BusyError("试用台正在关闭")
            if self.active is not None:
                raise BusyError("正在处理上一次操作，请等它完成后再开始")
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = dict(
                id=job_id,
                kind=task["kind"],
                state="queued",
                stage="queued",
                message="即将开始",
                created_at=time.time(),
                result=None,
                error=None,
            )
            self.active = job_id
            while len(self.jobs) > 12:
                self.jobs.popitem(last=False)
            self.tasks.put((job_id, task))
        return self.job(job_id)

    def job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("这次任务已不在当前会话，请重新开始分析")
            return copy.deepcopy(self.jobs[job_id])

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def _ensure_model(self, job_id):
        if self.model is not None and self.loaded:
            return False
        self.update(
            job_id, stage="loading", message="正在加载9B模型，首次加载需要等待片刻"
        )
        with self.lock:
            self.loading = True
            self.model = self.model_factory(
                self.weights, self.work_dir, self.model_port
            )
        try:
            self.model.__enter__()
            with self.lock:
                self.loaded = True
            return True
        except BaseException:
            with self.lock:
                self.model = None
            raise
        finally:
            with self.lock:
                self.loading = False

    def _unload(self):
        model = self.model
        if model is not None:
            model.__exit__(None, None, None)
        with self.lock:
            self.model = None
            self.loaded = False

    def _execute(self, job_id, task):
        start = time.perf_counter()
        if task["kind"] == "unload":
            self._unload()
            return dict(released=True)
        loaded = self._ensure_model(job_id)
        if self.closing:
            raise RuntimeError("试用台正在关闭")
        if task["kind"] == "load":
            return dict(loaded=True)
        original = self.model.engine.store
        reader = ""

        def notify(stage, message):
            self.update(
                job_id,
                stage=stage,
                message=(reader + " · " if reader else "") + message,
            )

        self.model.engine.store = ReportingStore(original, notify)
        try:
            if task["kind"] == "profile":
                generator = task["profile_generator"]
                if generator == "original":
                    result = self.model.generate_profile(task["history"])
                elif generator == "history_mean":
                    result = self.model.generate_profile(task["history"], strategy=generator)
                else:
                    result = self.profile_calibrations[generator].generate_profile(
                        self.model.engine, task["history"]
                    )
                result["history_source"] = task["history"].get(
                    "label_source", "user_provided"
                )
                if task["history_time_scope"] is not None:
                    result["history_time_scope"] = task["history_time_scope"]
                return dict(
                    profile=result,
                    profile_generator=generator,
                    wall_seconds=time.perf_counter() - start,
                )
            runs = []
            for preset in task["profiles"]:
                reader = preset["name"]
                notify("segmenting", "开始分析")
                prediction = self.model.predict(
                    task["input"],
                    preset["profile"],
                    segmentation_policy=task["segmentation_policy"],
                )
                runs.append(
                    dict(
                        name=preset["name"],
                        source_label=preset["source_label"],
                        profile=preset["profile"],
                        prediction=prediction,
                    )
                )
            return dict(
                input=task["input"],
                segmentation_policy=task["segmentation_policy"],
                runs=runs,
                wall_seconds=time.perf_counter() - start,
                loaded_model_this_job=loaded,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            self.model.engine.store = original

    def _run(self):
        try:
            while True:
                item = self.tasks.get()
                if item is None:
                    break
                job_id, task = item
                self.update(job_id, state="running", started_at=time.time())
                try:
                    result = self._execute(job_id, task)
                    outcome = dict(
                        state="complete",
                        stage="complete",
                        message="分析完成",
                        result=result,
                        finished_at=time.time(),
                    )
                except Exception as exc:
                    process = getattr(
                        getattr(self.model, "runtime", None), "process", None
                    )
                    if process is not None and process.poll() is not None:
                        self._unload()
                    outcome = dict(
                        state="failed",
                        stage="failed",
                        message="本次操作未完成",
                        error=str(exc),
                        finished_at=time.time(),
                    )
                finally:
                    with self.lock:
                        self.jobs[job_id].update(outcome)
                        self.active = None
                if self.closing:
                    break
        finally:
            self._unload()

    def close(self):
        with self.lock:
            self.closing = True
            model = self.model
        if model is not None:
            model.runtime.close()
        self.tasks.put(None)
        self.worker.join(timeout=20)


def handler_for(bench, assets=ASSETS):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def allowed(self):
            port = self.server.server_port
            hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
            if self.headers.get("Host") not in hosts:
                self.respond(403, dict(error="请从本机试用地址打开页面"))
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in [
                "http://" + host for host in hosts
            ]:
                self.respond(403, dict(error="仅接受本机试用页面的请求"))
                return False
            return True

        def respond(
            self, status, value, content_type="application/json; charset=utf-8"
        ):
            raw = (
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if isinstance(value, (dict, list))
                else value
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.allowed():
                return
            path = urlsplit(self.path).path
            if path == "/api/bootstrap":
                self.respond(200, bench.bootstrap())
            elif path == "/api/status":
                self.respond(200, bench.status())
            elif path.startswith("/api/jobs/"):
                try:
                    self.respond(200, bench.job(path.removeprefix("/api/jobs/")))
                except KeyError as exc:
                    self.respond(404, dict(error=str(exc)))
            else:
                files = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css"}
                if path not in files:
                    self.respond(404, dict(error="页面不存在"))
                    return
                file = assets / files[path]
                content_type = (
                    mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                )
                self.respond(200, file.read_bytes(), content_type + "; charset=utf-8")

        def do_POST(self):
            if not self.allowed():
                return
            path = urlsplit(self.path).path
            if path not in ["/api/jobs", "/api/stop"]:
                self.respond(404, dict(error="接口不存在"))
                return
            try:
                if (
                    self.headers.get("Content-Type", "").split(";")[0].strip()
                    != "application/json"
                ):
                    raise ValueError("请使用JSON提交")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 512 * 1024:
                    raise ValueError("输入为空或超过512KiB，请缩短历史记录")
                value = json.loads(
                    self.rfile.read(size).decode("utf-8-sig"),
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        ValueError("数值必须有限")
                    ),
                )
                if path == "/api/stop":
                    if value != {}:
                        raise ValueError("停止服务不接受其他参数")
                    self.respond(200, dict(stopping=True))
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                self.respond(202, bench.submit(value))
            except BusyError as exc:
                self.respond(409, dict(error=str(exc)))
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                self.respond(400, dict(error=str(exc)))

    return Handler


def main():
    load_dotenv(PACKAGE_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3002)
    parser.add_argument("--model-port", type=int, default=11436)
    parser.add_argument(
        "--calibration-prior",
        type=Path,
        default=os.getenv("MTF_CALIBRATION_PRIOR") or None,
    )
    parser.add_argument(
        "--gain-prior", type=Path, default=os.getenv("MTF_GAIN_PRIOR") or None
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=os.getenv("MTF_WEIGHTS") or DEFAULT_WEIGHTS,
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=os.getenv("MTF_DEMO_PROFILES") or PACKAGE_ROOT / "examples/profiles.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PACKAGE_ROOT / ".runtime/testbench",
    )
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "api/status", timeout=1) as stream:
            existing = json.load(stream)
        if existing.get("application") == APPLICATION and existing.get(
            "weights"
        ) == str(args.weights.resolve()):
            if args.stop:
                request = urllib.request.Request(
                    url + "api/stop",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                with opener.open(request, timeout=5) as stream:
                    json.load(stream)
                print("试用台正在关闭，本轮模型会同时释放。", flush=True)
                return
            print("试用台已经运行：" + url, flush=True)
            if args.browser:
                webbrowser.open(url)
            return
    except (OSError, ValueError):
        pass
    if args.stop:
        print("没有找到本目录正在运行的试用台。", flush=True)
        return
    bench = Bench(
        args.weights,
        args.work_dir,
        args.profiles,
        args.model_port,
        calibration_prior=args.calibration_prior,
        gain_prior=args.gain_prior,
    )
    try:
        with ThreadingHTTPServer(
            ("127.0.0.1", args.port), handler_for(bench)
        ) as server:
            print("意元与感受 · 本地试用台：" + url, flush=True)
            print(
                "模型在首次分析时加载。结束使用可释放显存；退出请运行停止试用台.cmd，或在此窗口按Ctrl+C。",
                flush=True,
            )
            if args.browser:
                webbrowser.open(url)
            server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        bench.close()


if __name__ == "__main__":
    main()
