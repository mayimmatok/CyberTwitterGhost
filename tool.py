import os
import re
import json
import time
import html as html_lib
import random
import socket
import ssl
import shutil
import hashlib
import urllib.parse
import urllib.request
import urllib.error
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
SNAPSHOT_ROOT = OUTPUT_DIR / "快照html"
DATA_DIR = OUTPUT_DIR / "训练数据"

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
DEFAULT_TIMEOUT = 30
socket.setdefaulttimeout(DEFAULT_TIMEOUT)
SSL_CONTEXT = ssl._create_unverified_context()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

MENTION_RE = re.compile(r"@([A-Za-z0-9_]{1,15})")
URL_RE = re.compile(r"https?://\S+|pic\.x\.com/\S+|pic\.twitter\.com/\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
STATUS_ID_RE = re.compile(r"status/(\d+)")

DEFAULT_MODEL_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
MAX_HUMAN_CHARS = 120
MAX_COMPLETION_TRIES = 3


def ensure_t2s_converter():
    try:
        import zhconv
    except ImportError:
        print("正在安装依赖 zhconv ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zhconv", "--quiet"])
        import zhconv

    def _convert(text: str) -> str:
        if not text:
            return ""
        return zhconv.convert(text, "zh-cn")

    return _convert


t2s = ensure_t2s_converter()


class ProgressBars:
    def __init__(self, label: str, total_steps: int):
        self.label = label
        self.total_steps = max(int(total_steps), 1)
        self.current_step = 0
        self.step_name = ""
        self.step_total = 1
        self.step_done = 0

    @staticmethod
    def _bar(done: float, total: float, width: int = 26) -> str:
        if total <= 0:
            total = 1
        ratio = max(0.0, min(1.0, done / total))
        filled = int(width * ratio)
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    def _print_total(self, completed_steps: int):
        bar = self._bar(completed_steps, self.total_steps)
        print(f"\n[{self.label}] 总进度 {bar} {completed_steps}/{self.total_steps}")

    def start_step(self, name: str, total: int):
        self.current_step += 1
        self.step_name = name
        self.step_total = max(int(total), 1)
        self.step_done = 0
        self._print_total(self.current_step - 1)
        self._print_step()

    def _print_step(self):
        bar = self._bar(self.step_done, self.step_total)
        msg = f"\r  [步骤] {self.step_name} {bar} {self.step_done}/{self.step_total}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def set_done(self, value: int):
        self.step_done = max(0, min(int(value), self.step_total))
        self._print_step()

    def advance(self, value: int = 1):
        self.set_done(self.step_done + value)

    def finish_step(self, note: str = ""):
        self.set_done(self.step_total)
        sys.stdout.write("\n")
        if note:
            print(f"  -> {note}")
        self._print_total(self.current_step)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def extract_status_id(timestamp: str, original: str) -> str:
    m = STATUS_ID_RE.search(original or "")
    if m:
        return m.group(1)
    digest = hashlib.md5(f"{timestamp}|{original}".encode("utf-8", errors="ignore")).hexdigest()
    return f"noid_{digest[:16]}"


def fetch_url(url: str, retries: int = 3, pause: float = 1.2):
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": random.choice(USER_AGENTS)})
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=DEFAULT_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == retries:
                return None
            time.sleep(pause * attempt)
    return None


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html_lib.unescape(text)
    text = MENTION_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = text.replace("\u200b", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return t2s(text)


def is_meaningful_text(text: str) -> bool:
    if not text:
        return False
    if len(text.strip()) < 2:
        return False
    return re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text) is not None


def extract_json_payload(page_html: str):
    if not page_html:
        return None

    pre_blocks = re.findall(r"<pre>(.*?)</pre>", page_html, flags=re.DOTALL | re.IGNORECASE)
    for block in pre_blocks:
        raw = html_lib.unescape(block).strip()
        if not raw.startswith("{"):
            continue
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict) and "data" in payload:
                return payload
        except json.JSONDecodeError:
            continue
    return None


def extract_tweet_content_blocks(page_html: str):
    blocks = re.findall(
        r'<div[^>]*class=["\']tweet-content["\'][^>]*>(.*?)</div>',
        page_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = [clean_text(re.sub(r"<[^>]+>", " ", b)) for b in blocks]
    return [x for x in cleaned if x]


def parse_conversation_from_payload(payload: dict, page_html: str):
    data = payload.get("data") or {}
    includes = payload.get("includes") or {}
    tweets = includes.get("tweets") or []

    gpt_text_raw = data.get("text", "")
    gpt_text = clean_text(gpt_text_raw)
    if not gpt_text:
        return None

    referenced = data.get("referenced_tweets") or []
    ref_by_type = {item.get("type"): item.get("id") for item in referenced if isinstance(item, dict)}

    is_retweet = "retweeted" in ref_by_type or gpt_text_raw.strip().startswith("RT @")
    if is_retweet:
        return None

    replied_id = ref_by_type.get("replied_to")

    if replied_id:
        tweet_map = {str(t.get("id")): t for t in tweets if isinstance(t, dict) and t.get("id")}
        replied = tweet_map.get(str(replied_id))
        human_text = clean_text((replied or {}).get("text", ""))

        if not human_text:
            blocks = extract_tweet_content_blocks(page_html)
            if len(blocks) >= 2:
                # Wayback 模板通常是 [当前推文, 嵌入父推文]
                human_text = blocks[1]

        if not human_text:
            return None

        return {"human": human_text, "gpt": gpt_text}

    return {"human": "", "gpt": gpt_text}


def extract_conversation_from_file(file_path: Path):
    try:
        page_html = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    payload = extract_json_payload(page_html)
    if not payload:
        return None

    return parse_conversation_from_payload(payload, page_html)


def download_snapshots_for_user(username: str):
    params = {
        "url": f"twitter.com/{username}/status/*",
        "output": "json",
        "fl": "timestamp,original",
        "collapse": "digest",
    }
    cdx_url = f"{CDX_ENDPOINT}?{urllib.parse.urlencode(params)}"

    raw = fetch_url(cdx_url, retries=5)
    if not raw:
        raise RuntimeError("无法连接 Wayback CDX 接口")

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CDX 返回不是合法 JSON") from exc

    if not rows or len(rows) <= 1:
        return []
    return rows[1:]


def build_snapshot_items(rows):
    item_map = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        timestamp, original = row[0], row[1]
        status_id = extract_status_id(timestamp, original)
        # 同一条 status 取最新快照。
        if status_id not in item_map or str(timestamp) > str(item_map[status_id]["timestamp"]):
            item_map[status_id] = {
                "status_id": status_id,
                "timestamp": str(timestamp),
                "original": str(original),
            }
    items = list(item_map.values())
    items.sort(key=lambda x: x["timestamp"])
    return items


def default_checkpoint(username: str):
    return {
        "version": 2,
        "username": username,
        "updated_at": int(time.time()),
        "files": {},
        "processed": {},
        "failed_downloads": {},
    }


def load_checkpoint(path: Path, username: str):
    if not path.exists():
        return default_checkpoint(username)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_checkpoint(username)
    if not isinstance(data, dict):
        return default_checkpoint(username)
    if data.get("username") != username:
        return default_checkpoint(username)
    for key, default_val in {
        "files": {},
        "processed": {},
        "failed_downloads": {},
    }.items():
        if not isinstance(data.get(key), dict):
            data[key] = default_val
    data["version"] = 2
    return data


def save_checkpoint(path: Path, data: dict):
    data["updated_at"] = int(time.time())
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def build_snapshot_file_path(user_dir: Path, item: dict) -> Path:
    safe_name = sanitize_filename(item["original"])
    filename = f"snapshot_{item['timestamp']}_{safe_name}.html"
    return user_dir / filename


def _download_one_snapshot(item: dict, user_dir: Path):
    file_path = build_snapshot_file_path(user_dir, item)
    if file_path.exists():
        return item["status_id"], True, file_path.name, ""

    url_iframe = f"https://web.archive.org/web/{item['timestamp']}if_/{item['original']}"
    content = fetch_url(url_iframe, retries=3)
    if not content:
        url_normal = f"https://web.archive.org/web/{item['timestamp']}/{item['original']}"
        content = fetch_url(url_normal, retries=3)

    if not content:
        return item["status_id"], False, file_path.name, "download_failed"

    try:
        file_path.write_text(content, encoding="utf-8")
    except Exception:
        return item["status_id"], False, file_path.name, "write_failed"

    return item["status_id"], True, file_path.name, ""


def parallel_download_snapshots(items, user_dir: Path, checkpoint: dict, progress: ProgressBars):
    files = checkpoint["files"]
    failed = checkpoint["failed_downloads"]

    existing = 0
    pending = []
    for item in items:
        sid = item["status_id"]
        file_path = build_snapshot_file_path(user_dir, item)
        if file_path.exists():
            files[sid] = file_path.name
            failed.pop(sid, None)
            existing += 1
        else:
            pending.append(item)

    progress.start_step("多线下载快照", len(items))
    progress.set_done(existing)

    if not pending:
        progress.finish_step("全部快照已存在，跳过下载")
        return

    max_workers = min(24, max(4, (os.cpu_count() or 4) * 2))
    done_pending = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_download_one_snapshot, item, user_dir): item
            for item in pending
        }
        for future in as_completed(future_map):
            sid, ok, filename, err = future.result()
            if ok:
                files[sid] = filename
                failed.pop(sid, None)
            else:
                failed[sid] = err or "unknown_error"
            done_pending += 1
            progress.advance(1)
            if done_pending % 10 == 0:
                save_checkpoint(DATA_DIR / f"{checkpoint['username']}_checkpoint.json", checkpoint)

    progress.finish_step(f"新增下载 {done_pending} 条")


def parse_with_resume(items, user_dir: Path, checkpoint: dict, progress: ProgressBars):
    processed = checkpoint["processed"]

    already_done = 0
    for item in items:
        if item["status_id"] in processed:
            already_done += 1

    progress.start_step("解析与规则筛选", len(items))
    progress.set_done(already_done)

    new_done = 0
    for item in items:
        sid = item["status_id"]
        if sid in processed:
            continue

        file_path = build_snapshot_file_path(user_dir, item)
        if not file_path.exists():
            processed[sid] = None
            progress.advance(1)
            new_done += 1
            continue

        conv = extract_conversation_from_file(file_path)
        processed[sid] = conv if conv else None

        progress.advance(1)
        new_done += 1
        if new_done % 20 == 0:
            save_checkpoint(DATA_DIR / f"{checkpoint['username']}_checkpoint.json", checkpoint)

    progress.finish_step(f"本次新解析 {new_done} 条")


def dedupe_records(items, checkpoint: dict, progress: ProgressBars):
    processed = checkpoint["processed"]
    progress.start_step("整理并去重对话", len(items))

    records = []
    seen_pairs = set()
    for item in items:
        sid = item["status_id"]
        conv = processed.get(sid)
        if not conv or not isinstance(conv, dict):
            progress.advance(1)
            continue

        human = (conv.get("human") or "").strip()
        gpt = (conv.get("gpt") or "").strip()
        if not gpt:
            progress.advance(1)
            continue

        key = (human, gpt)
        if key in seen_pairs:
            progress.advance(1)
            continue
        seen_pairs.add(key)

        records.append(
            {
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": gpt},
                ]
            }
        )
        progress.advance(1)

    progress.finish_step(f"当前有效对话 {len(records)} 条")
    return records


def _extract_message_content(choice_message):
    content = (choice_message or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                chunks.append(c.get("text", ""))
        return "".join(chunks).strip()
    return ""


def call_chat_completion(messages, model=None, temperature=0.65, max_tokens=120, retries=3):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY，无法补全空 human。")

    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_MODEL_BASE).rstrip("/")
    model_name = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME)
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=DEFAULT_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("模型返回 choices 为空")
            content = _extract_message_content((choices[0] or {}).get("message") or {})
            if not content:
                raise RuntimeError("模型文本为空")
            return content
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            detail = err_body[:300].replace("\n", " ").strip()
            msg = f"HTTP {e.code}"
            if detail:
                msg += f": {detail}"
            if attempt == retries:
                raise RuntimeError(msg) from e
            time.sleep(1.2 * attempt)
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(str(e)) from e
            time.sleep(1.2 * attempt)

    raise RuntimeError("调用模型失败")


def synthesize_human_from_gpt(gpt_text: str) -> str:
    prompt = (
        "你是对话数据构造器。给定 assistant 的一句回复，请反推一个自然、合理、简短的 user 上一句。"
        "只输出 user 这句话本身，不要解释。"
        "禁止包含@用户名和链接。"
        "优先简体中文，长度2-120字。"
    )

    temperatures = (0.6, 0.75, 0.45)
    last_error = None
    for i in range(min(MAX_COMPLETION_TRIES, len(temperatures))):
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"assistant回复：{gpt_text}"},
            ]
            text = call_chat_completion(
                messages=messages,
                temperature=temperatures[i],
                max_tokens=120,
                retries=2,
            )
            text = clean_text(text)
            if len(text) > MAX_HUMAN_CHARS:
                text = text[:MAX_HUMAN_CHARS].strip()
            if is_meaningful_text(text) and text != gpt_text:
                return text
        except Exception as e:
            last_error = str(e)
            continue
    if last_error:
        preview = gpt_text[:40].replace("\n", " ")
        print(f"[补全失败] {last_error} | gpt='{preview}...'")
    return ""


def fill_empty_human_records(records, enabled: bool, progress: ProgressBars | None = None):
    if not enabled:
        return records, {"empty": 0, "filled": 0, "dropped": 0}

    empty_indices = []
    for idx, item in enumerate(records):
        convs = item.get("conversations") or []
        if len(convs) < 2:
            continue
        if convs[0].get("from") == "human" and not (convs[0].get("value") or "").strip():
            empty_indices.append(idx)

    if progress:
        progress.start_step("补全空 human（失败会删除）", max(len(empty_indices), 1))

    if not empty_indices:
        if progress:
            progress.finish_step("无空 human，跳过补全")
        return records, {"empty": 0, "filled": 0, "dropped": 0}

    dropped = set()
    filled = 0

    for i, idx in enumerate(empty_indices, start=1):
        convs = records[idx]["conversations"]
        gpt_text = (convs[1].get("value") or "").strip()

        if not is_meaningful_text(gpt_text):
            dropped.add(idx)
            if progress:
                progress.advance(1)
            continue

        generated = synthesize_human_from_gpt(gpt_text)
        if generated:
            convs[0]["value"] = generated
            filled += 1
        else:
            dropped.add(idx)

        if progress:
            progress.advance(1)

    filtered = [r for idx, r in enumerate(records) if idx not in dropped]

    if progress:
        progress.finish_step(f"空human {len(empty_indices)}，补全 {filled}，删除 {len(dropped)}")

    return filtered, {
        "empty": len(empty_indices),
        "filled": filled,
        "dropped": len(dropped),
    }


def export_records(username: str, records, progress: ProgressBars):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / f"{username}_sharegpt.json"

    progress.start_step("导出 ShareGPT JSON", 1)
    out_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.advance(1)
    progress.finish_step(str(out_file))
    return out_file


def collect_for_user(username: str, fill_empty_human: bool = False):
    username = username.strip().lstrip("@")
    if not username:
        return [], None

    user_dir = SNAPSHOT_ROOT / username
    user_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_steps = 6 if fill_empty_human else 5
    progress = ProgressBars(label=username, total_steps=total_steps)

    progress.start_step("获取 Wayback 快照索引", 1)
    rows = download_snapshots_for_user(username)
    progress.advance(1)
    progress.finish_step(f"索引返回 {len(rows)} 条")

    items = build_snapshot_items(rows)
    if not items:
        print(f"[{username}] 未找到可处理的状态快照")
        return [], None

    checkpoint_path = DATA_DIR / f"{username}_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, username)

    parallel_download_snapshots(items, user_dir, checkpoint, progress)
    save_checkpoint(checkpoint_path, checkpoint)

    parse_with_resume(items, user_dir, checkpoint, progress)
    save_checkpoint(checkpoint_path, checkpoint)

    records = dedupe_records(items, checkpoint, progress)

    fill_stats = {"empty": 0, "filled": 0, "dropped": 0}
    if fill_empty_human:
        records, fill_stats = fill_empty_human_records(records, enabled=True, progress=progress)
    else:
        before_count = len(records)
        records = [
            r for r in records
            if ((r.get("conversations") or [{}])[0].get("value") or "").strip()
        ]
        dropped_empty = before_count - len(records)
        if dropped_empty > 0:
            print(f"[{username}] 未启用补全，已移除空human样本 {dropped_empty} 条")

    out_file = export_records(username, records, progress)
    save_checkpoint(checkpoint_path, checkpoint)

    print(
        f"[{username}] 完成。对话 {len(records)} 条"
        + (f"；空human补全 {fill_stats['filled']}，删除 {fill_stats['dropped']}" if fill_empty_human else "")
    )

    return records, out_file


def parse_usernames(raw: str):
    if not raw:
        return []
    parts = re.split(r"[,，\s]+", raw.strip())
    return [p.lstrip("@").strip() for p in parts if p.strip()]


def prompt_yes_no(text: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{text} {suffix}: ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes", "1", "true")


def load_json_file(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def merge_dataset_files(json_paths, out_path: Path):
    merged = []
    seen = set()
    for path in json_paths:
        rows = load_json_file(path)
        for item in rows:
            convs = item.get("conversations") or []
            if len(convs) < 2:
                continue
            human = (convs[0].get("value") or "").strip()
            gpt = (convs[1].get("value") or "").strip()
            key = (human, gpt)
            if not gpt or key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "conversations": [
                        {"from": "human", "value": human},
                        {"from": "gpt", "value": gpt},
                    ]
                }
            )
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, len(merged)


def run_cmd(cmd, cwd: Path | None = None, check: bool = True):
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"[执行] {cmd_str}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def get_venv_bin_paths(venv_dir: Path):
    if os.name == "nt":
        python_bin = venv_dir / "Scripts" / "python.exe"
        pip_bin = venv_dir / "Scripts" / "pip.exe"
        cli_bin = venv_dir / "Scripts" / "llamafactory-cli.exe"
    else:
        python_bin = venv_dir / "bin" / "python"
        pip_bin = venv_dir / "bin" / "pip"
        cli_bin = venv_dir / "bin" / "llamafactory-cli"
    return python_bin, pip_bin, cli_bin


def ensure_llamafactory_installed(repo_dir: Path):
    if not repo_dir.exists():
        run_cmd(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", str(repo_dir)])

    venv_dir = repo_dir / ".venv"
    python_bin, pip_bin, cli_bin = get_venv_bin_paths(venv_dir)

    if not python_bin.exists():
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])

    run_cmd([str(pip_bin), "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=repo_dir)

    try:
        run_cmd([str(pip_bin), "install", "-e", ".[torch,metrics]"], cwd=repo_dir)
    except Exception:
        run_cmd([str(pip_bin), "install", "-e", "."], cwd=repo_dir)

    return python_bin, cli_bin


def infer_template_from_model(model_name: str) -> str:
    m = (model_name or "").lower()
    if "qwen" in m:
        return "qwen"
    if "llama-3" in m or "llama3" in m:
        return "llama3"
    if "glm" in m:
        return "chatglm"
    return "qwen"


def write_llamafactory_dataset_info(dataset_dir: Path, dataset_name: str, dataset_file_name: str):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    info_path = dataset_dir / "dataset_info.json"
    info = {}
    if info_path.exists():
        try:
            loaded = json.loads(info_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                info = loaded
        except Exception:
            info = {}

    info[dataset_name] = {
        "file_name": dataset_file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }

    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def suggest_train_hparams(sample_count: int):
    if sample_count >= 8000:
        epochs = 2.0
    elif sample_count >= 3000:
        epochs = 3.0
    else:
        epochs = 4.0

    return {
        "num_train_epochs": epochs,
        "learning_rate": "2.0e-4",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "cutoff_len": 2048,
        "quantization_bit": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
    }


def write_train_yaml(config_path: Path, model_name: str, template: str, dataset_name: str, dataset_dir: Path, output_dir: Path, sample_count: int):
    hp = suggest_train_hparams(sample_count)

    yaml_text = f"""### model
model_name_or_path: {model_name}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: {dataset_name}
template: {template}
dataset_dir: {dataset_dir.as_posix()}
cutoff_len: {hp['cutoff_len']}
max_samples: {sample_count}
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: {output_dir.as_posix()}
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_total_limit: 3

### train
per_device_train_batch_size: {hp['per_device_train_batch_size']}
gradient_accumulation_steps: {hp['gradient_accumulation_steps']}
learning_rate: {hp['learning_rate']}
num_train_epochs: {hp['num_train_epochs']}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.03
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 200

### qlora
quantization_bit: {hp['quantization_bit']}
lora_rank: {hp['lora_rank']}
lora_alpha: {hp['lora_alpha']}
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml_text, encoding="utf-8")


def run_training_pipeline(dataset_json: Path, sample_count: int):
    if sample_count <= 0:
        print("数据为空，跳过训练")
        return

    repo_dir = SCRIPT_DIR / "LLaMA-Factory"
    assets_dir = OUTPUT_DIR / "llamafactory_assets"
    dataset_dir = assets_dir / "datasets"
    config_dir = assets_dir / "configs"
    model_out_dir = OUTPUT_DIR / "models" / "sft_lora"

    progress = ProgressBars(label="train", total_steps=6)

    progress.start_step("准备/安装 LLaMA-Factory 环境", 1)
    python_bin, cli_bin = ensure_llamafactory_installed(repo_dir)
    progress.advance(1)
    progress.finish_step(str(repo_dir))

    progress.start_step("准备并注册数据集", 1)
    dataset_name = f"wayback_sharegpt_{int(time.time())}"
    dataset_file_name = f"{dataset_name}.json"
    dataset_file_path = dataset_dir / dataset_file_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_json, dataset_file_path)
    write_llamafactory_dataset_info(dataset_dir, dataset_name, dataset_file_name)
    progress.advance(1)
    progress.finish_step(dataset_file_name)

    progress.start_step("生成训练配置", 1)
    model_name = input("输入基础模型（默认 Qwen/Qwen2.5-7B-Instruct）: ").strip() or "Qwen/Qwen2.5-7B-Instruct"
    template = infer_template_from_model(model_name)
    config_path = config_dir / f"{dataset_name}_sft.yaml"
    write_train_yaml(
        config_path=config_path,
        model_name=model_name,
        template=template,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        output_dir=model_out_dir,
        sample_count=sample_count,
    )
    progress.advance(1)
    progress.finish_step(str(config_path))

    progress.start_step("启动训练", 1)
    if cli_bin.exists():
        train_cmd = [str(cli_bin), "train", str(config_path)]
    else:
        train_cmd = [str(python_bin), "-m", "llamafactory.cli", "train", str(config_path)]
    run_cmd(train_cmd, cwd=repo_dir)
    progress.advance(1)
    progress.finish_step("训练完成")

    progress.start_step("启动 WebUI", 1)
    if cli_bin.exists():
        webui_cmd = [str(cli_bin), "webui"]
    else:
        webui_cmd = [str(python_bin), "-m", "llamafactory.cli", "webui"]
    subprocess.Popen(webui_cmd, cwd=str(repo_dir))
    progress.advance(1)
    progress.finish_step("WebUI 已启动")

    progress.start_step("结束", 1)
    progress.advance(1)
    progress.finish_step("流程完成")


def main():
    input_key = input("请输入 OPENAI_API_KEY（留空跳过）: ").strip()
    if input_key:
        os.environ["OPENAI_API_KEY"] = input_key

    raw = input("请输入用户名（可多个，逗号分隔）: ")
    usernames = parse_usernames(raw)
    if not usernames:
        print("未提供有效用户名")
        return

    fill_empty_human = prompt_yes_no("是否补全空human（失败样本会删除）", default=False)
    if fill_empty_human and not os.getenv("OPENAI_API_KEY", "").strip():
        print("未检测到 OPENAI_API_KEY，无法补全，自动关闭补全。")
        fill_empty_human = False

    out_files = []
    total_pairs = 0

    for username in usernames:
        try:
            records, out_file = collect_for_user(username, fill_empty_human=fill_empty_human)
            total_pairs += len(records)
            if out_file:
                out_files.append(out_file)
        except Exception as exc:
            print(f"[{username}] 处理失败: {exc}")

    print(f"总计收集对话对: {total_pairs}")
    if not out_files:
        return

    merged_path, merged_count = merge_dataset_files(
        out_files,
        DATA_DIR / "all_users_sharegpt.json",
    )
    print(f"合并数据集: {merged_path} ({merged_count} 条)")

    if prompt_yes_no("是否在数据整理后直接开始训练并自动启动 WebUI", default=False):
        try:
            run_training_pipeline(merged_path, merged_count)
        except Exception as exc:
            print(f"训练流程失败: {exc}")


if __name__ == "__main__":
    main()
