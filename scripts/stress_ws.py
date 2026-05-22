#!/usr/bin/env python3
"""Stress test an already running WhisperLive WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly


RATE = 16000


@dataclass
class ClientStats:
    name: str
    ready: bool = False
    closed: bool = False
    timeout: bool = False
    errors: int = 0
    segment_msgs: int = 0
    segment_items: int = 0
    translation_msgs: int = 0
    translation_items: int = 0
    first_message_at: float | None = None
    ready_at: float | None = None
    last_message_at: float | None = None
    error_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.ready,
            "closed": self.closed,
            "timeout": self.timeout,
            "errors": self.errors,
            "segment_msgs": self.segment_msgs,
            "segment_items": self.segment_items,
            "translation_msgs": self.translation_msgs,
            "translation_items": self.translation_items,
            "first_message_at": self.first_message_at,
            "ready_at": self.ready_at,
            "last_message_at": self.last_message_at,
            "error_messages": self.error_messages,
        }


def load_audio(path: Path) -> tuple[np.ndarray, float]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sample_rate != RATE:
        divisor = math.gcd(sample_rate, RATE)
        data = resample_poly(data, RATE // divisor, sample_rate // divisor).astype(np.float32)

    audio = np.ascontiguousarray(data.astype(np.float32))
    return audio, len(audio) / RATE


def build_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "uid": str(uuid.uuid4()),
        "language": args.language,
        "task": args.task,
        "model": args.model,
        "use_vad": args.use_vad,
        "send_last_n_segments": args.send_last_n_segments,
        "no_speech_thresh": args.no_speech_thresh,
        "clip_audio": args.clip_audio,
        "same_output_threshold": args.same_output_threshold,
        "enable_translation": args.enable_translation,
        "target_language": args.target_language,
        "translation_provider": args.translation_provider,
        "zh_en_model_path": args.zh_en_model_path,
        "en_zh_model_path": args.en_zh_model_path,
        "hotwords": args.hotwords,
        "enable_diarization": False,
        "max_speakers": 10,
        "word_timestamps": args.word_timestamps,
    }


async def receive_messages(ws: Any, stats: ClientStats, started_at: float) -> None:
    try:
        async for message in ws:
            now = time.monotonic() - started_at
            stats.first_message_at = stats.first_message_at if stats.first_message_at is not None else now
            stats.last_message_at = now

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            if payload.get("message") == "SERVER_READY" or payload.get("status") == "SERVER_READY":
                stats.ready = True
                stats.ready_at = now

            if payload.get("status") == "ERROR":
                stats.errors += 1
                if payload.get("message"):
                    stats.error_messages.append(str(payload["message"]))

            if "segments" in payload:
                stats.segment_msgs += 1
                stats.segment_items += len(payload.get("segments") or [])

            if "translated_segments" in payload:
                stats.translation_msgs += 1
                stats.translation_items += len(payload.get("translated_segments") or [])
    except Exception as exc:
        stats.errors += 1
        stats.error_messages.append(repr(exc))
    finally:
        stats.closed = True


async def run_client(
    name: str,
    uri: str,
    audio: np.ndarray,
    args: argparse.Namespace,
    started_at: float,
) -> ClientStats:
    stats = ClientStats(name=name)
    async with websockets.connect(uri, max_size=None) as ws:
        options = build_options(args)
        await ws.send(json.dumps(options))

        async def receiver() -> None:
            await receive_messages(ws, stats, started_at)

        recv_task = asyncio.create_task(receiver())

        ready_deadline = time.monotonic() + args.ready_timeout
        while not stats.ready and not recv_task.done():
            if time.monotonic() >= ready_deadline:
                stats.timeout = True
                recv_task.cancel()
                return stats
            await asyncio.sleep(0.05)

        if not stats.ready:
            stats.timeout = True
            recv_task.cancel()
            return stats

        chunk_samples = max(1, int(RATE * args.chunk_seconds))
        for offset in range(0, len(audio), chunk_samples):
            packet = audio[offset : offset + chunk_samples]
            await ws.send(packet.astype(np.float32, copy=False).tobytes())
            if args.realtime:
                await asyncio.sleep(len(packet) / RATE)

        await asyncio.sleep(args.post_audio_wait)
        await ws.close()
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    return stats


async def run_stress(args: argparse.Namespace) -> tuple[list[ClientStats], float, float]:
    audio, audio_sec = load_audio(Path(args.audio))
    uri = f"ws://{args.host}:{args.port}"
    started_at = time.monotonic()
    results = await asyncio.gather(
        *[
            run_client(f"client-{idx + 1}", uri, audio, args, started_at)
            for idx in range(args.clients)
        ],
        return_exceptions=True,
    )
    wall_sec = time.monotonic() - started_at

    stats: list[ClientStats] = []
    for idx, result in enumerate(results):
        if isinstance(result, ClientStats):
            stats.append(result)
        else:
            failed = ClientStats(name=f"client-{idx + 1}", errors=1, timeout=True)
            failed.error_messages.append(repr(result))
            stats.append(failed)
    return stats, audio_sec, wall_sec


def summarize(args: argparse.Namespace, stats: list[ClientStats], audio_sec: float, wall_sec: float) -> dict[str, Any]:
    clients_ready = sum(1 for item in stats if item.ready)
    clients_failed = sum(1 for item in stats if item.timeout or item.errors > 0 or not item.ready)
    total_translation_msgs = sum(item.translation_msgs for item in stats)
    total_errors = sum(item.errors for item in stats)
    rt_factor = wall_sec / audio_sec if audio_sec > 0 else 0.0

    success = (
        clients_ready == args.clients
        and clients_failed == 0
        and total_errors == 0
        and rt_factor <= args.max_rt_factor
    )
    if args.enable_translation:
        success = success and all(item.translation_msgs >= args.min_translation_msgs for item in stats)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "success": success,
        "host": args.host,
        "port": args.port,
        "audio": args.audio,
        "audio_sec": round(audio_sec, 3),
        "wall_sec": round(wall_sec, 3),
        "rt_factor": round(rt_factor, 3),
        "clients_total": args.clients,
        "clients_ready": clients_ready,
        "clients_failed": clients_failed,
        "total_errors": total_errors,
        "total_segment_msgs": sum(item.segment_msgs for item in stats),
        "total_segment_items": sum(item.segment_items for item in stats),
        "total_translation_msgs": total_translation_msgs,
        "total_translation_items": sum(item.translation_items for item in stats),
        "criteria": {
            "max_rt_factor": args.max_rt_factor,
            "min_translation_msgs": args.min_translation_msgs if args.enable_translation else None,
        },
        "clients": [item.to_dict() for item in stats],
    }


def render_table(summary: dict[str, Any]) -> str:
    lines = [
        "summary: "
        f"success={summary['success']} "
        f"ready={summary['clients_ready']}/{summary['clients_total']} "
        f"failed={summary['clients_failed']} "
        f"audio={summary['audio_sec']}s "
        f"wall={summary['wall_sec']}s "
        f"rt={summary['rt_factor']} "
        f"segments={summary['total_segment_msgs']} "
        f"translations={summary['total_translation_msgs']} "
        f"errors={summary['total_errors']}"
    ]
    lines.append("client	ready	seg_msgs	seg_items	tr_msgs	tr_items	errors	timeout")
    for item in summary["clients"]:
        lines.append(
            f"{item['name']}	{item['ready']}	{item['segment_msgs']}	"
            f"{item['segment_items']}	{item['translation_msgs']}	"
            f"{item['translation_items']}	{item['errors']}	{item['timeout']}"
        )
    return "\n".join(lines)


def write_log(args: argparse.Namespace, summary: dict[str, Any], table: str) -> Path | None:
    if args.no_log:
        return None

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = log_dir / args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(args.audio).stem
        log_path = log_dir / f"stress_{timestamp}_{audio_name}_{args.clients}clients.log"

    log_path.write_text(
        table + "\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test an already running WhisperLive server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--audio", default="test_zn.wav")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--model", default="model/asr/small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--enable_translation", action="store_true")
    parser.add_argument("--target_language", default="en")
    parser.add_argument("--translation_provider", default="helsinki_zh_en")
    parser.add_argument("--zh_en_model_path", default="model/opus-mt-zh-en")
    parser.add_argument("--en_zh_model_path", default="model/opus-mt-en-zh")
    parser.add_argument("--same_output_threshold", type=int, default=2)
    parser.add_argument("--send_last_n_segments", type=int, default=10)
    parser.add_argument("--no_speech_thresh", type=float, default=0.45)
    parser.add_argument("--chunk_seconds", type=float, default=0.5)
    parser.add_argument("--post_audio_wait", type=float, default=8.0)
    parser.add_argument("--ready_timeout", type=float, default=120.0)
    parser.add_argument("--max_rt_factor", type=float, default=1.5)
    parser.add_argument("--min_translation_msgs", type=int, default=1)
    parser.add_argument("--hotwords", default=None)
    parser.add_argument("--word_timestamps", action="store_true")
    parser.add_argument("--clip_audio", action="store_true")
    parser.add_argument("--no_vad", dest="use_vad", action="store_false")
    parser.add_argument("--no_realtime", dest="realtime", action="store_false")
    parser.add_argument("--json", action="store_true", help="Print full JSON only.")
    parser.add_argument("--log_dir", default="scripts/stress_logs")
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.set_defaults(use_vad=True, realtime=True)
    args = parser.parse_args()

    if args.clients < 1:
        parser.error("--clients must be >= 1")
    if args.chunk_seconds <= 0:
        parser.error("--chunk_seconds must be > 0")
    if not Path(args.audio).is_file():
        parser.error(f"audio file not found: {args.audio}")
    return args


def main() -> int:
    args = parse_args()
    try:
        stats, audio_sec, wall_sec = asyncio.run(run_stress(args))
    except OSError as exc:
        print(f"connection error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"stress test failed: {exc}", file=sys.stderr)
        return 2

    summary = summarize(args, stats, audio_sec, wall_sec)
    table = render_table(summary)
    log_path = write_log(args, summary, table)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(table)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if log_path is not None:
        print(f"log_file={log_path}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
