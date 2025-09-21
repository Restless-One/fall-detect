from __future__ import annotations
import os
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Deque, Dict, Any, Optional, AsyncGenerator
import logging

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import math
import subprocess


# --- Optional: import model deps; handle absence gracefully for dev ---
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # allow app to boot without ultralytics installed

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

app = FastAPI(title="Fall Detection Backend (FastAPI MVP)", version="0.1.0")
log = logging.getLogger("uvicorn.error")

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Config ------------------
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
VIDEO_DIR = os.path.join(STORAGE_DIR, "videos")
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONF_TH = float(os.getenv("CONF_TH", "0.5"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
WINDOW = int(os.getenv("WINDOW", "10"))
K_FALLEN = int(os.getenv("K_FALLEN", "6"))
K_STANDING = int(os.getenv("K_STANDING", "6"))
NO_STAND_TIMEOUT_S = float(os.getenv("NO_STAND_TIMEOUT_S", "20"))
MAX_RECENT_EVENTS = int(os.getenv("MAX_RECENT_EVENTS", "100"))

# ------------------ File helpers for annotated rendering ------------------
from typing import Dict as _Dict, Any as _Any

def _video_fs_paths(video_id: str, src_path: str) -> _Dict[str, str]:
    """Return canonical per-video paths near the saved source file."""
    base_dir = os.path.dirname(src_path)
    meta_path = os.path.join(base_dir, f"{video_id}_meta.json")
    annotated_path = os.path.join(base_dir, f"{video_id}_annotated.mp4")
    return {"base_dir": base_dir, "meta": meta_path, "annotated": annotated_path}


def _public_url_from_path(abs_path: str) -> str:
    """Map an absolute path inside STORAGE_DIR to a public /storage URL."""
    abs_storage = os.path.abspath(STORAGE_DIR)
    abs_path_norm = os.path.abspath(abs_path)
    if abs_path_norm.startswith(abs_storage):
        rel = os.path.relpath(abs_path_norm, abs_storage)
        return f"/storage/{rel.replace(os.sep, '/')}"
    return ""

# ------------------ In-Memory State ------------------
# Minimal in-memory DB for MVP (swap to SQLite later)
videos: Dict[str, Dict[str, Any]] = {}
# Per-video async event queues for SSE
_event_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

# Global model & class ids
model: Optional[Any] = None
fallen_id: Optional[int] = None
standing_id: Optional[int] = None
has_standing_class: bool = False

# ------------------ Utilities ------------------

def ensure_dirs() -> None:
    os.makedirs(VIDEO_DIR, exist_ok=True)


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def dummy_hr(mode: str = "recovered") -> int:
    import random
    if mode == "timeout":
        return random.randint(110, 130)
    return random.randint(70, 95)


async def put_event(video_id: str, event: Dict[str, Any]) -> None:
    # add unique id for deduplication on client
    if "event_id" not in event:
        event["event_id"] = uuid.uuid4().hex
    # store in recent ring buffer and push to SSE queue
    recents: Deque[Dict[str, Any]] = videos[video_id].setdefault("recent_events", deque(maxlen=MAX_RECENT_EVENTS))
    recents.append(event)
    await _event_queues[video_id].put(event)


def _find_class_id(names: Dict[Any, Any], target: str) -> Optional[int]:
    target = target.strip().lower()
    for k, v in names.items():
        if str(v).strip().lower() == target:
            try:
                return int(k)
            except Exception:
                continue
    return None


def _frame_vote(r, conf_th: float):
    """Aggregate per-frame predictions into booleans + extras."""
    has_fallen = False
    has_standing = False
    max_fallen_conf = 0.0
    best_fallen_box = None

    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        cls = r.boxes.cls.cpu().tolist()
        conf = r.boxes.conf.cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()
        for c, p, bb in zip(cls, conf, xyxy):
            c = int(c)
            if fallen_id is not None and c == fallen_id and p >= conf_th:
                has_fallen = True
                if p > max_fallen_conf:
                    max_fallen_conf = p
                    best_fallen_box = bb
            if has_standing_class and standing_id is not None and c == standing_id and p >= conf_th:
                has_standing = True
    return has_fallen, has_standing, float(max_fallen_conf), best_fallen_box


# ------------------ Startup ------------------
@app.on_event("startup")
async def _startup() -> None:
    ensure_dirs()
    global model, fallen_id, standing_id, has_standing_class
    if YOLO is None:
        log.warning("ultralytics not installed; model won't load. Install 'ultralytics'.")
        return
    if not os.path.exists(MODEL_PATH):
        log.warning(f"Model file not found at {MODEL_PATH}; /health will report model_loaded:false")
        return
    try:
        model = YOLO(MODEL_PATH)
        names = getattr(model.model, "names", getattr(model, "names", {})) or {}
        fallen_id = _find_class_id(names, "fallen")
        standing_id = _find_class_id(names, "standing")
        has_standing_class = standing_id is not None
        log.info(f"Model loaded. fallen_id={fallen_id}, standing_id={standing_id}")
    except Exception as e:  # pragma: no cover
        log.error(f"Failed to load model: {e}")
        model = None

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
# ------------------ Routes ------------------
@app.get("/")
async def frontend():
    # Try root index.html; fall back to static/index.html
    base_dir = os.path.dirname(__file__)
    root_index = os.path.join(base_dir, "index.html")
    static_index = os.path.join(base_dir, "static", "index.html")
    if os.path.exists(root_index):
        return FileResponse(root_index)
    elif os.path.exists(static_index):
        return FileResponse(static_index)
    else:
        # Last resort: return a tiny page hint
        from fastapi.responses import HTMLResponse
        html = """
        <!doctype html><html><body>
        <h3>index.html Not Found</h3>
        <p>Please put a <code>index.html</code> file  in the <code>static/index.html</code>。</p>
        </body></html>
        """
        return HTMLResponse(html, status_code=404)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "fallen_id": fallen_id,
        "standing_id": standing_id,
        "has_standing_class": has_standing_class,
        "storage_dir": os.path.abspath(STORAGE_DIR),
    }


@app.post("/videos")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type not in {"video/mp4", "application/octet-stream", "video/quicktime", "video/x-matroska"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    # Save to disk
    video_id = str(uuid.uuid4())
    safe_name = f"{video_id}_{os.path.basename(file.filename)}"
    dst_path = os.path.join(VIDEO_DIR, safe_name)
    ensure_dirs()
    try:
        with open(dst_path, "wb") as out:
            out.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create video record
    videos[video_id] = {
        "id": video_id,
        "filename": file.filename,
        "path": dst_path,
        "status": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "recent_events": deque(maxlen=MAX_RECENT_EVENTS),
    }

    paths = _video_fs_paths(video_id, dst_path)
    videos[video_id]["meta_path"] = paths["meta"]
    videos[video_id]["annotated_path"] = paths["annotated"]
    videos[video_id]["render_status"] = "pending"

    # Kick off background processing
    background_tasks.add_task(process_video_task, video_id)

    return {"video_id": video_id, "status": "queued"}


@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    rec = videos.get(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    payload = {k: v for k, v in rec.items() if k != "recent_events"}
    payload["recent_events"] = list(rec.get("recent_events", []))
    # add public URLs if paths exist
    annotated_path = rec.get("annotated_path")
    meta_path = rec.get("meta_path")
    if annotated_path and os.path.exists(annotated_path):
        payload["annotated_url"] = _public_url_from_path(annotated_path)
    if meta_path and os.path.exists(meta_path):
        payload["meta_url"] = _public_url_from_path(meta_path)
    if rec.get("render_status"):
        payload["render_status"] = rec["render_status"]
    return payload


@app.post("/videos/{video_id}/render")
async def render_video(video_id: str):
    rec = videos.get(video_id)
    if not rec:
        raise HTTPException(status_code=404, detail="video not found")
    rec["render_status"] = "running"
    result = await asyncio.to_thread(_render_annotated_mp4, video_id)
    if not result.get("ok"):
        rec["render_status"] = "failed"
        rec["error_msg"] = result.get("error", "render failed")
        raise HTTPException(status_code=500, detail=rec["error_msg"])
    return {"annotated_url": result.get("url"), "render_status": rec["render_status"]}


@app.get("/events/stream")
async def events_stream(request: Request, video_id: str, replay: bool = Query(True)):
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="video not found")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        queue = _event_queues[video_id]
        # On connect, optionally replay recent events
        if replay:
            for e in list(videos[video_id].get("recent_events", [])):
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode("utf-8")
        while True:
            if await request.is_disconnected():
                break
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
            except asyncio.TimeoutError:
                # Keep-alive comment for SSE
                yield b": keep-alive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



async def process_video_task(video_id: str) -> None:
    rec = videos.get(video_id)
    if not rec:
        return
    rec["status"] = "running"
    rec["updated_at"] = now_iso()

    path = rec["path"]

    if model is None:
        # Cannot process without model
        err = {"type": "ERROR", "ts": now_iso(), "message": "model not loaded"}
        await put_event(video_id, err)
        rec["status"] = "failed"
        rec["error_msg"] = "model not loaded"
        rec["updated_at"] = now_iso()
        return

    if cv2 is None:
        err = {"type": "ERROR", "ts": now_iso(), "message": "opencv not installed"}
        await put_event(video_id, err)
        rec["status"] = "failed"
        rec["error_msg"] = "opencv not installed"
        rec["updated_at"] = now_iso()
        return


    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
    except Exception:
        fps = 25.0

    def f2t(i: int) -> str:
        return str(timedelta(seconds=i / max(fps, 1.0)))

    fallen_win: Deque[bool] = deque(maxlen=WINDOW)
    standing_win: Deque[bool] = deque(maxlen=WINDOW)

    in_fall = False
    fall_start_frame: Optional[int] = None
    timeout_alerted = False

    frame_idx = -1
    meta_frames = []  # per-frame annotations for later rendering

    try:

        async for r in _predict_stream_async(path):
            frame_idx += 1
            has_fallen, has_standing, max_fc, best_box = _frame_vote(r, CONF_TH)


            if best_box is not None:
                meta_frames.append({
                    "frame": frame_idx,
                    "t": (frame_idx / max(fps, 1.0)),
                    "boxes": [
                        {
                            "x1": float(best_box[0]), "y1": float(best_box[1]),
                            "x2": float(best_box[2]), "y2": float(best_box[3]),
                            "label": "FALLEN", "score": float(max_fc)
                        }
                    ]
                })

            fallen_win.append(has_fallen)
            standing_win.append(has_standing)


            if len(fallen_win) < fallen_win.maxlen:
                continue
            can_check_recovered = True
            if has_standing_class and len(standing_win) < standing_win.maxlen:
                can_check_recovered = False

            # A) FALL_DETECTED
            if (not in_fall) and sum(fallen_win) >= K_FALLEN:
                in_fall = True
                fall_start_frame = frame_idx
                timeout_alerted = False
                # Optional: clear standing history on new fall to prevent premature recovery
                standing_win.clear()
                evt = {"type": "FALL_DETECTED", "frame": frame_idx, "ts": f2t(frame_idx),
                       "max_frame_score": round(max_fc, 3), "bbox": best_box}
                await put_event(video_id, evt)

            # B) RECOVERED
            recovered = False
            if has_standing_class:
                recovered = sum(standing_win) >= K_STANDING
            else:
                recovered = sum(fallen_win) == 0  # no fallen in the window

            if in_fall and can_check_recovered and recovered:
                in_fall = False
                dur = (frame_idx - (fall_start_frame or frame_idx)) / max(fps, 1.0)
                evt = {"type": "RECOVERED", "frame": frame_idx, "ts": f2t(frame_idx),
                       "fall_duration_sec": round(dur, 2), "rppg_heart_rate_bpm": dummy_hr("recovered")}
                await put_event(video_id, evt)
                fall_start_frame = None
                timeout_alerted = False

            # C) NO_STAND_TIMEOUT
            if in_fall and (fall_start_frame is not None) and (not timeout_alerted):
                elapsed = (frame_idx - fall_start_frame) / max(fps, 1.0)
                if elapsed >= NO_STAND_TIMEOUT_S:
                    evt = {"type": "NO_STAND_TIMEOUT", "frame": frame_idx, "ts": f2t(frame_idx),
                           "elapsed_sec": round(elapsed, 2), "rppg_heart_rate_bpm": dummy_hr("timeout")}
                    await put_event(video_id, evt)
                    timeout_alerted = True

        # After stream ends
        if in_fall and not timeout_alerted:
            elapsed = 0.0
            if fall_start_frame is not None:
                elapsed = (frame_idx - fall_start_frame) / max(fps, 1.0)
            evt = {"type": "VIDEO_ENDED_NO_RECOVERY", "frame": frame_idx, "ts": f2t(frame_idx),
                   "elapsed_sec": round(elapsed, 2), "rppg_heart_rate_bpm": dummy_hr("timeout")}
            await put_event(video_id, evt)

        try:
            paths = _video_fs_paths(video_id, path)
            meta_payload = {"fps": float(fps), "frames": meta_frames}
            with open(paths["meta"], "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, ensure_ascii=False)
            rec["meta_path"] = paths["meta"]
            rec["render_status"] = rec.get("render_status") or "pending"
        except Exception as _e:
            log.warning(f"Failed to write meta for {video_id}: {_e}")

        rec["status"] = "finished"
        rec["updated_at"] = now_iso()

    except Exception as e:
        rec["status"] = "failed"
        rec["error_msg"] = str(e)
        rec["updated_at"] = now_iso()
        await put_event(video_id, {"type": "ERROR", "ts": now_iso(), "message": str(e)})


async def _predict_stream_async(path: str):
    """Wrap YOLO.predict(stream=True) in an async generator using to_thread."""
    assert model is not None

    # We create a sync generator first
    def _sync_gen():
        yield from model.predict(source=path, stream=True, conf=CONF_TH, imgsz=IMG_SIZE, verbose=False, max_det=5)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            for item in _sync_gen():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, StopAsyncIteration)

    # Run producer in a thread
    await asyncio.to_thread(_producer)

    while True:
        item = await queue.get()
        if item is StopAsyncIteration:
            break
        yield item

# ------------------ Rendering (annotated MP4) ------------------

def _render_annotated_mp4(video_id: str) -> dict:
    rec = videos.get(video_id)
    if not rec:
        return {"ok": False, "error": "video not found"}
    src_path = rec.get("path")
    if not src_path or not os.path.exists(src_path):
        return {"ok": False, "error": "source video missing"}

    paths = _video_fs_paths(video_id, src_path)
    meta_path = rec.get("meta_path") or paths["meta"]
    out_path = paths["annotated"]

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"meta load failed: {e}"}

    if cv2 is None:
        return {"ok": False, "error": "opencv not installed"}

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return {"ok": False, "error": "cannot open source video"}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS) or meta.get("fps") or 25.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"ok": False, "error": "cannot open VideoWriter (mp4v)"}

    frames_meta = {int(it.get("frame", -1)): it for it in meta.get("frames", []) if int(it.get("frame", -1)) >= 0}

    frame_idx = -1
    # Heartbeat text control
    last_hr_update = -1
    hr_text = f"HR: {dummy_hr()} bpm"


    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            current_time_sec = frame_idx / fps
            if int(current_time_sec) != last_hr_update:
                last_hr_update = int(current_time_sec)
                hr_text = f"HR: {dummy_hr()} bpm"

            dat = frames_meta.get(frame_idx)
            if dat and dat.get("boxes"):
                for b in dat["boxes"]:
                    x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{b.get('label','')}: {b.get('score',0):.2f}"
                    y_text = max(0, y1 - 8)

                    # Common text styling
                    _font = cv2.FONT_HERSHEY_SIMPLEX
                    _scale = 0.5
                    _th = 1

                    # Draw main label (FALLEN + score)
                    cv2.putText(frame, label, (x1, y_text), _font, _scale, (0, 255, 0), _th, cv2.LINE_AA)

                    # Place heartbeat text right after the label on the same baseline
                    label_size, _ = cv2.getTextSize(label, _font, _scale, _th)
                    hr_size, _ = cv2.getTextSize(hr_text, _font, _scale, _th)
                    hr_x = x1 + label_size[0] + 8  # 8px gap after label
                    hr_y = y_text

                    # If overflowing frame width, drop it just below the label as a fallback
                    if hr_x + hr_size[0] > width - 4:
                        hr_x = x1
                        hr_y = min(height - 4, y_text + 14)

                    cv2.putText(frame, hr_text, (hr_x, hr_y), _font, _scale, (0, 0, 255), _th, cv2.LINE_AA)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    # Re-encode to H.264 MP4 with faststart for browser playback
    final_out = out_path
    h264_out = out_path + ".h264.mp4"
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_out,
            "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
            "-movflags", "+faststart",
            h264_out,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(h264_out):
            final_out = h264_out
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
        else:
            # Fallback: simple move (may not play in all browsers)
            if os.path.exists(out_path):
                os.remove(out_path)
            os.replace(tmp_out, out_path)
            final_out = out_path
    except FileNotFoundError:
        # ffmpeg not installed; fallback to simple move
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
            os.replace(tmp_out, out_path)
            final_out = out_path
        except Exception as e:
            return {"ok": False, "error": f"finalize failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"ffmpeg failed: {e}"}

    url = _public_url_from_path(final_out)
    rec["annotated_path"] = final_out
    rec["annotated_url"] = url
    rec["render_status"] = "finished"
    rec["updated_at"] = now_iso()
    return {"ok": True, "url": url, "path": final_out}
