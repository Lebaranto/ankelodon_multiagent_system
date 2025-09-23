import os, io, base64, json, tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from PIL import Image
import google.generativeai as genai
from langchain_core.tools import tool

# ======================== CONFIG & CORE ========================

def _configure() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GENAI_API_KEY) in environment")
    genai.configure(api_key=api_key)
    return api_key

def _clean_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").replace("json", "", 1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s

def _call_model(parts: List[Any], temperature: float, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Единая точка вызова модели. Возвращает dict с ключом "answer".
    """
    MODEL_NAME = model_name or os.getenv("GEMMA_MODEL", "gemma-3-27b-it")
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(parts, generation_config={"temperature": temperature})
    text = (getattr(resp, "text", None) or "").strip()
    try:
        return json.loads(_clean_json_text(text))
    except Exception:
        fixer = genai.GenerativeModel(MODEL_NAME)
        fix_prompt = (
            "Convert the following text into STRICT valid JSON matching schema {\"answer\": string}. "
            "Return ONLY JSON, no extra text:\n" + text
        )
        fix_resp = fixer.generate_content([{"text": fix_prompt}])
        return json.loads(_clean_json_text((getattr(fix_resp, "text", "") or "").strip()))

# ======================== VIDEO HELPERS (OpenCV-only) ========================

_VIDEO_QA_PROMPT = (
    "You will be given ONE video and a question about its visual content.\n"
    "Answer STRICTLY and CONCISELY based only on what is visible/audible in the provided video.\n"
    "If the video does not contain enough information, reply 'not enough information'.\n"
    "Return ONLY valid JSON with the schema:\n"
    "{\"answer\": string}\n"
)

def _uniform_sample_paths(paths: List[Path], k: int) -> List[Path]:
    n = len(paths)
    if n <= k:
        return paths
    idxs = [round(i*(n-1)/(k-1)) for i in range(k)]
    return [paths[i] for i in idxs]

def _ensure_png_bytes(img: Image.Image, max_pixels: int = 25_000_000) -> bytes:
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _image_bytes_to_part(img_bytes: bytes, mime: str = "image/png") -> Dict[str, Any]:
    return {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}

def _extract_frames_cv2(video_path: str, out_dir: Path, fps: float, start_s: float, duration_s: Optional[float]) -> List[Path]:
    """
    Извлекаем кадры через OpenCV (без системного ffmpeg).
    Требует: pip install opencv-python
    """
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    total_ms = (total_frames / in_fps) * 1000.0 if total_frames and in_fps else None

    start_ms = max(0.0, float(start_s) * 1000.0)
    end_ms = start_ms + float(duration_s) * 1000.0 if duration_s is not None else (total_ms or start_ms + 30_000.0)
    step_ms = 1000.0 / max(0.001, fps)  # период семплинга по ms

    t = start_ms
    idx = 0
    saved: List[Path] = []
    while t <= end_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, frame = cap.read()
        if not ok:
            break
        fp = out_dir / f"{idx:06d}.jpg"
        # JPEG сохраняем без ffmpeg
        ok = cv2.imwrite(str(fp), frame)
        if ok:
            saved.append(fp)
            idx += 1
        t += step_ms

    cap.release()
    if not saved:
        raise RuntimeError("No frames extracted (OpenCV).")
    return saved

def _frames_to_image_parts(frame_paths: List[Path], max_images: int) -> List[Dict[str, Any]]:
    """
    Прореживаем кадры до <= max_images и упаковываем как inline-изображения.
    """
    frame_paths = _uniform_sample_paths(frame_paths, k=max_images)
    out: List[Dict[str, Any]] = []
    for fp in frame_paths:
        img = Image.open(fp)
        img_bytes = _ensure_png_bytes(img)
        out.append(_image_bytes_to_part(img_bytes, "image/png"))
    return out

def _download_youtube_to_mp4(youtube_url: str, out_path: str) -> str:
    """
    Скачиваем YouTube через библиотеку yt_dlp (без системного ffmpeg).
    Требует: pip install yt-dlp
    Стараемся выбрать прогрессивный MP4 (single file), чтобы не потребовался mux.
    """
    from yt_dlp import YoutubeDL
    ydl_opts = {
        # выбираем ЛУЧШИЙ одиночный файл, предпочитая MP4 (без mux/ffmpeg)
        "format": "b[ext=mp4]/b",
        "outtmpl": out_path,
        "noprogress": True,
        "quiet": True,
        "nocheckcertificate": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        # yt-dlp может игнорировать outtmpl при некоторых шаблонах — подстрахуемся
        fn = ydl.prepare_filename(info)
    # Если получили другой путь, перенесём
    src = Path(fn)
    dst = Path(out_path)
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
    return str(dst)

def _get_client(api_key: Optional[str]):
    """
    Опционально: новый Google GenAI SDK (google-genai) для Files API в 'auto' режиме.
    Если нет — вернём None.
    """
    try:
        from google import genai as ggenai  # новый пакет "google-genai"
        return ggenai.Client(api_key=api_key)
    except Exception:
        return None

def _video_part_from_youtube(url: str) -> Dict[str, Any]:
    """Для mode='auto': передаём YouTube как file_data без скачивания."""
    return {"file_data": {"file_uri": url}}

def _video_part_from_file(path: str, api_key: Optional[str]) -> Dict[str, Any]:
    """
    Для mode='auto': загружаем локальный файл в Files API.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    client = _get_client(api_key)
    if client is not None and hasattr(client, "files"):
        try:
            f = client.files.upload(file=path)
            return {"file_data": {"file_uri": f.uri, "mime_type": getattr(f, "mime_type", None) or "video/mp4"}}
        except Exception:
            pass
    f = genai.upload_file(path=path)
    file_uri = getattr(f, "uri", None) or getattr(f, "file_uri", None)
    mime = getattr(f, "mime_type", None) or "video/mp4"
    return {"file_data": {"file_uri": file_uri, "mime_type": mime}}

# ======================== VIDEO QA TOOL (OpenCV frames по умолчанию) ========================

@tool
def video_qa_gemma(
    question: str,
    youtube_url: Optional[str] = None,
    video_path: Optional[str] = None,
    temperature: float = 0.2,
    model_name: Optional[str] = None,
    mode: Literal["frames", "auto"] = "frames",   # по умолчанию безопасный режим кадров (OpenCV)
    fps: float = 0.8,                              # 0.8 * 30s ≈ 24 кадров
    start_s: float = 0.0,
    duration_s: Optional[float] = 30.0,            # держим сегмент коротким
    max_images: int = 24,                          # < 32 — жёсткая крышка
) -> str:
    """
    Answer questions about the visual content of a video (YouTube URL or local file).

    Args:
      question: Natural language question about the video.
      youtube_url: Link to a YouTube video (exclusive with video_path).
      video_path: Local path to a video file.
      mode: "frames" (default, extracts ≤max_images frames with OpenCV) or "auto" (send whole video).
      fps/start_s/duration_s: Frame sampling parameters in "frames" mode.
      max_images: Max number of frames (<32). Default 24.

    Returns:
      JSON string: {"answer": "..."} (or "not enough information").

    Notes:
      - Provide exactly ONE of youtube_url or video_path.
      - Use "frames" mode to avoid API errors on models with image limits.
    """
    import json as _json
    try:
        api_key = _configure()

        if bool(youtube_url) == bool(video_path):
            return _json.dumps({"error": "Provide exactly ONE of youtube_url or video_path"})

        if mode == "auto":
            # Без OpenCV: отдаём видео целиком (иногда API внутри раздувает до >32 изображений).
            if youtube_url:
                video_part = _video_part_from_youtube(youtube_url)
            else:
                video_part = _video_part_from_file(video_path, api_key)
            parts = [video_part, {"text": _VIDEO_QA_PROMPT + "\nQuestion: " + question.strip()}]
            data = _call_model(parts, temperature, model_name=model_name)
        else:
            # OpenCV: извлекаем кадры и отправляем как <= max_images изображений
            tmp_video_path = None
            if youtube_url and not video_path:
                with tempfile.TemporaryDirectory(prefix="yt_") as td:
                    tmp_video_path = str(Path(td) / "video.mp4")
                    _download_youtube_to_mp4(youtube_url, tmp_video_path)
                    # внутри with мы не можем вернуть, поэтому делаем обработку ниже в том же блоке
                    frame_dir = Path(td) / "frames"
                    files = _extract_frames_cv2(tmp_video_path, frame_dir, fps=fps, start_s=start_s, duration_s=duration_s)
                    img_parts = _frames_to_image_parts(files, max_images=max_images)
                    parts = img_parts + [{"text": _VIDEO_QA_PROMPT + "\nQuestion: " + question.strip()}]
                    data = _call_model(parts, temperature, model_name=model_name)
                    # выходим из with — файлы удалятся
                    answer = data["answer"] if isinstance(data, dict) and "answer" in data else None
                    if not isinstance(answer, str):
                        answer = str(answer) if answer is not None else "not enough information"
                    return _json.dumps({"answer": answer})

            # локальный файл (или если youtube уже скачали и вышли return выше)
            frame_dir = Path(tempfile.mkdtemp(prefix="frames_"))
            try:
                src_video = video_path if video_path else tmp_video_path
                files = _extract_frames_cv2(src_video, frame_dir, fps=fps, start_s=start_s, duration_s=duration_s)
                img_parts = _frames_to_image_parts(files, max_images=max_images)
                parts = img_parts + [{"text": _VIDEO_QA_PROMPT + "\nQuestion: " + question.strip()}]
                data = _call_model(parts, temperature, model_name=model_name)
            finally:
                # подчистим временные файлы
                for p in frame_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    frame_dir.rmdir()
                except Exception:
                    pass

        answer = data["answer"] if isinstance(data, dict) and "answer" in data else None
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else "not enough information"
        return _json.dumps({"answer": answer})

    except Exception as e:
        return _json.dumps({"error": str(e)})