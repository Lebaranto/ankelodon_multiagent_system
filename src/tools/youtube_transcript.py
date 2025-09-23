from langchain.tools import tool

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

import re
from urllib.parse import urlparse, parse_qs


def _extract_video_id(url_or_id: str) -> str | None:
    s = (url_or_id or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    u = urlparse(s)
    # youtu.be/<id>
    if u.netloc.endswith("youtu.be"):
        vid = u.path.strip("/").split("/")[0]
        return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else None
    # watch?v=<id>
    qs = parse_qs(u.query or "")
    if "v" in qs:
        vid = qs["v"][0]
        return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else None
    # /embed/<id>, /shorts/<id>, /v/<id>
    for pref in ("/embed/", "/shorts/", "/v/"):
        if u.path.startswith(pref):
            vid = u.path[len(pref):].split("/")[0]
            return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else None
    return None


@tool
def extract_youtube_transcript(url: str, chars: int = 1000) -> str:
    """
    Simple YouTube transcript fetcher.

    Input:
      - url: Regular YouTube URL (or the 11-char video_id).
      - chars: Return the first `chars` characters of the transcript.

    Output:
      - String with the transcript (trimmed to `chars`), or an error string:
        "yt_error:<reason>"
    """
    if YouTubeTranscriptApi is None:
        return "yt_error:missing_dependency"

    vid = _extract_video_id(url)
    if not vid:
        return "yt_error:id_not_found"

    try:
        api = YouTubeTranscriptApi()
        # New API returns a list of FetchedTranscriptSnippet objects
        snippets = api.fetch(vid)

        parts = []
        for s in snippets:
            # Support both object (new) and dict (old) shapes
            text = getattr(s, "text", None)
            if text is None and isinstance(s, dict):
                text = s.get("text")
            if not text:
                continue
            parts.append(text.replace("\n", " ").strip())

        full_text = " ".join(p for p in parts if p)
        return full_text[: max(0, int(chars))]
    except Exception as e:
        return f"yt_error:{type(e).__name__}:{e}"