from __future__ import annotations

import os
import json
import base64
import tldextract
import tempfile
from urllib.parse import urlparse
from langchain_tavily import TavilyExtract
from youtube_transcript_api import YouTubeTranscriptApi
import io
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
from PIL import Image, ImageStat, ExifTags
import google.generativeai as genai
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader
from PIL import ImageDraw, ImageFont, ImageEnhance, ImageFilter
from utils.image_processing import *
import re

def _exif_dict(img: Image.Image) -> dict:
    try:
        exif = img._getexif() or {}
        out = {}
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, str(k))
            out[tag] = v if isinstance(v, (int, float, str)) else str(v)
        return out
    except Exception:
        return {}

def _clip(text: str | None, n: int) -> str:
    """Утилита: безопасно обрезаем длинные сниппеты."""
    if not text:
        return ""
    text = text.strip()
    return (text[: n - 1] + "…") if len(text) > n else text



def _parse_dt(v) -> Optional[str]:
    """[ИЗМЕНЕНИЕ] Приводим даты к ISO-строке, если возможно."""
    try:
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str) and v:
            
            return v
    except Exception:
        pass
    return None

def _read_text_best_effort(path: str, max_chars: int) -> tuple[str, str]:
    # пробуем utf-8 → fallback latin-1 (без chardet)
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
            return s[:max_chars], "utf-8"
    except Exception:
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            s = f.read()
            return s[:max_chars], "latin-1"

# ИСПРАВЛЕНИЕ 3: Улучшить preprocess_files с более точным определением типов
def preprocess_files(files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Анализирует файлы и возвращает их метаданные"""
    file_info = {}
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found")
            continue
            
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        info = {
            "path": file_path,
            "extension": file_ext,
            "size": file_size,
            "type": None,
            "suggested_tool": None,  # ✅ ДОБАВЛЕНО: подсказка для reasoning
            "preview": None
        }
        
        # ✅ УЛУЧШЕНО: Более точное определение типов и инструментов
        if file_ext in ['.csv']:
            info["type"] = "table"
            info["suggested_tool"] = "analyze_csv_file"
        elif file_ext in ['.xlsx', '.xls']:
            info["type"] = "excel"
            info["suggested_tool"] = "analyze_excel_file"
        elif file_ext in ['.pdf']:
            info["type"] = "document"
            info["suggested_tool"] = "analyze_pdf_file"
        elif file_ext in ['.docx', '.doc']:
            info["type"] = "document"
            info["suggested_tool"] = "analyze_docx_file"
        elif file_ext in ['.txt', '.md']:
            info["type"] = "text"
            info["suggested_tool"] = "analyze_txt_file"
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            info["type"] = "image"
            info["suggested_tool"] = "vision_qa_gemma"
        else:
            info["type"] = "unknown"
            info["suggested_tool"] = "analyze_txt_file (fallback)"
            
        # Безопасное превью для небольших текстовых файлов
        if file_ext == '.txt' and file_size < 1000:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    info["preview"] = content[:200] + "..." if len(content) > 200 else content
            except Exception as e:
                info["preview"] = f"Error reading file: {e}"
        
        file_info[file_path] = info
    
    return file_info

#----------------------------------------------WEB BROWSING TOOLS------------------------------------------------#


#WIKIPEDIA SEARCH TOOL

@tool
def wiki_search(
    query: str,
    max_results: int = 3,
    language: str = "en",
    content_chars_max: int = 5000,
    snippet_chars: int = 400,
) -> str:
    """
    Search Wikipedia using LangChain's WikipediaLoader.
    Returns a JSON string:
    {
      "query": "...",
      "language": "en",
      "items": [
        {
          "url": "https://en.wikipedia.org/wiki/...",
          "title": "Title",
          "snippet": "First N chars of page content",
          "page_content": "...(clipped to content_chars_max)..."
        }
      ]
    }
    """
    try:
        docs = WikipediaLoader(
            query=query,
            load_max_docs=max_results,
            lang=language,
            doc_content_chars_max=content_chars_max,
        ).load()

        items: List[dict] = []
        seen_urls = set()

        for d in docs:
            url = d.metadata.get("source") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = d.metadata.get("title") or ""
            page_content = d.page_content or ""
            snippet = _clip(page_content, snippet_chars)

            items.append(
                {
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "page_content": page_content,  # уже ограничен doc_content_chars_max
                }
            )

        payload = {
            "query": query,
            "language": language,
            "items": items,
        }
        return json.dumps(payload)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query, "language": language})

#TAVILY WEB SEARCH TOOL

def _domain(url: str) -> str:
    """Утилита: вытаскиваем домен в виде 'site.tld' (без поддоменов)."""
    ext = tldextract.extract(url)
    return ".".join([p for p in (ext.domain, ext.suffix) if p])

@tool
def web_search(
    query: str,
    max_results: int = 5,                          # [ИЗМЕНЕНИЕ] параметризуем число результатов (было зашито 3)
    unique_domains: int = 5,                        # [ИЗМЕНЕНИЕ] хотим максимум N разных доменов (борьба с дубликатами)
    snippet_chars: int = 400,                       # [ИЗМЕНЕНИЕ] ограничиваем длину сниппета
    include_domains: Optional[List[str]] = None,    # [ИЗМЕНЕНИЕ] вайтлист доменов
    exclude_domains: Optional[List[str]] = None,    # [ИЗМЕНЕНИЕ] блэклист доменов
) -> str:
    """
    Structured web search via Tavily.

    Возвращает JSON-строку такого вида:
    {
      "query": "...",
      "provider": "tavily",
      "items": [
        {
          "url": "...",
          "title": "...",
          "snippet": "...",
          "published": "2024-05-01T10:00:00Z",   # если Tavily отдал
          "source": "example.com"                # домен
        }
      ]
    }
    """
    # [ИЗМЕНЕНИЕ] раньше возвращалась сыровая строка с разметкой; теперь — строгий JSON для удобства парсинга
    try:
        # [ИЗМЕНЕНИЕ] используем официальный LangChain-тул, но берём больше (max_results), чтобы потом отфильтровать домены
        raw_results = TavilySearchResults(max_results=max_results).invoke(query)

        items: List[dict] = []
        seen_urls: set[str] = set()
        seen_domains: set[str] = set()

        inc = set(include_domains or [])          # [ИЗМЕНЕНИЕ] поддержка фильтров доменов (whitelist)
        exc = set(exclude_domains or [])          # [ИЗМЕНЕНИЕ] поддержка фильтров доменов (blacklist)

        for r in raw_results:
            url = (r.get("url") or "").strip()
            if not url:
                continue

            dom = _domain(url)

            # [ИЗМЕНЕНИЕ] применяем include/exclude-фильтры доменов
            if inc and dom not in inc:
                continue
            if dom in exc:
                continue

            # [ИЗМЕНЕНИЕ] дедупликация ссылок
            if url in seen_urls:
                continue

            # [ИЗМЕНЕНИЕ] ограничиваем разнообразие доменов (часто Tavily даёт много результатов с одного сайта)
            if unique_domains > 0 and dom in seen_domains:
                # если домен уже встречался и лимит по доменам строгий, пропускаем
                pass
            else:
                # засчитываем домен как использованный
                seen_domains.add(dom)

            title = (r.get("title") or "").strip()
            content = r.get("content") or r.get("snippet") or ""
            snippet = _clip(content, snippet_chars)  # [ИЗМЕНЕНИЕ] делаем аккуратный сниппет
            published = r.get("published_date") or r.get("created_at")  # [ИЗМЕНЕНИЕ] пытаемся достать дату

            items.append(
                {
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "published": published,
                    "source": dom,  # [ИЗМЕНЕНИЕ] явный домен — удобно для форматтера/критика
                }
            )
            seen_urls.add(url)

            # [ИЗМЕНЕНИЕ] если мы уже собрали нужное число результатов после фильтрации — выходим
            if len(items) >= max_results:
                break

        payload = {
            "query": query,
            "provider": "tavily",
            "items": items,
        }
        return json.dumps(payload)

    except Exception as e:
        # [ИЗМЕНЕНИЕ] единый формат ошибок в JSON — проще логировать и обрабатывать в агенте
        return json.dumps({"error": str(e), "query": query, "provider": "tavily"})


#ARXIV SEARCH TOOL

@tool
def arxiv_search(
    query: str,
    max_results: int = 5,                
) -> str:
    """
    Поиск по arXiv через LangChain ArxivLoader.

    [ИЗМЕНЕНИЕ] Возвращает **строгий JSON** вида:
    {
      "query": "...",
      "provider": "arxiv",
      "items": [
        {
          "title": "...",
          "authors": ["A. Author","B. Author"],
          "published": "YYYY-MM-DDTHH:MM:SS",
          "journal_ref": "…",                 # если есть
          "comment": "…",                     # если есть
          "snippet": "first N chars of summary",
          "summary": "… (может быть клипнут ArxivLoader'ом по умолчанию)"
        }
      ]
    }
    """
    try:
        docs = ArxivLoader(
            query=query,
            load_max_docs=max_results,
        ).load()

        items: List[dict] = []

        for d in docs:
            md = d.metadata or {}

            title = md.get("Title") or md.get("title") or ""
            authors = md.get("Authors") or md.get("authors") or []
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",") if a.strip()]

            published = _parse_dt(md.get("Published") or md.get("published"))
            summary = d.page_content or ""

            items.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "summary": summary,
                }
            )

            if len(items) >= max_results:
                break

        payload = {
            "query": query,
            "provider": "arxiv",
            "items": items,
        }
        return json.dumps(payload)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query, "provider": "arxiv"})
    

@tool
def web_extract(urls : List[str]) -> str:
    """
    Extract text content from web pages using TavilyExtract.
    Returns JSON with {url, title, text, images?} for each URL.
    """

    tool = TavilyExtract(
    extract_depth="basic",
    include_images=False,
)
    results = tool.invoke(urls)
    return json.dumps(results)

@tool
def extract_youtube_transcript(url: str, chars: int = 10_00) -> str:
    """
    Fetch full YouTube transcript (first *chars* characters).
    """

    video_id_match = re.search(r"[?&]v=([A-Za-z0-9_\-]{11})", url)
    if not video_id_match:
        return "yt_error:id_not_found"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
        text = " ".join(piece["text"] for piece in transcript)
        return text[:chars]
    except Exception as exc:
        return f"yt_error:{exc}"

#----------------------------------------------MATH TOOLS------------------------------------------------#
    
@tool
def add(a: float, b: float) -> float:
    """Returns the sum of two numbers.
        Example: add(2, 3) -> 5
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Returns the difference of two numbers.
        Example: subtract(5, 3) -> 2
    """
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Returns the product of two numbers.
        Example: multiply(2, 3) -> 6
    """
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Returns the quotient of two numbers.
        Example: divide(6, 3) -> 2
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def power(a: float, b: float) -> float:
    """Returns a raised to the power of b.
        Example: power(2, 3) -> 8
    """
    return a ** b


#----------------------------------------------FILE PROCESSING TOOLS------------------------------------------------#

@tool
def analyze_csv_file(file_path: str, preview_rows: int = 20) -> str:
    """
    Analyze a CSV file: returns JSON with {kind, path, shape, columns, head, numeric_summary}.
    - preview_rows: number of rows for preview (head)
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": "file not found", "path": file_path})
    try:
        df = pd.read_csv(file_path)
        head = df.head(preview_rows).to_dict(orient="records")
        numeric = df.select_dtypes("number").describe().to_dict()
        payload = {
            "kind": "csv",
            "path": file_path,
            "shape": list(df.shape),
            "columns": list(map(str, df.columns)),
            "head": head,
            "numeric_summary": numeric,  # {col: {count, mean, std, ...}}
        }
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": str(e), "path": file_path})
    
@tool
def analyze_excel_file(file_path: str, sheet: int | str | None = None, preview_rows: int = 20, list_sheets: bool = True) -> str:
    """
    Analyze an Excel file: {kind, path, sheets?, active_sheet, shape, columns, head}.
    - sheet: sheet index or name (None -> first sheet)
    - list_sheets: include all sheet names
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": "file not found", "path": file_path})
    try:
        xls = pd.ExcelFile(file_path)
        target = sheet if sheet is not None else 0
        df = pd.read_excel(xls, sheet_name=target)
        head = df.head(preview_rows).to_dict(orient="records")
        payload = {
            "kind": "excel",
            "path": file_path,
            "active_sheet": target if isinstance(target, int) else str(target),
            "shape": list(df.shape),
            "columns": list(map(str, df.columns)),
            "head": head,
        }
        if list_sheets:
            payload["sheets"] = list(map(str, xls.sheet_names))
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": str(e), "path": file_path})
    
@tool
def analyze_docx_file(file_path: str, max_chars: int = 20000, join_with: str = "\n") -> str:
    """
    Extract text from DOCX: {kind, path, paragraphs, text[:max_chars]}.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": "file not found", "path": file_path})
    try:
        from docx import Document  # pip install python-docx
    except Exception as e:
        return json.dumps({"error": f"python-docx not installed: {e}"})
    try:
        doc = Document(file_path)
        paras = [p.text for p in doc.paragraphs if p.text is not None]
        text = join_with.join(paras)
        payload = {
            "kind": "docx",
            "path": file_path,
            "paragraphs": len(paras),
            "text": text[:max_chars],
            "length": len(text),
        }
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": str(e), "path": file_path})
    

@tool
def analyze_txt_file(file_path: str, max_chars: int = 20000) -> str:
    """
    Read plain text: {kind, path, encoding_guess, text[:max_chars], length}.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": "file not found", "path": file_path})
    try:
        text, enc = _read_text_best_effort(file_path, max_chars=max_chars)
        payload = {
            "kind": "txt",
            "path": file_path,
            "encoding_guess": enc,
            "text": text,
            "length": os.path.getsize(file_path),
        }
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": str(e), "path": file_path})
    
@tool
def analyze_pdf_file(file_path: str, max_chars: int = 20000) -> str:
    """
    Extract text & page count from PDF: {kind, path, pages, text[:max_chars]}.
    Uses pdfminer.six for text and page counting.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": "file not found", "path": file_path})
    try:
        # text
        from pdfminer.high_level import extract_text
        text = extract_text(file_path) or ""
        # pages
        from pdfminer.pdfpage import PDFPage
        with open(file_path, "rb") as f:
            pages = sum(1 for _ in PDFPage.get_pages(f))
        payload = {
            "kind": "pdf",
            "path": file_path,
            "pages": pages,
            "text": text[:max_chars],
            "length": len(text),
        }
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": str(e), "path": file_path})
    

#----------------------------------------------IMAGE PROCESSING TOOLS------------------------------------------------#

@tool
def analyze_image_file(file_path: str, ocr: bool = False, lang: Optional[str] = None, max_ocr_chars: int = 10000) -> str:
    """
    Analyze image: {kind, path, format, mode, size, mean_brightness, exif?, ocr_text?}.
    - ocr: optional Tesseract OCR (pip install pytesseract + tesseract)
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": "file not found", "path": file_path})
    try:
        img = Image.open(file_path)
        stat = ImageStat.Stat(img.convert("L"))
        mean_brightness = float(stat.mean[0])  # 0..255
        payload = {
            "kind": "image",
            "path": file_path,
            "format": img.format,
            "mode": img.mode,
            "size": list(img.size),  # [width, height]
            "mean_brightness": mean_brightness,
        }
        exif = _exif_dict(img)
        if exif:
            payload["exif"] = exif

        if ocr:
            try:
                import pytesseract
                conf = {}
                if lang:
                    conf["lang"] = lang
                text = pytesseract.image_to_string(img, **conf) or ""
                payload["ocr_text"] = text[:max_ocr_chars]
                payload["ocr_length"] = len(text)
            except Exception as e:
                payload["ocr_error"] = str(e)

        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": str(e), "path": file_path})
    



# ------------------------- helpers for QA image TOOL -------------------------

def _configure():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GENAI_API_KEY) in environment")
    genai.configure(api_key=api_key)

def _image_bytes_to_part(img_bytes: bytes, mime: str = "image/png") -> Dict[str, Any]:
    # формат, который понимает genai.generate_content
    return {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}

def _ensure_png_bytes(img: Image.Image, max_pixels: int = 25_000_000) -> bytes:
    # мягко даунскейлим огромные изображения (защита от "image bomb")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)

    # приводим к PNG (надёжно для SDK)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _load_image_as_png_bytes_from_path(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    return _ensure_png_bytes(img)

def _load_image_as_png_bytes_from_b64(b64: str) -> bytes:
    raw = base64.b64decode(b64, validate=True)
    img = Image.open(io.BytesIO(raw))
    return _ensure_png_bytes(img)

def _clean_json_text(s: str) -> str:
    # вычищаем обёртки ```json ... ``` и забираем объект { ... }
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").replace("json", "", 1).strip()
    # вырезать по внешним фигурным скобкам
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s

_SINGLE_IMAGE_QA_PROMPT = (
    "You will be given ONE image and a user question about it.\n"
    "Answer STRICTLY and CONCISELY based only on the image content.\n"
    "If the image does not contain enough information to answer, reply 'not enough information'.\n"
    "If the answer is numeric, include units if visible.\n"
    "Return ONLY valid JSON with the schema:\n"
    "{\"answer\": string}\n"
)

def _call_model(parts: List[Any], temperature: float) -> Dict[str, Any]:
    MODEL_NAME = "gemma-3-27b-it"
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(parts, generation_config={"temperature": temperature})
    text = (resp.text or "").strip()

    # пробуем сразу распарсить
    try:
        return json.loads(_clean_json_text(text))
    except Exception:
        # второй шанс: попросим модель вернуть строгий JSON
        fixer = genai.GenerativeModel(MODEL_NAME)
        fix_prompt = (
            "Convert the following text into STRICT valid JSON matching schema {\"answer\": string}. "
            "Return ONLY JSON, no extra text:\n" + text
        )
        fix_resp = fixer.generate_content([{"text": fix_prompt}])
        return json.loads(_clean_json_text((fix_resp.text or "").strip()))

# --------------------------- TOOL ---------------------------

@tool
def vision_qa_gemma(
    question: str,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    Vision QA with Google GenAI (Gemma/Gemini). Returns JSON: {"answer": "..."}.

    Args:
      question: user question about the image.
      image_path: local file path to the image (PNG/JPG/...).
      image_base64: base64-encoded image (if no path).
      temperature: decoding temperature (default 0.2).

    Exactly ONE of (image_path, image_base64) must be provided.
    """
    import json as _json
    try:
        _configure()
        if bool(image_path) == bool(image_base64):
            return _json.dumps({"error": "Provide exactly ONE of image_path or image_base64"})

        if image_path:
            img_bytes = _load_image_as_png_bytes_from_path(image_path)
        else:
            img_bytes = _load_image_as_png_bytes_from_b64(image_base64)

        parts = [
            {"text": _SINGLE_IMAGE_QA_PROMPT + "\nQuestion: " + question.strip()},
            _image_bytes_to_part(img_bytes, "image/png"),
        ]

        data = _call_model(parts, temperature)
        # финальная защита: оставляем только "answer"
        answer = data["answer"] if isinstance(data, dict) and "answer" in data else None
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else "not enough information"

        return _json.dumps({
            "answer": answer,
        })

    except Exception as e:
        return _json.dumps({"error": str(e)})


#-------------------------------------------------------------- ADDITIONAL TOOLS -------------------------------------------------------------#
@tool
def draw_on_image(
    image_base64: str, drawing_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Draw shapes (rectangle, circle, line) or text onto an image.
    Args:
        image_base64 (str): Base64 encoded input image
        drawing_type (str): Drawing type
        params (Dict[str, Any]): Drawing parameters
    Returns:
        Dictionary with result image (base64)
    """
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)
        color = params.get("color", "red")

        if drawing_type == "rectangle":
            draw.rectangle(
                [params["left"], params["top"], params["right"], params["bottom"]],
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "circle":
            x, y, r = params["x"], params["y"], params["radius"]
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "line":
            draw.line(
                (
                    params["start_x"],
                    params["start_y"],
                    params["end_x"],
                    params["end_y"],
                ),
                fill=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "text":
            font_size = params.get("font_size", 20)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            draw.text(
                (params["x"], params["y"]),
                params.get("text", "Text"),
                fill=color,
                font=font,
            )
        else:
            return {"error": f"Unknown drawing type: {drawing_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"result_image": result_base64}

    except Exception as e:
        return {"error": str(e)}
    
@tool
def transform_image(
    image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply transformations: resize, rotate, crop, flip, brightness, contrast, blur, sharpen, grayscale.
    Args:
        image_base64 (str): Base64 encoded input image
        operation (str): Transformation operation
        params (Dict[str, Any], optional): Parameters for the operation
    Returns:
        Dictionary with transformed image (base64)
    """
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            img = img.resize(
                (
                    params.get("width", img.width // 2),
                    params.get("height", img.height // 2),
                )
            )
        elif operation == "rotate":
            img = img.rotate(params.get("angle", 90), expand=True)
        elif operation == "crop":
            img = img.crop(
                (
                    params.get("left", 0),
                    params.get("top", 0),
                    params.get("right", img.width),
                    params.get("bottom", img.height),
                )
            )
        elif operation == "flip":
            if params.get("direction", "horizontal") == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif operation == "adjust_brightness":
            img = ImageEnhance.Brightness(img).enhance(params.get("factor", 1.5))
        elif operation == "adjust_contrast":
            img = ImageEnhance.Contrast(img).enhance(params.get("factor", 1.5))
        elif operation == "blur":
            img = img.filter(ImageFilter.GaussianBlur(params.get("radius", 2)))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
        elif operation == "grayscale":
            img = img.convert("L")
        else:
            return {"error": f"Unknown operation: {operation}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"transformed_image": result_base64}

    except Exception as e:
        return {"error": str(e)}

@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."


import requests

@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"
