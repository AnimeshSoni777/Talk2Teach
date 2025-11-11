import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import subprocess
import tempfile
import json
import os
import re
import textwrap
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from mutagen.mp3 import MP3
from mutagen import MutagenError
import requests
from io import BytesIO
import base64

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


# =========================
# Page & Global Settings
# =========================
st.set_page_config(
    page_title="AI Hinglish Video Lecture Generator",
    page_icon="🎬",
    layout="wide"
)

# -------- Fonts & Palettes --------
DEFAULT_FONTS = ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]  # cross-OS friendly

def _load_font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()

COLOR_PALETTES = [
    {'name': 'A. Academic & Warm', 'bg': '#FAF3E0', 'text': '#4E342E', 'title': '#800000'},
    {'name': 'B. Tech Dark Mode',  'bg': '#121212', 'text': '#E0E0E0', 'title': '#FFD700'},
    {'name': 'C. Clean & Minimal', 'bg': '#FFFFFF', 'text': '#212121', 'title': '#008060'},
    {'name': 'D. Cool Slate',      'bg': '#EDF2F7', 'text': '#2D3748', 'title': '#2B6CB0'},
    {'name': 'E. Forest & Amber',  'bg': '#004D40', 'text': '#F1F1F1', 'title': '#F2C94C'},
    {'name': 'F. Earthy & Natural','bg': '#F4F1DE', 'text': '#5D4037', 'title': '#81A684'},
    {'name': 'G. Indigo & Cyan',   'bg': '#002244', 'text': '#FFFFFF', 'title': '#66B2FF'},
    {'name': 'H. Slate & Orange',  'bg': '#34495E', 'text': '#ECF0F1', 'title': '#F39C12'},
    {'name': 'I. Soft & Purple',   'bg': '#F8F9FA', 'text': '#343A40', 'title': '#6F42C1'},
    {'name': 'J. Rich Plum',       'bg': '#241B2F', 'text': '#D1C4E9', 'title': '#90CAF9'},
]
DEFAULT_PALETTE_INDEX = 0  # default theme

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except Exception:
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

# =========================
# Session State
# =========================
ss = st.session_state
ss.setdefault('slides', [])
ss.setdefault('audio', [])
ss.setdefault('video_path', None)
ss.setdefault('gemini_model_name', 'gemini-2.5-flash')  # <-- set default model
ss.setdefault('palette_index', DEFAULT_PALETTE_INDEX)


# =========================
# Vector DB + Embeddings (for Chatbot)
# =========================
if 'chroma_client' not in ss:
    ss.chroma_client = chromadb.Client()

if 'collection' not in ss:
    ss.collection = None  # will be created after PDF upload

# Load embedding model once
if 'embed_model' not in ss:
    ss.embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# =========================
# Utilities
# =========================
def cleanup(paths: List[str]):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""
    
def store_pdf_text_in_db(pdf_text: str, pdf_name: str):
    from textwrap import wrap
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', pdf_name.lower())
    safe_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', safe_name)
    if len(safe_name) < 3:
        safe_name = "defaultpdf"
    chunks = wrap(pdf_text, width=800)
    ss.collection = ss.chroma_client.get_or_create_collection(name=f"pdf_{safe_name}")
    try:
        existing = ss.collection.get()
        if existing and existing.get('ids'):
            ss.collection.delete(ids=existing['ids'])
    except Exception:
        pass
    for i, chunk in enumerate(chunks):
        emb = ss.embed_model.encode(chunk).tolist()
        ss.collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[f"{safe_name}_{i}"]
        )
    st.success(f"✅ Stored {len(chunks)} chunks in vector DB for chatbot context.")


def answer_question_with_context(question: str, gemini_key: str) -> str:
    if ss.collection is None:
        return "⚠️ No document context found. Please upload a PDF first."
    query_emb = ss.embed_model.encode(question).tolist()
    results = ss.collection.query(query_embeddings=[query_emb], n_results=3)
    context = " ".join(results['documents'][0]) if results['documents'] else ""
    if not context.strip():
        return "I couldn’t find relevant info in the PDF."
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(ss.gemini_model_name or "gemini-2.0-flash")  # ✅ FIXED LINE
    prompt = f"""
You are a helpful AI tutor.
Use the following PDF context to answer the student's question accurately and concisely.

Context:
{context}

Question:
{question}

Answer in Hinglish (Roman Script, friendly teacher tone).
"""

    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Error while fetching answer: {e}"


def safe_json_from_llm(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM output.
    Strips fences, fixes trailing commas, quotes, and attempts relaxed parsing.
    """
    import re, json

    candidate = text.strip()
    # Extract JSON block
    fence = re.search(r"```json\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()

    # Clean up common issues
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)  # trailing commas
    candidate = re.sub(r"(\w)" r"(\s*[\]}])", r"\1\2", candidate)  # spacing issues

    # If model wrapped with commentary, carve out first JSON block
    if not candidate.strip().startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            candidate = candidate[start:end + 1]

    # --- attempt standard load first ---
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try fixing common missing commas (e.g., between "}" and "{")
        candidate = re.sub(r'"\s*"', '", "', candidate)
        candidate = re.sub(r'}\s*{', '}, {', candidate)
        candidate = re.sub(r'\]\s*\[', '], [', candidate)
        try:
            return json.loads(candidate)
        except Exception:
            # Last resort — return empty
            st.error("⚠️ JSON parse failed — Gemini output malformed. Try regenerating.")
            st.text_area("Raw Gemini Output", text, height=200)
            return {"slides": []}

def draw_multiline(draw, text, xy, font, fill, max_width_chars=60, line_spacing=6):
    y = xy[1]
    for line in textwrap.wrap(text or "", width=max_width_chars):
        draw.text((xy[0], y), line, font=font, fill=fill)
        bbox = font.getbbox(line)
        y += (bbox[3] - bbox[1]) + line_spacing

# ---------- Wikimedia image fetch ----------
def wikimedia_thumb_from_query(query: str, width: int = 800) -> Optional[bytes]:
    """
    Try to fetch a relevant image thumbnail bytes for a topic using Wikimedia / Wikipedia APIs.
    Returns image bytes or None.
    """
    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1
            },
            timeout=12
        ).json()
        results = s.get("query", {}).get("search", [])
        if not results:
            return None
        page_title = results[0]["title"]

        imginfo = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": str(width),
                "titles": page_title,
                "format": "json"
            },
            timeout=12
        ).json()
        pages = imginfo.get("query", {}).get("pages", {})
        for _, v in pages.items():
            thumb = v.get("thumbnail", {})
            src = thumb.get("source")
            if src:
                resp = requests.get(src, timeout=12)
                if resp.status_code == 200 and resp.content:
                    return resp.content
        return None
    except Exception:
        return None

# ---------- Gemini image generation (fallback) ----------
def gemini_generate_image_bytes(api_key: str, prompt: str, size: str = "1024x768") -> Optional[bytes]:
    """
    Try to generate an image via Gemini/Imagen using google-generativeai SDK.
    Handles multiple SDK shapes to be robust across versions.
    Returns raw image bytes or None.
    """
    try:
        genai.configure(api_key=api_key)

        # Path A: Dedicated Images API (Imagen 3) if available in the SDK
        try:
            from google.generativeai import images as gai_images  # type: ignore
            # Common model name aliases; adjust as available in your project
            model_name = "imagen-3.0-generate-001"
            img_resp = gai_images.generate(
                model=model_name,
                prompt=prompt,
                size=size,
                number_of_images=1,
                safety_filter_level="block_only_high"  # keep it safe-ish
            )
            # Different SDKs return differently; try common fields
            if hasattr(img_resp, "generated_images") and img_resp.generated_images:
                # Newer SDK: generated_images[0].image.data (bytes) or .image.base64_data
                gi = img_resp.generated_images[0]
                if hasattr(gi, "image") and gi.image is not None:
                    if hasattr(gi.image, "data") and gi.image.data:
                        return gi.image.data
                    if hasattr(gi.image, "base64_data") and gi.image.base64_data:
                        return base64.b64decode(gi.image.base64_data)
            # Some SDKs return a list of dicts or parts with base64
            if isinstance(img_resp, dict):
                # Try known paths
                b64 = (
                    img_resp.get("images", [{}])[0].get("base64_data")
                    or img_resp.get("generated_images", [{}])[0].get("base64_data")
                )
                if b64:
                    return base64.b64decode(b64)
        except Exception:
            pass

        # Path B: Use a text model that can return inline image parts (fallback)
        try:
            # Try a capable model (adjust if you prefer a different one)
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            # Ask explicitly for an image; many SDKs will return inline_data image parts
            resp = model.generate_content(
                f"Create a clean, educational illustration for: {prompt}. "
                "Return one PNG image only.",
                request_options={"mime_type": "image/png"}
            )
            # Try to extract inline bytes from the response
            # Different SDKs expose parts differently:
            candidates = getattr(resp, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content: continue
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    # Inline image data
                    if hasattr(part, "inline_data") and part.inline_data:
                        data = part.inline_data.get("data") or part.inline_data.get("bytes")
                        if data:
                            # data may already be bytes or base64
                            return data if isinstance(data, (bytes, bytearray)) else base64.b64decode(data)
                    # Some SDKs use 'blob' or 'data' attributes
                    if hasattr(part, "data"):
                        dd = part.data
                        if isinstance(dd, (bytes, bytearray)):
                            return dd
                        try:
                            return base64.b64decode(dd)
                        except Exception:
                            pass
        except Exception:
            pass

        return None
    except Exception:
        return None

def fetch_topic_image_bytes(prompt: str, gemini_key: Optional[str], width: int = 800) -> Optional[bytes]:
    """
    1) Try Wikimedia
    2) If None and gemini_key present: try Gemini image generation
    3) If still None: return None (no image shown)
    """
    # try Wikipedia/Wikimedia first
    img = wikimedia_thumb_from_query(prompt, width=width)
    if img:
        return img
    # fallback to Gemini if key present
    if gemini_key:
        # Convert width to string like "1024x768" (preserve 16:9-ish)
        # We keep a standard size; it will be resized on paste anyway
        side = max(512, min(1536, width))
        size_str = f"{side}x{int(side * 0.75)}"
        return gemini_generate_image_bytes(gemini_key, prompt, size=size_str)
    return None

# ---------- Audio (gTTS + speed) ----------
def speed_change_ffmpeg(in_path: str, out_path: str, atempo: float) -> bool:
    """
    Change audio speed with ffmpeg atempo (0.5–2.0).
    Returns True if succeeded.
    """
    if not FFMPEG_AVAILABLE or abs(atempo - 1.0) < 1e-3:
        # Either ffmpeg missing or no speed change requested
        if in_path != out_path:
            try:
                with open(in_path, "rb") as fi, open(out_path, "wb") as fo:
                    fo.write(fi.read())
            except Exception:
                return False
        return True

    # atempo supports 0.5 to 2.0 per filter.
    atempo = max(0.5, min(2.0, atempo))
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-filter:a", f"atempo={atempo}",
        "-c:a", "mp3",
        out_path
    ]
    try:
        return subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    except Exception:
        return False

def gtts_audio(script: str, idx: int, atempo: float):
    """
    Generate Hinglish audio with gTTS, then optionally speed it up/down via ffmpeg.
    Returns (final_audio_path, duration_seconds)
    """
    base_tmp = os.path.join(tempfile.gettempdir(), f"slide_{idx}_base.mp3")
    final_tmp = os.path.join(tempfile.gettempdir(), f"slide_{idx}_final.mp3")

    try:
        tts = gTTS(text=script, lang='hi', slow=False)
        tts.save(base_tmp)
    except Exception:
        try:
            tts = gTTS(text=script, lang='en', slow=False)
            tts.save(base_tmp)
        except Exception:
            return None, 0.0

    if not speed_change_ffmpeg(base_tmp, final_tmp, atempo):
        final_tmp = base_tmp  # fallback to original

    # Duration
    try:
        duration = MP3(final_tmp).info.length
    except MutagenError:
        duration = max(2.0, len(script) * 0.09)

    # Clean base if different
    if final_tmp != base_tmp:
        try:
            os.remove(base_tmp)
        except Exception:
            pass

    return final_tmp, float(duration)

# === Replace ONLY the image logic with this ===

# ---------- Wikimedia image fetch (robust) ----------
WIKI_HEADERS = {
    "User-Agent": "HinglishLectureGenerator/1.0 (contact: your_email_or_site)"
}

@st.cache_data(show_spinner=False, ttl=3600)
def wiki_thumb_by_title(title: str, width: int = 1000) -> Optional[bytes]:
    if not title:
        return None
    try:
        # Try REST summary first (better images)
        rest = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
            headers=WIKI_HEADERS, timeout=12
        )
        if rest.ok:
            js = rest.json()
            src = (js.get("originalimage") or {}).get("source") or (js.get("thumbnail") or {}).get("source")
            if src:
                r = requests.get(src, headers=WIKI_HEADERS, timeout=12)
                if r.ok and r.content:
                    return r.content

        # Fallback: classic pageimages
        imginfo = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": str(width),
                "titles": title,
                "format": "json"
            },
            headers=WIKI_HEADERS, timeout=12
        ).json()
        pages = imginfo.get("query", {}).get("pages", {})
        for _, v in pages.items():
            thumb = v.get("thumbnail", {})
            src = thumb.get("source")
            if src:
                r = requests.get(src, headers=WIKI_HEADERS, timeout=12)
                if r.ok and r.content:
                    return r.content
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def wiki_thumb_from_query(query: str, width: int = 1000) -> Optional[bytes]:
    if not query:
        return None
    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1
            },
            headers=WIKI_HEADERS, timeout=12
        )
        if not s.ok:
            return None
        results = s.json().get("query", {}).get("search", [])
        if not results:
            return None
        page_title = results[0]["title"]
        return wiki_thumb_by_title(page_title, width=width)
    except Exception:
        return None


def best_image_bytes_for_slide(slide: Dict[str, Any]) -> Optional[bytes]:
    """Try image_prompt → title → first bullet."""
    for candidate in [
        slide.get("image_prompt", "").strip(),
        slide.get("title", "").strip(),
        (slide.get("bullet_points", []) or [""])[0].strip()
    ]:
        if not candidate:
            continue
        b = wiki_thumb_from_query(candidate)
        if b:
            return b
    return None


# ---------- Slide image compose ----------
def generate_slide_image(slide: Dict[str, Any], palette: Dict[str, str], gemini_key: Optional[str], size=(1920,1080), for_preview: bool = False):
    """
    Preview: no script box; no prompt text; only title/bullets and image if available.
    Final: same visual rules (no prompt text). If no image -> no right panel drawn.
    """
    img = Image.new('RGB', size, color=palette['bg'])
    d = ImageDraw.Draw(img)

    title_font = _load_font(DEFAULT_FONTS, 64)
    text_font = _load_font(DEFAULT_FONTS, 40)
    script_font = _load_font(DEFAULT_FONTS, 30)
    small_font = _load_font(DEFAULT_FONTS, 24)

    W, H = size
    margin = 80
    col_left = margin
    col_right = int(W*0.68)
    right_w = W - col_right - margin
    right_h = int(H*0.62)
    right_xy = (col_right, 150)

    # Title
    d.text((col_left, 60), slide.get('title',"Untitled"), fill=palette['title'], font=title_font)

    # Bullets
    y = 180
    bullets = slide.get('bullet_points', [])
    for bp in bullets:
        lines = textwrap.wrap(bp, width=44)
        d.text((col_left, y), "•", font=text_font, fill=palette['text'])
        y_line = y
        for ln in lines:
            d.text((col_left+35, y_line), ln, font=text_font, fill=palette['text'])
            hb = text_font.getbbox(ln); y_line += (hb[3]-hb[1]) + 4
        y = y_line + 10
        if y > H - 300:
            break

    # Hinglish script (hide in preview; show in final)
    if not for_preview:
        script_hdr_y = H - 230
        d.text((col_left, script_hdr_y), "Hinglish Script:", font=script_font, fill=palette['title'])
        draw_multiline(d, slide.get('hinglish_script',''), (col_left, script_hdr_y+36), script_font, palette['text'], max_width_chars=90, line_spacing=4)

    # Right panel image: fetch (Wiki -> Gemini fallback)
    thumb_bytes = best_image_bytes_for_slide(slide)

    if thumb_bytes:
        try:
            im = Image.open(BytesIO(thumb_bytes)).convert("RGB")
            im = im.resize((right_w, right_h))
            img.paste(im, right_xy)
            # small caption strip (optional; can be removed if you want absolute clean)
            cap_h = 36
            overlay = Image.new("RGBA", (right_w, cap_h), (0,0,0,120))
            img.paste(overlay, (right_xy[0], right_xy[1] + right_h - cap_h), overlay)
            d2 = ImageDraw.Draw(img)
            d2.text((right_xy[0]+10, right_xy[1]+right_h-cap_h+8),
                    "Auto-generated visual", font=small_font, fill="#FFFFFF")
        except Exception:
            # If paste fails, we just skip showing any image (no placeholders)
            pass
    # If no image -> do nothing (no box, no placeholder, no prompt)

    return img

# ---------- FFmpeg helpers ----------
def ffmpeg_concat_image_audio(img_path: str, audio_path: str, out_path: str):
    cmd = [
        'ffmpeg','-y',
        '-loop','1','-i',img_path,
        '-i',audio_path,
        '-c:v','libx264','-c:a','aac',
        '-pix_fmt','yuv420p',
        '-shortest',
        out_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0

def ffmpeg_join_segments(paths: List[str], out_path: str):
    concat_file = os.path.join(tempfile.gettempdir(), "concat_list.txt")
    with open(concat_file, 'w') as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    cmd = ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_file,'-c','copy',out_path]
    ok = subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    try: os.remove(concat_file)
    except Exception: pass
    return ok

# =========================
# LLM Slide Generation
# =========================
SYSTEM_PROMPT = (
    "You are a precise lecture slide composer.\n"
    "Return STRICT JSON ONLY. No commentary. No markdown fences unless asked.\n"
    "Hinglish should be clear (Roman script), concise (50-80 words per slide)."
)

USER_TEMPLATE = """
Convert the following text into a structured video lecture outline in JSON.

Constraints:
- 6–10 slides.
- Each slide fields:
  - title: concise English title
  - bullet_points: 3–5 crisp points (English)
  - hinglish_script: 50–80 words, Roman-script Hinglish, teacher-like friendly tone
  - image_prompt: technical, specific visual description matching content

Text (trimmed):
{TEXT}
Return:
{
  "slides": [
    {
      "title": "...",
      "bullet_points": ["...","...","..."],
      "hinglish_script": "...",
      "image_prompt": "..."
    }
  ]
}
"""

def gemini_slides(api_key: str, model_name: str, text: str) -> List[Dict[str, Any]]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name or "gemini-2.0-flash")  # ✅ FIXED LINE
    prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.replace("{TEXT}", text[:4000])
    resp = model.generate_content(prompt)
    data = safe_json_from_llm(resp.text)
    slides = data.get("slides", [])
    cleaned = []
    for s in slides:
        cleaned.append({
            "title": s.get("title","Untitled").strip(),
            "bullet_points": [str(x).strip() for x in s.get("bullet_points", [])][:5],
            "hinglish_script": s.get("hinglish_script","").strip(),
            "image_prompt": s.get("image_prompt","").strip()
        })
    return cleaned

# =========================
# Pipeline
# =========================
def build_audio_for_slides(slides: List[Dict[str,Any]], atempo: float):
    audio_meta = []
    progress = st.progress(0.0, text="Generating slide audio (gTTS)…")
    for i, s in enumerate(slides):
        script = s.get("hinglish_script") or f"Chaliye {s.get('title','is slide')} ke baare mein samajhte hain."
        p, dur = gtts_audio(script, i, atempo)
        if p:
            audio_meta.append({"path": p, "duration": float(dur)})
        else:
            audio_meta.append({"path": None, "duration": 0.0})
        progress.progress((i+1)/max(1,len(slides)), text=f"Audio {i+1}/{len(slides)} ready")
    progress.empty()
    return audio_meta

def build_video(slides, audio_meta, palette_index: int, gemini_key: Optional[str]):
    if not FFMPEG_AVAILABLE:
        st.error("FFmpeg not available on system.")
        return None
    tmp = tempfile.gettempdir()
    segments = []
    garbage = []
    progress = st.progress(0.0, text="Rendering slide videos…")

    for i, (s, a) in enumerate(zip(slides, audio_meta)):
        pal = COLOR_PALETTES[palette_index if i==0 else (i % len(COLOR_PALETTES))]
        img = generate_slide_image(s, pal, gemini_key=gemini_key, for_preview=False)
        img_path = os.path.join(tmp, f"slide_{i:02d}.png")
        img.save(img_path)
        garbage.append(img_path)

        if not a["path"]:
            # generate 1s silence
            a["path"] = os.path.join(tmp, f"silence_{i:02d}.mp3")
            subprocess.run(
                ['ffmpeg','-y','-f','lavfi','-i','anullsrc=r=44100:cl=mono','-t','1',a["path"]],
                capture_output=True, text=True
            )

        seg_path = os.path.join(tmp, f"segment_{i:02d}.mp4")
        ok = ffmpeg_concat_image_audio(img_path, a["path"], seg_path)
        if not ok:
            st.error(f"Segment {i+1} failed to render.")
            return None
        segments.append(seg_path)
        garbage.append(seg_path)
        progress.progress((i+1)/max(1,len(slides)), text=f"Rendered {i+1}/{len(slides)}")

    out_path = os.path.join(tmp, "hinglish_lecture.mp4")
    if ffmpeg_join_segments(segments, out_path):
        cleanup(garbage)
        return out_path
    else:
        st.error("Joining segments failed.")
        return None

# =========================
# UI
# =========================
st.markdown(
    "<h1 style='margin-bottom:0'>🎬 PDF ➜ Hinglish Video Lecture</h1>"
    "<p style='opacity:0.8;margin-top:6px'>Inline audio, inline video preview, auto visuals (Wiki→Gemini), speed control, and clean parsing.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("🔑 Keys & Settings")
    gemini_key = st.text_input("Gemini API Key", type="password")
    ss.gemini_model_name = st.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    index=0
    )

    st.caption("Audio uses free gTTS (Hindi accent for Hinglish); FFmpeg needed for speed & video.")

    st.divider()
    st.subheader("🎙️ Voice")
    voice_speed = st.slider("Voice speed (0.5–2.0×)", min_value=0.5, max_value=2.0, value=1.1, step=0.05)

    st.divider()
    st.subheader("🎨 Theme")
    names = [p['name'] for p in COLOR_PALETTES]
    ss.palette_index = st.selectbox(
        "Color Palette",
        options=list(range(len(names))),
        format_func=lambda i: names[i],
        index=DEFAULT_PALETTE_INDEX
    )

    st.divider()
    st.subheader("🧩 Environment")
    if FFMPEG_AVAILABLE:
        st.success("FFmpeg detected")
    else:
        st.error("FFmpeg not found (video & speed change disabled)")

colA, colB = st.columns([2,1])
with colA:
    st.subheader("📄 Upload PDF")
    up = st.file_uploader("Choose a PDF", type=["pdf"])
    generate_btn = st.button("🚀 Generate Slides & Audio", type="primary", use_container_width=True)

with colB:
    st.subheader("📊 Status")
    if ss.slides:
        st.success(f"{len(ss.slides)} slides ready")
    if ss.video_path and os.path.exists(ss.video_path):
        st.video(ss.video_path)  # inline video preview
        with open(ss.video_path, "rb") as vf:
            st.download_button(
                "📥 Download Video (MP4)",
                vf.read(),
                file_name="hinglish_lecture.mp4",
                mime="video/mp4",
                use_container_width=True
            )

# Actions
if generate_btn:
    if not up:
        st.warning("Please upload a PDF first.")
    elif not gemini_key:
        st.warning("Please paste your Gemini API Key in the sidebar.")
    else:
        with st.status("Processing…", expanded=True) as status:
            st.write("1) Extracting text")
            text = extract_text_from_pdf(up.read())
            store_pdf_text_in_db(text, up.name.replace(".pdf",""))
            if not text.strip():
                st.error("No text extracted from PDF.")
                st.stop()

            st.write("2) Generating slides with Gemini")
            try:
                slides = gemini_slides(gemini_key, ss.gemini_model_name, text)
            except Exception as e:
                st.error(f"Slide generation failed: {e}")
                st.stop()

            if not slides:
                st.error("No slides were returned.")
                st.stop()

            ss.slides = slides

            st.write("3) Generating gTTS audio with speed")
            ss.audio = build_audio_for_slides(slides, voice_speed)

            if FFMPEG_AVAILABLE:
                st.write("4) Rendering video with FFmpeg")
                out = build_video(slides, ss.audio, ss.palette_index, gemini_key=gemini_key)
                if out:
                    ss.video_path = out
                    st.success("All done! Inline preview and download are available in the Status panel.")
                    st.balloons()
                    # show video immediately without any extra toggle
                    st.rerun()
            else:
                st.info("FFmpeg missing—generated per-slide audio only.")

# Preview & inline editing
if ss.slides:
    st.divider()
    st.subheader("🧑‍🏫 Review & Tweak Slides")
    tabs = st.tabs([f"Slide {i+1}" for i in range(len(ss.slides))])
    for i, tab in enumerate(tabs):
        with tab:
            s = ss.slides[i]
            # Editable fields (CLEAN labels)
            s['title'] = st.text_input("Title", s['title'], key=f"title_{i}")

            edited_points = st.data_editor(
                [{"Point": p} for p in s.get('bullet_points', [])] or [{"Point": ""}],
                key=f"bp_{i}",
                use_container_width=True
            )
            s['bullet_points'] = [str(row.get("Point","")).strip() for row in edited_points if str(row.get("Point","")).strip()]

            s['hinglish_script'] = st.text_area(
                "Script",
                s.get('hinglish_script', ''),
                key=f"hs_{i}",
                height=140
            )
            s['image_prompt'] = st.text_area(
                "Image Prompt",
                s.get('image_prompt',''),
                key=f"ip_{i}",
                height=100
            )

            # Render current slide preview image (NO script/prompt text drawn on image)
            pal = COLOR_PALETTES[ss.palette_index if i==0 else (i % len(COLOR_PALETTES))]
            preview = generate_slide_image(s, pal, gemini_key=gemini_key, for_preview=True)
            st.image(preview, caption=f"Slide {i+1} preview (theme: {pal['name']})", use_container_width=True)

            # Inline audio playback per slide
            if i < len(ss.audio) and ss.audio[i].get("path") and os.path.exists(ss.audio[i]["path"]):
                with open(ss.audio[i]["path"], "rb") as af:
                    audio_bytes = af.read()
                    st.audio(audio_bytes, format="audio/mp3")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔁 Rebuild Audio (after edits)", use_container_width=True):
            ss.audio = build_audio_for_slides(ss.slides, voice_speed)
            st.success("Audio refreshed.")
    with col2:
        if st.button("🎞️ Rebuild Video", use_container_width=True, disabled=not FFMPEG_AVAILABLE):
            if not FFMPEG_AVAILABLE:
                st.warning("FFmpeg not available.")
            else:
                out = build_video(ss.slides, ss.audio, ss.palette_index, gemini_key=gemini_key)
                if out:
                    ss.video_path = out
                    st.success("Video rebuilt.")
                    st.rerun()
    with col3:
        if st.button("🧹 Reset Project", use_container_width=True):
            ss.slides = []
            ss.audio = []
            ss.video_path = None
            st.rerun()

# =========================
# 💬 AI Chatbot Section
# =========================
st.divider()
st.subheader("💬 AI Chatbot — Ask Doubts from Your Notes")

if not gemini_key:
    st.info("Please enter your Gemini API key in the sidebar.")
elif ss.collection is None:
    st.warning("Upload and process a PDF first to activate chatbot.")
else:
    # Input field for user's question
    user_q = st.text_input("Ask your question about this PDF:")
    
    # Initialize chat history if not present
    if 'chat_history' not in ss:
        ss.chat_history = []

    # When user submits a question
    if user_q:
        with st.spinner("Thinking..."):
            ans = answer_question_with_context(user_q, gemini_key)
        ss.chat_history.append({"q": user_q, "a": ans})

    # Display last few Q&A pairs
    for chat in reversed(ss.chat_history[-10:]):  # show latest 10 messages
        with st.chat_message("user"):
            st.markdown(f"**You:** {chat['q']}")
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {chat['a']}")

    # Clear chat history option
    if st.button("🧹 Clear Chat History"):
        ss.chat_history = []
        st.rerun()