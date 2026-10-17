"""Resume parsing (in memory, never written to disk) and safe-rendering helpers."""
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from pypdf import PdfReader

MAX_PDF_PAGES = 10
MAX_RESUME_CHARS = 20_000
MAX_DOCX_XML_BYTES = 20_000_000
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ResumeError(ValueError):
    """The uploaded file could not be turned into usable resume text."""


def extract_resume_text(filename: str, data: bytes) -> str:
    """Return plain text from a PDF, DOCX or TXT resume held in memory."""
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _pdf_text(data)
    elif name.endswith(".docx"):
        text = _docx_text(data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ResumeError("Unsupported file type. Upload a PDF, DOCX or TXT resume.")

    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    if len(text) < 200:
        raise ResumeError(
            "Could not read enough text from this file. If it is a scanned or image-only PDF, "
            "export your resume as a text PDF or DOCX and upload that instead."
        )
    return text[:MAX_RESUME_CHARS]


def _pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumeError("This PDF is password-protected. Remove the password and upload it again.")
        parts = []
        for i, page in enumerate(reader.pages):
            if i >= MAX_PDF_PAGES:
                break
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except ResumeError:
        raise
    except Exception as e:
        raise ResumeError(f"Could not read this PDF ({type(e).__name__}).") from e


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise ResumeError("This DOCX is too large to be a resume.")
            root = ET.fromstring(z.read(info))
    except ResumeError:
        raise
    except Exception as e:
        raise ResumeError(f"Could not read this DOCX ({type(e).__name__}).") from e
    return "\n".join(
        "".join(t.text or "" for t in p.iter(f"{_W}t")) for p in root.iter(f"{_W}p")
    )


# --- rendering untrusted text (scraped job data and LLM output) -------------

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def md_escape(text) -> str:
    """Escape a short scraped string so it renders literally in st.markdown."""
    return _MD_SPECIAL.sub(r"\\\1", str(text or ""))


def untrusted_md(text) -> str:
    """Markdown from the web or an LLM: drop images (remote trackers) and stop `$` turning into LaTeX."""
    text = _MD_IMAGE.sub("", str(text or ""))
    return _UNESCAPED_DOLLAR.sub(r"\\$", text)


def safe_url(url) -> str | None:
    """Only allow plain http(s) links through to buttons and tables."""
    url = str(url or "").strip()
    parsed = urlparse(url)
    return url if parsed.scheme in ("http", "https") and parsed.netloc else None


def csv_safe(value):
    """Neutralise spreadsheet formulas (CSV injection) in scraped text before export."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value
