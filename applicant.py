"""Your saved application answers (contact details, visa status, EEO choices) and resume copy for autofill.py.

Everything lives in my_data/ next to the app, on this computer only (git-ignored). Nothing is saved until you
press Save in the app's auto-fill section.
"""
import io
import json
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

MY_DATA = Path(__file__).with_name("my_data")
ANSWERS_FILE = MY_DATA / "applicant.json"
RESUME_TEXT_FILE = MY_DATA / "resume.txt"

YES_NO = ["Yes", "No"]
DECLINE = "Decline to self-identify"

# (key, label, kind, default). kind: text | area | yesno | choice:<a>|<b>|...
FIELDS = [
    ("first_name", "First name", "text", ""),
    ("last_name", "Last name", "text", ""),
    ("preferred_name", "Preferred first name (optional)", "text", ""),
    ("email", "Email", "text", ""),
    ("phone", "Phone", "text", ""),
    ("linkedin", "LinkedIn URL", "text", ""),
    ("github", "GitHub URL (optional)", "text", ""),
    ("website", "Website / portfolio (optional)", "text", ""),
    ("city", "City, State (e.g. Austin, TX)", "text", ""),
    ("country", "Country you live in", "text", "United States"),
    ("current_company", "Current or last employer", "text", ""),
    ("current_title", "Current or last job title", "text", ""),
    ("work_authorized", "Legally authorized to work in the US?", "yesno", "Yes"),
    ("needs_sponsorship", "Will you now or in the future need visa sponsorship?", "yesno", "No"),
    ("visa_status", "Visa / citizenship status in your own words (optional)", "text", ""),
    ("relocate", "Open to relocating?", "choice:Yes|No|Depends on the role", "Depends on the role"),
    ("onsite", "Willing to work on-site / hybrid in an office?", "choice:Yes|No|Remote only", "Yes"),
    ("been_at", "Companies you've worked or interviewed at before, comma-separated (all others get 'No')", "text", ""),
    ("salary", "Salary expectation (optional, e.g. $140,000-$160,000)", "text", ""),
    ("start", "Earliest start / notice period", "text", "2 weeks after an offer"),
    ("how_heard", "How did you hear about us? (default answer)", "text", "Company careers page"),
    ("pronouns", "Pronouns (optional)", "text", ""),
    ("gender", "Gender (EEO)", "text", DECLINE),
    ("hispanic", "Hispanic or Latino? (EEO)", "text", DECLINE),
    ("race", "Race / ethnicity (EEO)", "text", DECLINE),
    ("veteran", "Veteran status (EEO)", "text", DECLINE),
    ("disability", "Disability status (EEO)", "text", DECLINE),
    ("extra_facts", "Other facts the AI may use in answers (e.g. 'Interviewed at Stripe in 2024', 'Fluent in "
                    "Spanish'). One per line.", "area", ""),
]
REQUIRED = ("first_name", "last_name", "email", "phone")


def defaults() -> dict:
    return {k: d for k, _, _, d in FIELDS}


def load() -> dict:
    try:
        saved = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return defaults() | {k: v for k, v in saved.items() if isinstance(v, str)}


def resume_file() -> Path | None:
    return next(iter(sorted(MY_DATA.glob("resume.pdf")) + sorted(MY_DATA.glob("resume.docx"))), None)


def ready() -> bool:
    a = load()
    return ANSWERS_FILE.exists() and all(a.get(k, "").strip() for k in REQUIRED) and resume_file() is not None


def save(answers: dict, resume_name: str | None = None, resume_bytes: bytes | None = None,
         resume_text: str | None = None) -> None:
    MY_DATA.mkdir(exist_ok=True)
    clean = {k: str(answers.get(k, "")).strip() for k, _, _, _ in FIELDS}
    ANSWERS_FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    if resume_bytes and resume_name and resume_name.lower().endswith((".pdf", ".docx")):
        for old in MY_DATA.glob("resume.*"):
            old.unlink()
        (MY_DATA / f"resume{Path(resume_name).suffix.lower()}").write_bytes(resume_bytes)
    if resume_text:
        RESUME_TEXT_FILE.write_text(resume_text, encoding="utf-8")


def save_tailored_resume(text: str, job: dict) -> Path:
    """Save an edited tailored resume as .docx in its own folder per job. The file is named like a normal resume
    (First_Last_Resume.docx) because employers see the name of the uploaded file."""
    a = load()
    folder = MY_DATA / "tailored" / re.sub(r"[^A-Za-z0-9]+", "_", f"{job['company']}_{job['title']}")[:80]
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9]+", "_", f"{a['first_name']} {a['last_name']}".strip()) or "My"
    path = folder / f"{name}_Resume.docx"
    path.write_bytes(docx_bytes(text))
    (folder / "resume.txt").write_text(text, encoding="utf-8")
    return path


def _is_heading(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") or (s.isupper() and 2 < len(s) < 45 and any(c.isalpha() for c in s))


def _is_role(line: str) -> bool:
    """A job or degree line such as 'Founder — DarkoApp | Feb 2026 – Present': short, has a year, not a sentence."""
    s = line.strip()
    return len(s) < 130 and bool(re.search(r"\b(19|20)\d{2}\b", s)) and not s.endswith(".")


def docx_bytes(text: str) -> bytes:
    """A plain, ATS-friendly Word file from resume text: line 1 = name, line 2 = contact, CAPITAL headings with a
    rule under them, '- ' bullets. Built by hand (a .docx is zipped XML) so no extra library is needed."""
    def run(t: str, bold: bool = False, size: int = 21) -> str:
        return (f'<w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>{"<w:b/>" if bold else ""}'
                f'<w:sz w:val="{size}"/></w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r>')

    paras = []
    for i, line in enumerate(text.strip().splitlines()):
        s = line.strip().lstrip("#").strip()
        if not s:
            continue
        if i == 0:  # child order inside w:pPr is fixed by the schema: pBdr, spacing, ind, jc
            ppr, r = '<w:spacing w:after="40"/><w:jc w:val="center"/>', run(s, True, 32)
        elif i == 1 and not _is_heading(line):
            ppr, r = '<w:spacing w:after="120"/><w:jc w:val="center"/>', run(s, size=19)
        elif _is_heading(line):
            ppr = ('<w:pBdr><w:bottom w:val="single" w:sz="4" w:space="1" w:color="808080"/></w:pBdr>'
                   '<w:spacing w:before="200" w:after="80"/>')
            r = run(s.upper(), True, 22)
        elif s[:2] in ("- ", "* ", "• "):
            ppr, r = '<w:spacing w:after="40"/><w:ind w:left="360" w:hanging="220"/>', run("•  " + s[2:].strip())
        elif _is_role(s):  # bold job lines so a reader can scan the roles
            ppr, r = '<w:spacing w:before="120" w:after="40"/>', run(s, True)
        else:
            ppr, r = '<w:spacing w:after="60"/>', run(s)
        paras.append(f"<w:p><w:pPr>{ppr}</w:pPr>{r}</w:p>")
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + "".join(paras) +
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="864" w:bottom="720" '
        'w:left="864" w:header="0" w:footer="0" w:gutter="0"/></w:sectPr></w:body></w:document>')
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="word/document.xml" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
    return buf.getvalue()


def facts_text(a: dict) -> str:
    """The answers as plain lines for the AI that writes free-text application answers."""
    labels = {k: label for k, label, _, _ in FIELDS}
    return "\n".join(f"{labels[k]}: {v}" for k, v in a.items() if k in labels and v and k != "extra_facts") + (
        f"\nOther facts:\n{a['extra_facts']}" if a.get("extra_facts") else "")
