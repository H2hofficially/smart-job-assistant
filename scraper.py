"""Job search on real job portals (Indeed + LinkedIn), with full job descriptions.

Uses python-jobspy, which talks to the portals' own job endpoints, so every
result carries the complete posting text the matcher needs. Glassdoor,
ZipRecruiter, Google Jobs and Naukri are also in jobspy but were blocking
automated requests when this was built, so they are not offered.
"""
import logging
import re

import pandas as pd
from jobspy import scrape_jobs
from jobspy.model import Country

SITES = {"Indeed": "indeed", "LinkedIn": "linkedin"}
JOB_TYPES = {"Any": None, "Full-time": "fulltime", "Part-time": "parttime",
             "Contract": "contract", "Internship": "internship"}


def _country_label(c: Country) -> str:
    name = c.value[0].split(",")[0]
    return name.upper() if len(name) <= 3 else name.title()


COUNTRIES = sorted(_country_label(c) for c in Country if c.name not in ("US_CANADA",))
_COUNTRY_ALIASES = [
    (alias.strip(), _country_label(c))
    for c in Country if c.name not in ("US_CANADA", "WORLDWIDE")
    for alias in c.value[0].split(",") if alias.strip()
] + [("uae", "United Arab Emirates"), ("england", "UK"), ("scotland", "UK"), ("wales", "UK")]


def guess_country(location: str) -> str | None:
    """Map a free-text location like 'Pune, India' to one of COUNTRIES."""
    loc = (location or "").lower()
    for alias, label in _COUNTRY_ALIASES:
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", loc):
            return label
    return None


class _Collect(logging.Handler):
    """Captures jobspy's error logs (e.g. 'LinkedIn rate limited') so the UI can show them."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(f"{record.name.removeprefix('JobSpy:')}: {record.getMessage()[:200]}")


def _jobspy_loggers() -> list[logging.Logger]:
    return [logging.getLogger(n) for n in list(logging.root.manager.loggerDict) if n.startswith("JobSpy:")]


def _attach(collector: logging.Handler) -> None:
    # jobspy creates some loggers lazily, so re-check before every search.
    for lg in _jobspy_loggers():
        if collector not in lg.handlers:
            lg.addHandler(collector)


def search_jobs(terms: list[str], location: str, country: str, sites: list[str],
                results_per_site: int = 15, days: int = 14, remote_only: bool = False,
                job_type: str | None = None) -> tuple[list[dict], list[str]]:
    """Search every term on every site. Returns (de-duplicated jobs, warnings)."""
    collector = _Collect()
    rows, warnings = [], []
    try:
        for term in terms:
            _attach(collector)
            try:
                df = scrape_jobs(
                    site_name=sites,
                    search_term=f"{term} remote" if remote_only else term,
                    location=location or None,
                    country_indeed=country,
                    results_wanted=results_per_site,
                    hours_old=days * 24 if days else None,
                    is_remote=remote_only,
                    job_type=job_type,
                    linkedin_fetch_description=True,
                    description_format="markdown",
                    verbose=0,
                )
            except Exception as e:
                warnings.append(f"Search for '{term}' failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            if df is not None and not df.empty:
                rows.extend(_normalise(r, term) for r in df.to_dict("records"))
    finally:
        for lg in _jobspy_loggers():
            lg.removeHandler(collector)

    warnings += list(dict.fromkeys(collector.messages))
    jobs = dedupe(rows)
    # Indeed ignores the remote / job-type filters when a date window is set, so enforce them here.
    if remote_only:
        jobs = [j for j in jobs if j["is_remote"] or "remote" in f"{j['location']} {j['title']}".lower()]
    if job_type:
        jobs = [j for j in jobs if not j["job_type"] or job_type in j["job_type"]]
    return jobs, warnings


def _text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _salary(r: dict) -> str:
    lo, hi = _num(r.get("min_amount")), _num(r.get("max_amount"))
    if lo is None and hi is None:
        return ""
    if lo is not None and hi is not None and lo != hi:
        amount = f"{lo:,.0f}–{hi:,.0f}"
    else:
        amount = f"{lo if lo is not None else hi:,.0f}"
    interval = _text(r.get("interval"))
    return " ".join(x for x in (_text(r.get("currency")), amount, f"/ {interval}" if interval else "") if x)


def _normalise(r: dict, term: str) -> dict:
    site = _text(r.get("site"))
    return {
        "id": f"{site}:{_text(r.get('id')) or _text(r.get('job_url'))}",
        "site": {"indeed": "Indeed", "linkedin": "LinkedIn"}.get(site, site.title()),
        "title": _text(r.get("title")),
        "company": _text(r.get("company")) or "Unknown company",
        "location": _text(r.get("location")) or "Not specified",
        "posted": _text(r.get("date_posted"))[:10],
        "job_type": _text(r.get("job_type")),
        "is_remote": _text(r.get("is_remote")) == "True",
        "salary": _salary(r),
        "url": _text(r.get("job_url")),
        "apply_url": _text(r.get("job_url_direct")),
        "description": clean_description(_text(r.get("description"))),
        "search_term": term,
        "company_size": _text(r.get("company_num_employees")),
        "company_industry": _text(r.get("company_industry")),
        "company_about": _text(r.get("company_description"))[:600],
    }


_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                         "−": "-", " ": " "})


def normalize_dashes(s: str) -> str:
    """Replace typographic dashes and non-breaking spaces (common in AI output) with plain ASCII."""
    return (s or "").translate(_DASHES)


def ascii_lower(s: str) -> str:
    """Lowercase + plain dashes, so AI-written phrases can be matched against posting text."""
    return normalize_dashes(s).lower()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def dedupe(jobs: list[dict]) -> list[dict]:
    """Same title at the same company counts once; keep the copy with the longest description."""
    best: dict[tuple, dict] = {}
    for j in jobs:
        if not j["title"]:
            continue
        key = (_norm(j["title"]), _norm(j["company"]))
        if key not in best or len(j["description"]) > len(best[key]["description"]):
            best[key] = j
    return list(best.values())


_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")
_BOILERPLATE = re.compile(
    r"equal (employment )?opportunit|\beeo\b|affirmative action|reasonable accommodation|e-verify"
    r"|without regard to|protected veteran|privacy (policy|notice)|pay transparency",
    re.I,
)


def clean_description(md: str) -> str:
    md = _IMAGE.sub("", md)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", md).strip()


def trim_for_llm(description: str, limit: int = 4000) -> str:
    """Shorter posting text for the LLM: drop legal boilerplate and markdown noise, cap the length."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", description) if p.strip()]
    text = "\n\n".join(p for p in paras if not _BOILERPLATE.search(p)) or description
    text = _MD_ESCAPE.sub(r"\1", text).replace("**", "")
    if len(text) > limit:
        cut = text[:limit]
        # End on a line break if one is near the limit, otherwise on a word; postings are often one long block.
        end = cut.rfind("\n", int(limit * 0.8))
        text = cut[:end if end > 0 else cut.rfind(" ")].rstrip() + "\n[...truncated]"
    return text


if __name__ == "__main__":
    found, warns = search_jobs(["python developer"], "Austin, TX", "USA", ["indeed", "linkedin"], 5)
    print(f"Found {len(found)} jobs")
    for w in warns:
        print("WARNING:", w)
    for j in found:
        print(f"- [{j['site']}] {j['title']} | {j['company']} | {j['location']} | "
              f"{len(j['description'])} chars | {j['url']}")
