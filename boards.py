"""Jobs straight from company career boards: Greenhouse, Lever and Ashby.

All three publish free, public job-board APIs, so there is no scraping and no blocking, and every job has
its full description. You choose which companies to watch in companies.txt (one 'ats:slug  Name' per line).
Every search also adds companies it spots in Indeed/LinkedIn apply links. Jobs from these boards are the
easiest ones for autofill.py to fill in.
"""
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from markdownify import markdownify

from matcher import title_level, title_relevance
from scraper import clean_description, guess_country

COMPANIES_FILE = Path(__file__).with_name("companies.txt")
ATS = ("greenhouse", "lever", "ashby")
_TTL = 3600  # a board's job list is fetched at most once an hour
_cache: dict[str, tuple[float, list[dict]]] = {}
_http = requests.Session()
_http.headers["User-Agent"] = "SmartJobAssistant/1.0 (personal job search)"

_US_STATES = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK "
    "OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
)
_US_WORDS = re.compile(
    r"\b(united states|usa|u\.s\.|us|san francisco|new york|nyc|seattle|boston|austin|chicago|los angeles|"
    r"denver|atlanta|miami|palo alto|mountain view|menlo park|sunnyvale|san jose|san mateo|redwood city|"
    r"oakland|berkeley|cambridge, ma|washington,? d\.?c|portland|philadelphia|pittsburgh|salt lake)\b", re.I)
_CITIES = {"london": "UK", "dublin": "Ireland", "bengaluru": "India", "bangalore": "India", "hyderabad": "India",
           "pune": "India", "mumbai": "India", "gurgaon": "India", "gurugram": "India", "toronto": "Canada",
           "vancouver": "Canada", "montreal": "Canada", "berlin": "Germany", "munich": "Germany", "paris": "France",
           "amsterdam": "Netherlands", "singapore": "Singapore", "sydney": "Australia", "melbourne": "Australia",
           "tel aviv": "Israel", "tokyo": "Japan", "madrid": "Spain", "barcelona": "Spain", "stockholm": "Sweden",
           "zurich": "Switzerland", "warsaw": "Poland", "sao paulo": "Brazil", "são paulo": "Brazil",
           "mexico city": "Mexico", "lisbon": "Portugal", "copenhagen": "Denmark", "seoul": "South Korea"}
_ISO = {"US": "USA", "GB": "UK", "CA": "Canada", "IN": "India", "AU": "Australia", "DE": "Germany",
        "FR": "France", "IE": "Ireland", "NL": "Netherlands", "ES": "Spain", "SG": "Singapore", "JP": "Japan",
        "BR": "Brazil", "MX": "Mexico", "PL": "Poland", "SE": "Sweden", "CH": "Switzerland", "IL": "Israel"}
_JOB_TYPES = {"fulltime": "fulltime", "full-time": "fulltime", "permanent": "fulltime", "parttime": "parttime",
              "part-time": "parttime", "contract": "contract", "contractor": "contract", "temporary": "contract",
              "intern": "internship", "internship": "internship"}


# ------------------------------------------------------------ company list ---

def load_companies() -> list[tuple[str, str, str]]:
    """(ats, slug, display name) for every line of companies.txt."""
    out, seen = [], set()
    if not COMPANIES_FILE.exists():
        return out
    for line in COMPANIES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"(greenhouse|lever|ashby):([A-Za-z0-9._-]+)\s*(.*)", line)
        if m and (m[1], m[2].lower()) not in seen:
            seen.add((m[1], m[2].lower()))
            out.append((m[1], m[2], m[3].strip() or m[2]))
    return out


_LEARN = [
    ("greenhouse", re.compile(r"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_app\?for=)?([A-Za-z0-9_-]+)")),
    ("lever", re.compile(r"jobs\.lever\.co/([A-Za-z0-9._-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9._%-]+)")),
]


def learn_companies(jobs: list[dict]) -> list[str]:
    """Add boards seen in apply links (e.g. an Indeed job that applies on Greenhouse) to companies.txt."""
    known = {(a, s.lower()) for a, s, _ in load_companies()}
    new = []
    for j in jobs:
        for ats, rx in _LEARN:
            m = rx.search(j.get("apply_url") or "")
            if m and m[1].lower() not in ("embed", "v1") and (ats, m[1].lower()) not in known:
                known.add((ats, m[1].lower()))
                new.append(f"{ats}:{m[1]}  {j['company']}")
    if new:
        with COMPANIES_FILE.open("a", encoding="utf-8") as f:
            f.write("\n".join(new) + "\n")
    return new


# ---------------------------------------------------------------- fetching ---

def _get(url: str):
    r = _http.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _to_md(raw_html: str) -> str:
    return clean_description(markdownify(html.unescape(raw_html or ""), heading_style="ATX"))


def _date(value) -> str:
    if isinstance(value, (int, float)):  # Lever: epoch milliseconds
        return datetime.fromtimestamp(value / 1000, timezone.utc).strftime("%Y-%m-%d")
    return str(value or "")[:10]


def _base(ats: str, slug: str, name: str, **kw) -> dict:
    job = {"site": ats.title(), "company": name, "job_type": "", "is_remote": False, "salary": "",
           "description": "", "search_term": "", "company_size": "", "company_industry": "", "company_about": "",
           "country": None, "ats": ats, "slug": slug}
    job.update(kw)
    job["is_remote"] = job["is_remote"] or "remote" in job["location"].lower()
    return job


def _greenhouse(slug: str, name: str) -> list[dict]:
    jobs = _get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs")["jobs"]
    return [_base("greenhouse", slug, j.get("company_name") or name,
                  id=f"greenhouse:{j['id']}", gh_id=j["id"], title=j["title"].strip(),
                  location=(j.get("location") or {}).get("name") or "Not specified",
                  posted=_date(j.get("first_published") or j.get("updated_at")),
                  url=j["absolute_url"], apply_url=j["absolute_url"])
            for j in jobs]


def _lever(slug: str, name: str) -> list[dict]:
    out = []
    for j in _get(f"https://api.lever.co/v0/postings/{slug}?mode=json"):
        cat = j.get("categories") or {}
        sal = j.get("salaryRange") or {}
        lists = "\n\n".join(f"### {x.get('text', '')}\n{_to_md(x.get('content', ''))}" for x in j.get("lists") or [])
        out.append(_base(
            "lever", slug, name, id=f"lever:{j['id']}", title=j["text"].strip(),
            location=cat.get("location") or "Not specified", posted=_date(j.get("createdAt")),
            job_type=_JOB_TYPES.get((cat.get("commitment") or "").lower(), ""),
            is_remote=j.get("workplaceType") == "remote", country=_ISO.get(j.get("country") or ""),
            salary=f"{sal.get('currency', '')} {sal['min']:,}–{sal['max']:,} / {sal.get('interval', '')}".strip()
            if sal.get("min") and sal.get("max") else "",
            description=clean_description("\n\n".join(
                x for x in (j.get("openingPlain"), j.get("descriptionBodyPlain") or j.get("descriptionPlain"),
                            lists, j.get("additionalPlain")) if x)),
            url=j["hostedUrl"], apply_url=j.get("applyUrl") or j["hostedUrl"] + "/apply"))
    return out


def _ashby(slug: str, name: str) -> list[dict]:
    out = []
    for j in _get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true")["jobs"]:
        if j.get("isListed") is False:
            continue
        addr = ((j.get("address") or {}).get("postalAddress") or {}).get("addressCountry") or ""
        out.append(_base(
            "ashby", slug, name, id=f"ashby:{j['id']}", title=j["title"].strip(),
            location=j.get("location") or "Not specified", posted=_date(j.get("publishedAt")),
            job_type=_JOB_TYPES.get((j.get("employmentType") or "").lower(), ""),
            is_remote=bool(j.get("isRemote")) or j.get("workplaceType") == "Remote",
            country=guess_country(addr) if addr else None,
            salary=((j.get("compensation") or {}).get("compensationTierSummary") or ""),
            description=clean_description(j.get("descriptionPlain") or _to_md(j.get("descriptionHtml", ""))),
            url=j["jobUrl"], apply_url=j.get("applyUrl") or j["jobUrl"] + "/application"))
    return out


_FETCH = {"greenhouse": _greenhouse, "lever": _lever, "ashby": _ashby}


def fetch_board(ats: str, slug: str, name: str) -> list[dict]:
    key = f"{ats}:{slug}"
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    jobs = _FETCH[ats](slug, name)
    _cache[key] = (time.time(), jobs)
    return jobs


def _greenhouse_content(job: dict) -> None:
    if job["ats"] == "greenhouse" and not job["description"]:
        d = _get(f"https://boards-api.greenhouse.io/v1/boards/{job['slug']}/jobs/{job['gh_id']}")
        job["description"] = _to_md(d.get("content", ""))


# --------------------------------------------------------------- filtering ---

def job_country(job: dict) -> str | None:
    if job.get("country"):
        return job["country"]
    loc = job["location"]
    if any(code in _US_STATES for code in re.findall(r",\s*([A-Z]{2})\b", loc)) or _US_WORDS.search(loc):
        return "USA"
    return guess_country(loc) or next((c for city, c in _CITIES.items() if city in loc.lower()), None)


def _keep(job: dict, location: str, country: str, cutoff: str, remote_only: bool, job_type: str | None) -> bool:
    if cutoff and job["posted"] and job["posted"] < cutoff:
        return False
    if remote_only and not job["is_remote"]:
        return False
    if job_type and job["job_type"] and job["job_type"] != job_type:
        return False
    jc = job_country(job)
    if country and country != "Worldwide" and jc and jc != country:
        return False
    city = location.split(",")[0].strip().lower()
    return not city or job["is_remote"] or city in job["location"].lower()


def search_boards(terms: list[str], location: str, country: str, days: int = 14, remote_only: bool = False,
                  job_type: str | None = None, max_jobs: int = 300,
                  ats: list[str] | tuple = ATS, levels=None) -> tuple[list[dict], list[str]]:
    """Every watched company's open jobs whose title matches a search term. Returns (jobs, warnings)."""
    companies = [c for c in load_companies() if c[0] in ats]
    warnings: list[str] = []

    def fetch(c):
        try:
            return fetch_board(*c)
        except Exception as e:
            warnings.append(f"{c[0]}:{c[1]}: {type(e).__name__}")
            return []

    with ThreadPoolExecutor(16) as ex:
        everything = [j for board in ex.map(fetch, companies) for j in board]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d") if days else ""
    found = []
    for job in everything:
        if levels and (lv := title_level(job["title"])) and lv not in levels:
            continue  # e.g. 'Senior ...' when you only want junior jobs: skip before the max_jobs cap
        best = max(terms, key=lambda t: title_relevance(job["title"], [t]))
        # Stricter than the Indeed filter (0.5): boards list every job a company has, so '...Product Manager'
        # must not match 'Product Recruiter'. Two-word titles need both words; longer ones two thirds.
        if title_relevance(job["title"], [best]) >= 0.66 and _keep(job, location, country, cutoff, remote_only,
                                                                    job_type):
            found.append(job | {"search_term": best})
    found.sort(key=lambda j: j["posted"], reverse=True)
    found = found[:max_jobs]
    with ThreadPoolExecutor(16) as ex:  # Greenhouse lists carry no description: fetch only the matches
        list(ex.map(lambda j: _safe(_greenhouse_content, j), found))
    if len(warnings) > 5:
        warnings = warnings[:5] + [f"...and {len(warnings) - 5} more boards could not be read"]
    return found, [f"Company board {w}" for w in warnings]


_SUFFIX = re.compile(r"\b(inc|llc|ltd|limited|corp|corporation|co|company|technologies|technology|labs|hq|group)\b\.?")


def find_on_boards(company: str, title: str) -> dict | None:
    """The same job on the company's own Greenhouse/Lever/Ashby board, or None. LinkedIn and Indeed hide the
    company's apply link from logged-out visitors, so jobs from them often arrive without one."""
    name = company.lower().split(",")[0]
    short = _SUFFIX.sub(" ", name)
    slugs = dict.fromkeys(s for s in (re.sub(r"[^a-z0-9]+", "", name), re.sub(r"[^a-z0-9]+", "", short),
                                      re.sub(r"[^a-z0-9]+", "-", short.strip()).strip("-")) if s)
    known = [(a, s) for a, s, n in load_companies() if _SUFFIX.sub(" ", n.lower()).strip() == short.strip()]
    for ats, slug in dict.fromkeys(known + [(a, s) for s in slugs for a in ATS]):
        try:
            jobs = fetch_board(ats, slug, company)
        except Exception:
            continue  # no board under that name
        best = max(jobs, key=lambda j: title_relevance(j["title"], [title]), default=None)
        if best and title_relevance(best["title"], [title]) >= 0.8 and title_relevance(title, [best["title"]]) >= 0.8:
            _safe(_greenhouse_content, best)
            return best
    return None


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception:
        pass


if __name__ == "__main__":
    t = time.time()
    jobs, warns = search_boards(["Product Manager"], "", "USA", days=14, max_jobs=50)
    print(f"{len(load_companies())} companies, {len(jobs)} matching jobs in {time.time() - t:.0f}s")
    for w in warns:
        print("WARNING:", w)
    for j in jobs[:15]:
        print(f"- [{j['site']}] {j['title']} | {j['company']} | {j['location']} | {j['posted']} | "
              f"{len(j['description'])} chars")
