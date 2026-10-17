"""Smart Job Assistant.

Upload a resume -> the AI reads it into a profile and suggests job titles ->
search Indeed + LinkedIn + company career boards (Greenhouse, Lever, Ashby) ->
pre-rank every job by keyword overlap -> AI-score the best ones against their
full descriptions, on several Groq models at once -> cover letters, resume
tweaks, a tailored portfolio page and browser auto-fill of the application on demand.
"""
import base64
import hashlib
import json
import queue
import re
import threading
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import applicant
import autofill
import boards
import llm
import portfolio
from matcher import (LEVELS, OUTSIDE_PENALTY, build_profile, clean_direction, draft_directions, keyword_score, match_job,
                     job_level, parse_direction, title_relevance)
from scraper import COUNTRIES, JOB_TYPES, SITES, dedupe, guess_country, search_jobs
from tailor import cover_letter, tailored_resume
from utils import ResumeError, csv_safe, extract_resume_text, md_escape, safe_url, untrusted_md

st.set_page_config(page_title="Smart Job Assistant", page_icon="💼", layout="wide")

ss = st.session_state
for key, default in {
    "resume_hash": None, "resume_text": None, "resume_error": None,
    "profile": None, "profile_error": None,
    "jobs": [], "matches": {}, "docs": {}, "search_warnings": [],
    "terms": "", "location": "", "country": "USA",
    "directions": [], "directions_error": None, "dir_base": None, "dir_version": 0, "own_direction": "",
    "levels": list(LEVELS),
}.items():
    if key not in ss:
        ss[key] = default


@st.cache_data(ttl=3600, show_spinner=False)
def model_options() -> list[str]:
    return llm.list_models()


CAUTION_COLORS = {"go": "green", "selective": "orange", "careful": "red"}
BOARD_PORTALS = {"Greenhouse": "greenhouse", "Lever": "lever", "Ashby": "ashby"}


def level_text(job: dict) -> str:
    level = job.get("level", "")
    return f"{level} ({job['years_required']}+ yrs)" if job.get("years_required") else level


def portal_counts(jobs: list[dict]) -> str:
    return " · ".join(f"{site} {n}" for site, n in Counter(j["site"] for j in jobs).most_common())
LIST_COLS = ("proof", "search_titles", "good_signals", "bad_signals")
DIR_COLS = ["use", "role", "caution", "fits_where", "avoid_where", *LIST_COLS]


def _cell(v) -> str:
    return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()


def directions_to_df(directions: list[dict]) -> pd.DataFrame:
    rows = [{"use": d.get("use", True), **{k: "; ".join(d[k]) if k in LIST_COLS else d[k] for k in DIR_COLS[1:]}}
            for d in directions]
    return pd.DataFrame(rows, columns=DIR_COLS)


def df_to_directions(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        role = _cell(r.get("role"))
        if not role:
            continue
        d = {k: [s.strip() for s in _cell(r.get(k)).split(";") if s.strip()] for k in LIST_COLS}
        d.update(role=role, caution=_cell(r.get("caution")), fits_where=_cell(r.get("fits_where")),
                 avoid_where=_cell(r.get("avoid_where")))
        d["search_titles"] = d["search_titles"] or [role]
        # _cell() turns numpy False into "False"; a new row (None/NaN) counts as ticked.
        out.append(clean_direction(d) | {"use": _cell(r.get("use")) != "False"})
    return out


def set_directions(directions: list[dict]) -> None:
    """Replace the editor's base table. A new key makes a fresh editor (old edits must not replay onto it)."""
    ss.directions = directions
    ss.dir_base = directions_to_df(directions)
    ss.dir_version += 1


def terms_from_directions(directions: list[dict]) -> list[str]:
    """Up to 6 search titles, 2 per ticked direction. Newest rows (yours) first so they are never cut off."""
    seen, terms = set(), []
    for d in reversed(directions):
        if d.get("use", True):
            for t in d["search_titles"][:2]:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    terms.append(t)
    return terms[:6]


def direction_of(job: dict) -> dict | None:
    # reversed: rows you added sit lower in the table and must win over the AI's draft with the same name.
    return next((d for d in reversed(ss.directions) if d["role"] == job.get("direction")), None)


def use_directions_for_search() -> None:
    # Callback: runs before the sidebar is drawn, so setting the 'terms' widget value is allowed here.
    ss.terms = "\n".join(terms_from_directions(ss.directions))


MY_DIRECTIONS = applicant.MY_DATA / "my_directions.json"


def saved_directions() -> list[dict]:
    """Directions you added yourself, kept in my_data/my_directions.json across resume uploads."""
    try:
        return [clean_direction(d) for d in json.loads(MY_DIRECTIONS.read_text(encoding="utf-8"))]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def remember_direction(d: dict) -> None:
    keep = [x for x in saved_directions() if x["role"].lower() != d["role"].lower()]
    MY_DIRECTIONS.parent.mkdir(exist_ok=True)
    MY_DIRECTIONS.write_text(json.dumps(keep + [clean_direction(d)], indent=2), encoding="utf-8")


def with_saved(directions: list[dict]) -> list[dict]:
    """The AI's draft plus your saved directions. Yours replace a draft row of the same name and sit last."""
    mine = saved_directions()
    roles = {d["role"].lower() for d in mine}
    return [d for d in directions if d["role"].lower() not in roles] + [d | {"use": True} for d in mine]


def add_own_direction() -> None:
    text = ss.own_direction.strip()
    if not text:
        return
    try:
        new = parse_direction(text, ss.resume_text, ss.model)
    except Exception as e:
        ss.directions_error = llm.explain(e)
        return
    ss.directions_error = None
    remember_direction(new)
    set_directions(ss.directions + [new | {"use": True}])
    ss.own_direction = ""
    use_directions_for_search()
    st.toast(f"Added direction: {new['role']}")


def load_resume(upload) -> None:
    """Parse a newly uploaded resume and have the AI read it (once per file)."""
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if ss.resume_hash != digest:
        ss.update(resume_hash=digest, resume_text=None, resume_error=None, profile=None,
                  profile_error=None, jobs=[], matches={}, docs={}, search_warnings=[],
                  directions=[], directions_error=None, dir_base=None)
        try:
            ss.resume_text = extract_resume_text(upload.name, data)
        except ResumeError as e:
            ss.resume_error = str(e)
    if ss.resume_text and ss.profile is None and ss.profile_error is None:
        with st.spinner("Reading your resume…"):
            try:
                ss.profile = build_profile(ss.resume_text, ss.get("model", llm.DEFAULT_MODEL))
            except Exception as e:
                ss.profile_error = llm.explain(e)
                return
        # Pre-fill the search from what the AI read. Safe here: those widgets aren't created yet this run.
        with st.spinner("Working out your career directions…"):
            try:
                drafted = draft_directions(ss.resume_text, ss.get("model", llm.DEFAULT_MODEL))
            except Exception as e:
                drafted, ss.directions_error = [], llm.explain(e)
        if drafted or saved_directions():
            set_directions(with_saved(drafted))
        ss.terms = "\n".join(terms_from_directions(ss.directions) or ss.profile["target_titles"][:3])
        ss.location = ss.profile["location"]
        ss.country = guess_country(ss.profile["location"]) or ss.country


def score_jobs(jobs: list[dict], model: str) -> None:
    """AI-score jobs, one worker thread per Groq model (each model has its own free-tier limit).
    Workers never touch st.*: they hand results back through a queue and this thread updates the page."""
    todo = [j for j in jobs if j["id"] not in ss.matches]
    if not todo:
        return
    models = llm.scoring_pool(model, model_options()) if ss.get("parallel", True) else [model]
    work, results = queue.Queue(), queue.Queue()
    for job in todo:
        work.put((job, direction_of(job), 0))
    profile, stop = ss.profile, threading.Event()

    def worker(m: str) -> None:
        while not stop.is_set():
            try:
                job, direction, tries = work.get_nowait()
            except queue.Empty:
                return
            try:
                results.put((job, match_job(profile, job, m, direction=direction) | {"model": m}, None))
            except Exception as e:
                if llm.is_daily_limit(e) and any(t.is_alive() and t.name != m for t in threads):
                    work.put((job, direction, tries))  # this model is done for today: the others take its jobs
                    return
                if tries == 0 and len(models) > 1 and not isinstance(e, RuntimeError):
                    work.put((job, direction, 1))  # one retry, usually picked up by another model
                else:
                    results.put((job, None, e))
                if llm.is_fatal(e) or llm.is_daily_limit(e):  # this model can't go on; the others carry on
                    return

    threads = [threading.Thread(target=worker, args=(m,), name=m, daemon=True) for m in models]
    for t in threads:
        t.start()
    bar = st.progress(0.0, text=f"AI-scoring {len(todo)} jobs on {len(models)} model(s)…")
    done, errors = 0, []
    while any(t.is_alive() for t in threads) or not results.empty():
        try:
            job, match, err = results.get(timeout=0.5)
        except queue.Empty:
            continue
        done += 1
        if match:
            ss.matches[job["id"]] = match
        else:
            errors.append(f"{md_escape(job['title'])}: {llm.explain(err)}")
            if isinstance(err, RuntimeError) or type(err).__name__ == "AuthenticationError":
                stop.set()
        bar.progress(min(1.0, done / len(todo)), text=f"AI-scored {done}/{len(todo)}: {md_escape(job['title'])} "
                                                      f"at {md_escape(job['company'])}")
    bar.empty()
    for msg in list(dict.fromkeys(errors))[:3]:
        st.error(msg)
    if len(errors) > 3:
        st.caption(f"…and {len(errors) - 3} more scoring errors.")
    if not work.empty():
        st.warning(f"{work.qsize()} jobs weren't scored: every model hit a Groq limit. Try again later, or tomorrow "
                   "if it's the daily limit (about 100 jobs per model per day on the free tier).")


def run_search(terms, location, country, sites, per_site, days, job_type, remote_only, top_n, model,
               ats=("greenhouse", "lever", "ashby"), board_max=300, levels=tuple(LEVELS)) -> None:
    with st.status("Finding jobs…", expanded=True) as status:
        jobs, ss.search_warnings = [], []
        if sites:
            st.write(f"Searching {', '.join(sites)} for: {md_escape(', '.join(terms))}")
            jobs, ss.search_warnings = search_jobs(
                terms, location, country, [SITES[s] for s in sites], per_site, days, remote_only, JOB_TYPES[job_type]
            )
            if learned := boards.learn_companies(jobs):
                st.write(f"Learned {len(learned)} new company boards from apply links (added to companies.txt).")
        if ats:
            names = ", ".join(a.title() for a in ats)
            st.write(f"Searching {sum(c[0] in ats for c in boards.load_companies())} company career boards ({names})…")
            board_jobs, warns = boards.search_boards(terms, location, country, days, remote_only,
                                                     JOB_TYPES[job_type], board_max, ats, levels)
            st.write(f"{names}: {len(board_jobs)} matching jobs.")
            jobs = dedupe(board_jobs + jobs)  # same title at the same company once; longest description wins
            ss.search_warnings += warns
        if not jobs:
            ss.jobs = []
            status.update(label="No jobs found. Try broader titles, another location, or a longer date range.",
                          state="error")
            return
        for job in jobs:
            d = next((d for d in reversed(ss.directions)  # your rows win over the AI's on shared titles
                      if job["search_term"].lower() in (t.lower() for t in d["search_titles"])), None)
            job["direction"] = d["role"] if d else ""
            job["on_target"] = title_relevance(job["title"], d["search_titles"] if d else terms) >= 0.5
            job["keyword_score"] = keyword_score(ss.profile, job, d)
            job["level"], job["years_required"] = job_level(job)
        if set(levels) != set(LEVELS):  # filter before AI scoring, so the daily quota goes to the right jobs
            kept = [j for j in jobs if j["level"] in levels]
            st.write(f"Seniority filter ({', '.join(levels)}): kept {len(kept)} of {len(jobs)} jobs.")
            jobs = kept
            if not jobs:
                ss.jobs = []
                status.update(label="No jobs at the seniority levels you picked. Add 'Not stated' or "
                                    "another level.", state="error")
                return
        jobs.sort(key=lambda j: (j["on_target"], j["keyword_score"]), reverse=True)
        ss.jobs = jobs
        st.write(f"Found {len(jobs)} unique jobs. AI-scoring the top {min(top_n, len(jobs))} against their "
                 "full descriptions…")
        score_jobs([j for j in jobs if j["on_target"]][:top_n], model)
        status.update(label=f"Done: {len(jobs)} jobs found ({portal_counts(jobs)}), "
                            f"{sum(j['id'] in ss.matches for j in jobs)} AI-scored", state="complete", expanded=False)


@st.fragment
def job_tools(job: dict) -> None:
    """Per-job buttons; a fragment so clicking one doesn't redraw the whole page."""
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    actions = ((c1, "cover", "✉️ Write cover letter", cover_letter),
               (c2, "tailored", "📝 Tailor my resume", tailored_resume))
    for col, kind, label, fn in actions:
        if col.button(label, key=f"{kind}:{job['id']}"):
            with st.spinner("Writing…"):
                try:
                    ss.docs[(kind, job["id"])] = fn(ss.resume_text, job, ss.model)
                    if kind == "tailored":  # a new rewrite isn't saved yet: don't upload the old file
                        ss.docs.pop(("resume_file", job["id"]), None)
                        ss.docs.pop(("resume_text", job["id"]), None)
                except Exception as e:
                    st.error(llm.explain(e))
    ready = applicant.ready()
    if c3.button("🤖 Fill application", key=f"fill:{job['id']}", type="primary", disabled=not ready,
                 help="Opens the form in a browser window and fills it in. You review it and click Submit yourself."
                 if ready else "Set up 🤖 Auto-fill (above the job list) first: your answers + resume."):
        apply_url = job["apply_url"]
        if not apply_url:  # LinkedIn/Indeed hide the company link when logged out: try the company's own board
            with st.spinner("Looking for this job on the company's own career site…"):
                found = boards.find_on_boards(job["company"], job["title"])
            if found:
                apply_url = found["apply_url"]
                boards.learn_companies([{"apply_url": apply_url, "company": job["company"]}])
                st.success(f"Found it on {found['company']}'s {found['site']} board: filling that form.")
            else:
                st.info("This company's own career site wasn't found, and LinkedIn/Indeed don't allow automation, so "
                        "the job page opens unfilled. Sign in there once (the helper window remembers it), click "
                        "**Apply**, and on the company's site press **✨ Fill this page**.")
        autofill.queue_job({
            "title": job["title"], "company": job["company"], "url": safe_url(job["url"]),
            "apply_url": safe_url(apply_url), "description": job["description"][:6000],
            "cover_letter": ss.docs.get(("cover", job["id"]), ""), "model": ss.model,
            "resume_path": ss.docs.get(("resume_file", job["id"])),  # tailored resume, if saved for this job
            "resume_text": ss.docs.get(("resume_text", job["id"])),
        })
        st.toast("Opening the application in the auto-fill browser window…")
    if c4.button("🌐 Portfolio page", key=f"portfolio:{job['id']}",
                 help="A one-page website for this job. Every claim on it comes from my_data/portfolio_bank.json."):
        with st.spinner("Building your portfolio page for this job…"):
            try:
                ss.docs[("portfolio", job["id"])] = portfolio.build(
                    job, ss.docs.get(("resume_file", job["id"])), ss.model)
                ss.docs.pop(("portfolio_url", job["id"]), None)
            except Exception as e:
                st.error(llm.explain(e))
    if letter := ss.docs.get(("cover", job["id"])):
        st.markdown("**Cover letter** (copy button at top right)")
        st.code(letter, language=None, wrap_lines=True)
    if tailored := ss.docs.get(("tailored", job["id"])):
        tailored_resume_editor(job, tailored)
    if built := ss.docs.get(("portfolio", job["id"])):
        portfolio_panel(job, built)


def portfolio_panel(job: dict, built: dict) -> None:
    """This job's portfolio page: a preview, a download, and a shareable link once GitHub Pages is set up."""
    page = built["page"]
    st.markdown("**🌐 Portfolio page for this job.** Every claim on it is copied from your fact bank "
                "(`my_data/portfolio_bank.json`); the AI chose which ones fit this posting. "
                f"Saved in `{md_escape(built['folder'])}`.")
    if page["gaps"]:
        st.markdown("**Asks your fact bank can't prove yet** (left off the page)\n"
                    + "\n".join(f"- {md_escape(g)}" for g in page["gaps"]))
    # A data: URL gives the preview its own opaque origin, so nothing on the page can reach the app itself.
    st.iframe("data:text/html;base64," + base64.b64encode(built["html"].encode("utf-8")).decode("ascii"), height=760)
    c1, c2 = st.columns([1, 2])
    c1.download_button("⬇️ Download page (.html)", built["html"].encode("utf-8"), file_name="index.html",
                       mime="text/html", key=f"dl_portfolio:{job['id']}")
    if not portfolio.publish_settings():
        c2.caption("For a shareable link, set up GitHub Pages once: see README, *Portfolio links*.")
        return
    if c2.button("🔗 Publish link", key=f"publish_portfolio:{job['id']}", type="primary"):
        with st.spinner("Publishing to GitHub Pages…"):
            try:
                ss.docs[("portfolio_url", job["id"])] = portfolio.publish(built["folder"])
            except Exception as e:
                st.error(str(e))
    if url := ss.docs.get(("portfolio_url", job["id"])):
        st.success("Published. GitHub Pages needs about a minute to update, then this link works:")
        st.code(url, language=None)


def tailored_resume_editor(job: dict, tailored: dict) -> None:
    """The AI's rewrite, editable. Saving it makes 🤖 Fill application upload this version for this job."""
    st.markdown("**📝 Tailored resume for this job.** Edit anything, then save it. 🤖 Fill application will "
                "upload this version for this job; your other applications keep your normal resume.")
    if tailored["changes"]:
        st.markdown("**What changed**\n" + "\n".join(f"- {md_escape(c)}" for c in tailored["changes"]))
    # A new rewrite gets a new key, so the box shows it instead of keeping the old edits.
    text = st.text_area("Tailored resume (line 1 = name, CAPITAL lines = headings, '- ' = bullets)",
                        tailored["resume"], height=480, key=f"tailored_text:{job['id']}:{hash(tailored['resume'])}")
    if tailored["gaps"]:
        st.markdown("**Gaps to close** (advice only, not in the resume)\n"
                    + "\n".join(f"- {md_escape(g)}" for g in tailored["gaps"]))
    if st.button("💾 Save as this job's resume", key=f"save_tailored:{job['id']}", type="primary"):
        if placeholders := re.findall(r"\[[^\]\n]{1,60}\]", text):
            st.warning("Fill in or delete these placeholders first, or they go to the employer as written: "
                       + ", ".join(md_escape(p) for p in dict.fromkeys(placeholders)))
        else:
            ss.docs[("resume_file", job["id"])] = str(applicant.save_tailored_resume(text, job))
            ss.docs[("resume_text", job["id"])] = text
    if path := ss.docs.get(("resume_file", job["id"])):
        st.success(f"Saved as {Path(path).name}: 🤖 Fill application uploads it for this job.")
        st.download_button("⬇️ Download .docx", Path(path).read_bytes(), file_name=Path(path).name,
                           key=f"dl_tailored:{job['id']}",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


def render_job(job: dict, m: dict) -> None:
    with st.container(border=True):
        head, score = st.columns([5, 1])
        head.markdown(f"#### {md_escape(job['title'])}")
        meta = [job["company"], job["location"], job["site"], level_text(job),
                f"posted {job['posted']}" if job["posted"] else "", job["salary"]]
        head.caption(" · ".join(md_escape(x) for x in meta if x))
        color = "green" if m["score"] >= 70 else "orange" if m["score"] >= 50 else "red"
        with head:
            st.badge(m["verdict"], color=color)
            if d := direction_of(job):
                st.badge(f"{d['role']} · {d['caution']}", color=CAUTION_COLORS.get(d["caution"], "gray"))
        score.metric("AI match", f"{m['score']}%")

        if m.get("context_fit"):
            icon = {"fits": "✅", "unclear": "❔", "outside": "⛔"}.get(m["context_fit"], "")
            st.markdown(f"**Company context:** {icon} {m['context_fit']} · stage: {md_escape(m['company_stage'])}"
                        f" · {md_escape(m['context_evidence'])}")
        st.markdown(untrusted_md(m["summary"]))
        have, missing = st.columns(2)
        have.markdown("**✅ You have:** " + (", ".join(md_escape(s) for s in m["matching_skills"]) or "–"))
        missing.markdown("**❌ Missing:** " + (", ".join(md_escape(s) for s in m["missing_skills"]) or "nothing major"))
        if m["red_flags"]:
            st.warning("**Possible blockers:** " + "; ".join(md_escape(x) for x in m["red_flags"]), icon="⚠️")

        b1, b2, _ = st.columns([1, 1, 2])
        url, apply_url = safe_url(job["url"]), safe_url(job["apply_url"])
        if url:
            b1.link_button(f"Open on {job['site']}", url)
        if apply_url and apply_url != url:
            b2.link_button("Apply on company site", apply_url)
        with st.expander("Full job description"):
            st.markdown(untrusted_md(job["description"]) or "_No description provided._")
        job_tools(job)


EASY_FORMS = ("greenhouse.io", "lever.co", "ashbyhq.com")


def autofill_ease(job: dict) -> str:
    """How well 🤖 Fill application can handle this job's form."""
    url = job.get("apply_url") or ""
    if any(h in url for h in EASY_FORMS):
        return "✅ easy"
    if not url:
        return "🚫 on portal"  # LinkedIn Easy Apply / Indeed Apply: automation not allowed there
    return "◐ partial"


def applicant_setup(upload) -> None:
    """Your saved answers for 🤖 Fill application: contact details, visa status, EEO choices."""
    ready = applicant.ready()
    title = "🤖 Auto-fill setup: ready ✅" if ready else "🤖 Auto-fill setup: fill this in once to use 'Fill application'"
    with st.expander(title):
        st.caption("Saved in my_data/ on this computer only. These answers go into every application, so visa and "
                   "sponsorship questions get the same answer everywhere. The AI only writes answers to open "
                   "questions, from your resume and these facts.")
        a = applicant.load()
        with st.form("applicant_form", border=False):
            cols = st.columns(3)
            values = {}
            for i, (key, label, kind, _) in enumerate(applicant.FIELDS):
                if kind == "area":
                    values[key] = st.text_area(label, a[key], height=90)
                elif kind == "yesno" or kind.startswith("choice:"):
                    opts = applicant.YES_NO if kind == "yesno" else kind.split(":", 1)[1].split("|")
                    values[key] = cols[i % 3].selectbox(label, opts, index=opts.index(a[key]) if a[key] in opts else 0)
                else:
                    values[key] = cols[i % 3].text_input(label, a[key])
            saved = st.form_submit_button("💾 Save answers", type="primary")
        if saved:
            keep = upload is not None and upload.name.lower().endswith((".pdf", ".docx"))
            applicant.save(values, upload.name if keep else None, upload.getvalue() if keep else None,
                           ss.resume_text)
            missing = [label for key, label, _, _ in applicant.FIELDS if key in applicant.REQUIRED and not values[key]]
            if missing:
                st.warning("Saved, but still needed: " + ", ".join(missing))
            elif applicant.resume_file() is None:
                st.warning("Saved. Upload your resume as PDF or DOCX (not TXT) so forms can attach it.")
            else:
                st.success("Saved. 🤖 Fill application now works on every job card.")


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("1. Your resume")
    upload = st.file_uploader("PDF, DOCX or TXT (max 5 MB)", type=["pdf", "docx", "txt"])
    if upload is not None:
        load_resume(upload)

    st.header("2. Search")
    terms_text = st.text_area("Job titles to search (one per line, max 6)", key="terms", height=120,
                              help="Filled in from your resume. Edit freely.")
    location = st.text_input("Location", key="location", placeholder="e.g. Austin, TX · Bangalore · blank = anywhere")
    country = st.selectbox("Country", COUNTRIES, key="country")
    board_counts = Counter(ats for ats, _, _ in boards.load_companies())
    portals = st.multiselect(
        "Job portals", [*SITES, *BOARD_PORTALS], default=[*SITES, *BOARD_PORTALS],
        format_func=lambda s: f"{s} ({board_counts[BOARD_PORTALS[s]]} companies)" if s in BOARD_PORTALS else s,
        help="Greenhouse, Lever and Ashby are companies' own career boards (listed in companies.txt): full "
             "descriptions, no blocking, and the easiest forms to auto-fill. Searches add new companies they find "
             "in Indeed/LinkedIn apply links.")
    sites = [s for s in portals if s in SITES]
    ats = [BOARD_PORTALS[s] for s in portals if s in BOARD_PORTALS]
    c1, c2 = st.columns(2)
    days = c1.selectbox("Posted within", [1, 3, 7, 14, 30, 0], index=3,
                        format_func=lambda d: "Any time" if d == 0 else f"{d} day{'s' if d > 1 else ''}")
    job_type = c2.selectbox("Job type", list(JOB_TYPES))
    remote_only = st.checkbox("Remote only")
    levels = st.multiselect(
        "Seniority", LEVELS, key="levels",
        help="Read from the job title (Associate, Junior, I/II, Senior, Staff, Lead, Head of...). When the title "
             "has no level word, from the years of experience the posting asks for (0-2 = Entry/Junior, 3-4 = Mid, "
             "5-7 = Senior, 8+ = Lead/Staff+). 'Not stated' = neither; keep it ticked, many good jobs are there.")
    per_site = st.slider("Results per portal, per title", 5, 50, 15, step=5)
    board_max = st.slider("Max jobs from Greenhouse/Lever/Ashby", 50, 1000, 300, step=50, disabled=not ats)

    st.header("3. AI")
    model = st.selectbox("Groq model", model_options(), key="model")
    pool = llm.scoring_pool(model, model_options())
    st.checkbox(f"Score on all {len(pool)} models at once (~{len(pool)}× faster)", value=True, key="parallel",
                disabled=len(pool) < 2, help="Groq's free limits are per model, so each extra model adds its own "
                                             "8K tokens/min and 1K requests/day: " + ", ".join(pool))
    top_n = st.slider("AI-score the top N jobs", 1, 200, 40,
                      help="Every job is pre-ranked for free by keyword overlap with your resume; only the best N "
                           "use the LLM. Each model scores roughly 4 jobs a minute on the free tier.")
    go = st.button("🔎 Find matching jobs", type="primary", width="stretch", disabled=ss.profile is None)

    st.header("4. Portfolio")
    if st.button("🌐 Update my general portfolio", width="stretch",
                 help="Your all-purpose portfolio page (no job named), built from my_data/portfolio_bank.json. "
                      "With GitHub Pages set up it goes to your main portfolio link."):
        try:
            general = portfolio.build_general()
            ss.general_url = (portfolio.publish(general["folder"], at_root=True) if portfolio.publish_settings()
                              else "")
            if not ss.general_url:
                st.info(f"Saved in {general['folder']}. Set up GitHub Pages (README) to get a link.")
        except Exception as e:
            st.error(llm.explain(e))
    if ss.get("general_url"):
        st.caption(f"Live in about a minute: {ss.general_url}")

# ------------------------------------------------------------------- main ---
st.title("💼 Smart Job Assistant")

try:
    llm.api_key()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if upload is None:
    st.info("👈 Upload your resume to begin.")
    st.markdown(
        "**How it works**\n"
        "1. The AI reads your resume and suggests the job titles you should search for.\n"
        "2. It searches **Indeed**, **LinkedIn** and hundreds of **company career boards** (Greenhouse, Lever, "
        "Ashby) and pulls each job's full description.\n"
        "3. Every job is pre-ranked by skill overlap, then the best ones are scored by the AI against the full "
        "posting: fit score, matching and missing skills, and blockers such as clearance or visa requirements.\n"
        "4. For any job, write a tailored cover letter, or a tailored resume you edit and save for that job.\n"
        "5. **🤖 Fill application** opens the form in a browser and fills it from your saved answers. You check it "
        "and click Submit yourself."
    )
    st.caption("Your resume is processed in memory and sent to Groq for analysis. Nothing is saved to disk unless "
               "you save your auto-fill answers; those stay in my_data/ on this computer.")
    st.stop()

if ss.resume_error:
    st.error(ss.resume_error)
    st.stop()

if ss.profile_error:
    st.error(ss.profile_error)
    if st.button("Try again"):
        ss.profile_error = None
        st.rerun()
    st.stop()

if ss.profile is None:
    st.stop()

p = ss.profile
with st.expander(f"🧠 What the AI read from your resume: {md_escape(p['headline'])}", expanded=not ss.jobs):
    c1, c2, c3 = st.columns(3)
    c1.metric("Seniority", p["seniority"].title())
    c2.metric("Experience", f"{p['years_experience']:g} years")
    c3.metric("Skills found", len(p["skills"]))
    st.markdown("**Suggested job titles:** " + " · ".join(md_escape(t) for t in p["target_titles"]))
    st.markdown("**Skills:** " + ", ".join(md_escape(s) for s in p["skills"]))
    if p["experience"]:
        st.markdown("**Experience**\n" + "\n".join(f"- {md_escape(x)}" for x in p["experience"]))
    if p["education"]:
        st.markdown(f"**Education:** {md_escape(p['education'])}")
    st.caption("If something is wrong here, fix it in your resume and re-upload: every score builds on this profile.")

st.subheader("🧭 Your career directions")
st.caption("Where you're credible, and where to be careful. Edit any cell (lists are separated by ;), untick "
           "Search to leave a direction out, or add one in your own words. Your wording beats the AI's, and "
           "directions you add are remembered for every later resume upload (my_data/my_directions.json).")
if ss.directions_error:
    st.warning(ss.directions_error)
if ss.dir_base is not None:
    edited = st.data_editor(
        ss.dir_base, key=f"dir_editor_{ss.dir_version}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "use": st.column_config.CheckboxColumn("Search", default=True, width="small"),
            "role": st.column_config.TextColumn("Direction", required=True),
            "caution": st.column_config.SelectboxColumn("Caution", options=list(CAUTION_COLORS),
                                                        default="selective", required=True),
            "fits_where": st.column_config.TextColumn("Fits where", width="large"),
            "avoid_where": st.column_config.TextColumn("Avoid where", width="large"),
            "proof": st.column_config.TextColumn("Your proof"),
            "search_titles": st.column_config.TextColumn("Search titles"),
            "good_signals": st.column_config.TextColumn("Good signals"),
            "bad_signals": st.column_config.TextColumn("Bad signals"),
        },
    )
    ss.directions = df_to_directions(edited)
    for d in ss.directions:
        st.badge(f"{d['caution']}: {d['role']}", color=CAUTION_COLORS.get(d["caution"], "gray"))
c1, c2 = st.columns([3, 1])
c1.text_area("Add a direction in your own words", key="own_direction", height=90,
             placeholder="e.g. AI Product Manager - only at seed-stage startups, not established companies - "
                         "... This is the one to be most careful applying to.")
c2.button("➕ Add direction", on_click=add_own_direction, width="stretch")
c2.button("↻ Use for search", on_click=use_directions_for_search, width="stretch",
          help="Copy the ticked directions' search titles into the sidebar")

applicant_setup(upload)

if go:
    terms = [t.strip() for t in terms_text.splitlines() if t.strip()][:6]
    if not terms:
        st.error("Enter at least one job title to search.")
    elif not portals:
        st.error("Pick at least one job portal.")
    else:
        run_search(terms, location.strip(), country, sites, per_site, days, job_type, remote_only, top_n, model,
                   ats, board_max, levels or LEVELS)

for w in ss.search_warnings:
    st.warning(md_escape(w))

if not ss.jobs:
    st.info("Check the search settings in the sidebar, then click **Find matching jobs**.")
    st.stop()

jobs = ss.jobs
scored = sorted(((j, ss.matches[j["id"]]) for j in jobs if j["id"] in ss.matches),
                key=lambda jm: jm[1]["score"], reverse=True)
unscored = [j for j in jobs if j["id"] not in ss.matches and j.get("on_target", True)]
unscored += [j for j in jobs if j["id"] not in ss.matches and not j.get("on_target", True)]  # on-target first
off_target = sum(not j.get("on_target", True) for j in jobs)
if off_target:
    st.caption(f"{off_target} jobs had titles that don't match your directions, so they're scored last "
               "(they're all in the table).")

m1, m2, m3 = st.columns(3)
m1.metric("Jobs found", len(jobs))
m2.metric("AI-scored", len(scored))
m3.metric("Good or strong matches (70+)", sum(m["score"] >= 70 for _, m in scored))
st.caption(f"By portal: {portal_counts(jobs)}")

b1, b2, _ = st.columns([1, 1, 2])
if unscored and b1.button(f"AI-score the next {min(25, len(unscored))} jobs"):
    score_jobs(unscored[:25], model)
    st.rerun()
if len(unscored) > 25 and b2.button(f"AI-score all {len(unscored)} remaining",
                                    help="Long runs are fine: each model retries by itself when it hits its "
                                         "per-minute limit. About 1K jobs per model per day on the free tier."):
    score_jobs(unscored, model)
    st.rerun()

st.subheader("All jobs")
table = pd.DataFrame([
    {
        "AI score": ss.matches[j["id"]]["score"] if j["id"] in ss.matches else None,
        "Verdict": ss.matches[j["id"]]["verdict"] if j["id"] in ss.matches else "not AI-scored",
        "Keyword fit": j["keyword_score"],
        "Level": level_text(j),
        "Title": j["title"],
        "Company": j["company"],
        "Location": j["location"],
        "Portal": j["site"],
        "Posted": j["posted"],
        "Salary": j["salary"],
        "Direction": j.get("direction", ""),
        "Context": ss.matches[j["id"]].get("context_fit", "") if j["id"] in ss.matches else "",
        "Auto-fill": autofill_ease(j),
        "Link": safe_url(j["url"]),
    }
    for j in jobs
]).sort_values(["AI score", "Keyword fit"], ascending=False, na_position="last")
st.dataframe(
    table, hide_index=True,
    column_config={
        "AI score": st.column_config.ProgressColumn("AI score", min_value=0, max_value=100, format="%d"),
        "Keyword fit": st.column_config.ProgressColumn("Keyword fit", min_value=0, max_value=100, format="%d"),
        "Link": st.column_config.LinkColumn("Link", display_text="Open ↗"),
    },
)
st.download_button("⬇️ Download as CSV", table.map(csv_safe).to_csv(index=False).encode("utf-8"),
                   file_name="job_matches.csv", mime="text/csv")

st.subheader("Best matches")
min_score = st.slider("Show AI matches scoring at least", 0, 100, 40, step=5)
hide_outside = st.checkbox("Hide jobs outside my conditions", value=False,
                           help=f"Jobs outside a direction's 'fits where' already lose {OUTSIDE_PENALTY} points and "
                                "carry a warning; tick this to hide them completely.")
found_sites = sorted({j["site"] for j, _ in scored})
show_sites = st.multiselect("Portals to show", found_sites, default=found_sites)
shown = [(j, m) for j, m in scored
         if m["score"] >= min_score and not (hide_outside and m.get("context_fit") == "outside")
         and j["site"] in show_sites]
if not shown:
    st.info("No AI-scored jobs at this level. Lower the slider, or AI-score more jobs.")
for job, m in shown:
    render_job(job, m)

st.caption("Built with Groq + python-jobspy + Streamlit. Always check a posting on the portal before applying.")
