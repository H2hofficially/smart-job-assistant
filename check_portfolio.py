r"""Acceptance checks for the 🌐 Portfolio page feature (portfolio.py + the app button + no-placeholder tailoring).

    venv\Scripts\python check_portfolio.py          # offline checks only
    venv\Scripts\python check_portfolio.py live     # + one real Groq call: builds the Hello There Collective page

Do NOT edit a check to make it pass; fix the code instead.
"""
import base64
import html as html_lib
import json
import os
import re
import sys
import types
from pathlib import Path

import portfolio
import tailor

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))


def visible_text(page: str) -> str:
    page = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S)
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", page))


JOB = {"title": "AI & Automation Specialist", "company": "Hello There Collective",
       "description": "Build AI agents and automations with Zapier, Make or n8n. Connect Monday.com and Slack."}

bank = portfolio.load_bank()
facts = {f["id"]: f["text"] for f in bank["facts"]}
pids = [p["id"] for p in bank["projects"]]

# 1. The fact bank
check("bank: fact ids are unique", len(facts) == len(bank["facts"]))
check("bank: project ids are unique", len(set(pids)) == len(pids))
check("bank: every project has a flow and points", all(p.get("flow") and p.get("points") for p in bank["projects"]))
def strings(x):
    if isinstance(x, dict):
        return [s for v in x.values() for s in strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in strings(v)]
    return [x] if isinstance(x, str) else []


check("bank: no placeholders in any text", not any(re.search(r"\[[^\]]*\]", s) for s in strings(bank)))

# 2. clean_plan keeps only what the bank supports
raw = {
    "role_title": "AI & Automation Specialist",
    "headline": "I built Monday.com automations for 40 agencies.",
    "lede": "I build AI agents that run every day. [Add a metric here]",
    "matches": [
        {"ask": "Build AI agents and automations", "fact_ids": ["content_agent", "made_up_fact"]},
        {"ask": "Build AI agents and automations", "fact_ids": ["approval_gate"]},
        {"ask": "Monitoring and error handling", "fact_ids": ["monitoring", "rollback", "plans_tests"]},
        {"ask": "Connect platforms with APIs", "fact_ids": ["integrations"]},
        {"ask": "Monday.com expert", "fact_ids": ["invented_monday_fact"]},
        {"ask": "Creative AI tools", "fact_ids": ["media_lab"]},
    ],
    "project_ids": ["content_engine", "not_a_project", "darkoapp"],
    "plan": [{"title": "Understand", "items": ["Map each team's manual steps."]},
             {"title": "Build", "items": ["Turn meeting transcripts into Monday.com items."]},
             {"title": "Scale", "items": ["Measure hours saved per workflow."]}],
    "gaps": ["Monday.com experience"],
}
page = portfolio.clean_plan(raw, bank, JOB)
check("clean: an invented tool and number in the headline fall back to the bank headline",
      page["headline"] == bank["headline"], page["headline"])
check("clean: a bracket placeholder in the lede falls back to the bank lede", page["lede"] == bank["lede"], page["lede"])
check("clean: unknown fact ids are dropped", all(i in facts for m in page["matches"] for i in m["fact_ids"]))
check("clean: a row with no real fact is dropped", all(m["ask"] != "Monday.com expert" for m in page["matches"]))
check("clean: duplicate asks are dropped", len({m["ask"] for m in page["matches"]}) == len(page["matches"]))
check("clean: evidence is the bank's own text, at most 2 facts per row",
      all(m["evidence"] == " ".join(facts[i] for i in m["fact_ids"]) and len(m["fact_ids"]) <= 2
          for m in page["matches"]))
check("clean: unknown projects are dropped, order kept", page["project_ids"] == ["content_engine", "darkoapp"],
      str(page["project_ids"]))
check("clean: the 30/60/90 plan keeps its 3 phases", [p["days"] for p in page["plan"]] == [d for d, _ in portfolio.PHASES])
good = dict(raw, headline="I build AI agents and automations for Hello There Collective.",
            lede="I run a content engine with health checks every 15 minutes. Every post waits for approval.")
page_ok = portfolio.clean_plan(good, bank, JOB)
check("clean: a headline built from bank words and the company name is kept", page_ok["headline"] == good["headline"],
      page_ok["headline"])
check("clean: a lede whose number is in the bank is kept", page_ok["lede"] == good["lede"], page_ok["lede"])
try:
    portfolio.clean_plan(dict(raw, matches=raw["matches"][:2]), bank, JOB)
    check("clean: fewer than 3 provable asks is refused", False, "no error raised")
except RuntimeError:
    check("clean: fewer than 3 provable asks is refused", True)
half = portfolio.clean_plan(dict(raw, plan=raw["plan"][:2]), bank, JOB)
check("clean: a half plan is dropped instead of shown", half["plan"] == [])

# 3. The page
person = portfolio.person_info(bank)
evil = dict(page, role_title='<script>alert(1)</script>')
out = portfolio.render_html(evil, bank, person, JOB, "Hari_Prasath_Sundaresan_Resume.pdf")
text = visible_text(out)
check("page: shows the candidate's name", "Hari Prasath Sundaresan" in text, person["name"])
check("page: no bracket placeholders in the visible text", not re.search(r"\[[^\]]*\]", text),
      str(re.findall(r"\[[^\]]*\]", text)[:3]))
check("page: posting text is escaped (no script injection)", "<script>alert(1)" not in out)
check("page: asks search engines not to index it", 'name="robots" content="noindex' in out)
check("page: links are http(s) or the resume file only",
      all(h.startswith(("https://", "http://", "#")) or h == "Hari_Prasath_Sundaresan_Resume.pdf"
          for h in re.findall(r'href="([^"]+)"', out) if "fonts.g" not in h))
check("page: every match row's evidence appears verbatim", all(html_lib.escape(m["evidence"]) in out for m in page["matches"]))
check("page: names the company in the plan heading", "How I would start at Hello There Collective" in text)
bad_bank = json.loads(json.dumps(bank))
bad_bank["projects"][0]["links"] = [{"label": "evil", "url": "javascript:alert(1)"}]
check("page: javascript: links in the bank are skipped",
      "javascript:" not in portfolio.render_html(page, bad_bank, person, JOB, ""))
check("page: works without a resume file", 'Download my resume' not in portfolio.render_html(page, bank, person, JOB, ""))
check("slug: safe folder name", portfolio.slug(JOB) == "hello-there-collective-ai-automation-specialist",
      portfolio.slug(JOB))
general = portfolio.render_html(portfolio.general_page(bank), bank, person, None, "Me_Resume.pdf")
gtext = visible_text(general)
check("general page: no job named, no role match, no 30/60/90 plan",
      "portfolio for" not in gtext and "Matched to your role" not in gtext and "How I would start" not in gtext)
check("general page: shows every project", all(html_lib.escape(p["title"]) in general for p in bank["projects"]))
check("general page: titled with the name", f"<title>{person['name']}</title>" in general)
check("general page: no placeholders", not re.search(r"\[[^\]]*\]", gtext))

# 4. Publishing (GitHub API faked: nothing leaves this computer)
calls = []


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body, self.text = code, body, json.dumps(body)

    def json(self):
        return self._body


tmp = Path(portfolio.OUT_DIR) / "_check_publish"
tmp.mkdir(parents=True, exist_ok=True)
(tmp / "index.html").write_text("<p>hi</p>", encoding="utf-8")
(tmp / "Me_Resume.pdf").write_bytes(b"%PDF")
fake = types.SimpleNamespace(
    get=lambda url, **kw: calls.append(("GET", url, kw)) or Resp(200, [
        {"name": "index.html", "sha": "abc", "type": "file"}, {"name": "old.docx", "sha": "def", "type": "file"}]),
    put=lambda url, **kw: calls.append(("PUT", url, kw)) or Resp(200, {}),
    delete=lambda url, **kw: calls.append(("DELETE", url, kw)) or Resp(200, {}),
    Response=object)
real_requests, saved_env = portfolio.requests, dict(os.environ)
portfolio.requests = fake
portfolio.load_dotenv = lambda **kw: None  # keep the real .env out of this check
real_gh = portfolio._gh_token
try:
    for k in ("PORTFOLIO_GITHUB_TOKEN", "PORTFOLIO_REPO"):
        os.environ.pop(k, None)
    check("publish: off until .env names a repo", portfolio.publish_settings() is None)
    os.environ["PORTFOLIO_REPO"] = "H2hofficially/portfolio"
    portfolio._gh_token = lambda: ""
    check("publish: off when there is no token and no GitHub CLI login", portfolio.publish_settings() is None)
    portfolio._gh_token = lambda: "gh-login"
    check("publish: falls back to the GitHub CLI login", (portfolio.publish_settings() or {}).get("token") == "gh-login")
    os.environ["PORTFOLIO_GITHUB_TOKEN"] = "test-token"
    check("publish: a token in .env wins over the CLI login", portfolio.publish_settings()["token"] == "test-token")
    url = portfolio.publish(tmp)
    check("publish: returns the GitHub Pages link", url == "https://h2hofficially.github.io/portfolio/_check_publish/", url)
    puts = [c for c in calls if c[0] == "PUT"]
    check("publish: uploads every file in the folder", sorted(c[1].rsplit("/", 1)[1] for c in puts)
          == ["Me_Resume.pdf", "index.html"])
    check("publish: updates an existing file with its sha",
          any(c[1].endswith("/index.html") and c[2]["json"].get("sha") == "abc" for c in puts))
    check("publish: file content is base64 of the file",
          any(base64.b64decode(c[2]["json"]["content"]) == b"<p>hi</p>" for c in puts))
    check("publish: removes files left from an older build", any(c[0] == "DELETE" and c[1].endswith("/old.docx")
                                                                for c in calls))
    check("publish: token sent as a header, never in the URL",
          all("test-token" not in c[1] and c[2]["headers"]["Authorization"] == "Bearer test-token" for c in calls))
    calls.clear()
    fake.get = lambda url, **kw: calls.append(("GET", url, kw)) or Resp(200, [
        {"name": "README.md", "sha": "r1", "type": "file"}, {"name": ".nojekyll", "sha": "n1", "type": "file"},
        {"name": "index.html", "sha": "i1", "type": "file"}, {"name": "Old_Resume.pdf", "sha": "o1", "type": "file"},
        {"name": "hello-there-collective-ai-automation-specialist", "sha": "d1", "type": "dir"}])
    root_url = portfolio.publish(tmp, at_root=True)
    check("publish root: returns the main link", root_url == "https://h2hofficially.github.io/portfolio/", root_url)
    check("publish root: files go to the repo root",
          sorted(c[1].rsplit("/contents/", 1)[1] for c in calls if c[0] == "PUT") == ["Me_Resume.pdf", "index.html"])
    deleted = [c[1].rsplit("/", 1)[1] for c in calls if c[0] == "DELETE"]
    check("publish root: never deletes the README, .nojekyll or job folders, only an old resume copy",
          deleted == ["Old_Resume.pdf"], str(deleted))
    fake.get = lambda url, **kw: Resp(401, {"message": "Bad credentials"})
    try:
        portfolio.publish(tmp)
        check("publish: a bad token gives a clear message", False, "no error raised")
    except RuntimeError as e:
        check("publish: a bad token gives a clear message", "PORTFOLIO_GITHUB_TOKEN" in str(e), str(e))
finally:
    portfolio.requests = real_requests
    portfolio._gh_token = real_gh
    os.environ.clear()
    os.environ.update(saved_env)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()

# 5. No placeholders from the cover letter and resume tailoring either
prompts = tailor._SYSTEM + str(tailor.COVER_PROMPT) + str(tailor.TAILOR_PROMPT)
check("tailor: no instruction to write [placeholders]", not re.search(r"\[(X%|Your Name|Hiring Manager|placeholder)",
                                                                      prompts))
check("tailor: the cover letter is signed with the saved name", "sign_off" in tailor.COVER_PROMPT.input_variables
      and "Hari Prasath Sundaresan" in tailor._sign_off(), tailor._sign_off())

# 6. The app
app = Path("app.py").read_text(encoding="utf-8")
check("app: 🌐 Portfolio page button calls portfolio.build", "🌐 Portfolio page" in app and "portfolio.build(" in app)
check("app: publish button only when GitHub Pages is set up", "portfolio.publish_settings()" in app)
check("app: general portfolio button publishes at the root", "portfolio.build_general()" in app
      and "at_root=True" in app)

# 7. Saved career directions (your own directions survive a new resume upload)
from matcher import clean_direction  # noqa: E402

ns = {"json": json, "clean_direction": clean_direction,
      "applicant": types.SimpleNamespace(MY_DATA=Path(portfolio.OUT_DIR) / "_check_dirs")}
exec(app[app.index("MY_DIRECTIONS = "):app.index("def add_own_direction")], ns)
d0 = {"role": "Creative Technologist", "caution": "go", "fits_where": "agencies", "avoid_where": "game engines",
      "proof": ["VFX company"], "search_titles": ["Creative Technologist"], "good_signals": ["comfyui"],
      "bad_signals": ["unreal engine"]}
try:
    check("directions: nothing saved yet -> none", ns["saved_directions"]() == [])
    ns["remember_direction"](d0)
    ns["remember_direction"](dict(d0, caution="selective"))
    saved = ns["saved_directions"]()
    check("directions: remembered once per role, latest wins", len(saved) == 1 and saved[0]["caution"] == "selective")
    draft = [dict(d0, role="AI Automation Engineer"), dict(d0, caution="careful")]
    merged = ns["with_saved"](draft)
    check("directions: yours replace the AI's row of the same name and sit last",
          [d["role"] for d in merged] == ["AI Automation Engineer", "Creative Technologist"]
          and merged[-1]["caution"] == "selective" and merged[-1]["use"] is True)
finally:
    shutil_dir = ns["applicant"].MY_DATA
    for f in shutil_dir.glob("*"):
        f.unlink()
    if shutil_dir.exists():
        shutil_dir.rmdir()
real_dirs = Path(portfolio.applicant.MY_DATA) / "my_directions.json"
check("directions: your Creative Technologist direction is saved",
      any(d.get("role") == "Creative Technologist" for d in json.loads(real_dirs.read_text(encoding="utf-8")))
      if real_dirs.exists() else False)

# 8. Live: one real Groq call, builds the Hello There Collective page
if "live" in sys.argv[1:]:
    posting = Path(__file__).with_name("my_data") / "portfolio_test_job.txt"
    job = {"title": "AI & Automation Specialist", "company": "Hello There Collective",
           "description": posting.read_text(encoding="utf-8")}
    built = portfolio.build(job)
    live, out = built["page"], built["html"]
    check("live: 3-7 provable asks", 3 <= len(live["matches"]) <= 7, str(len(live["matches"])))
    check("live: evidence is bank text only", all(m["evidence"] == " ".join(facts[i] for i in m["fact_ids"])
                                                 for m in live["matches"]))
    check("live: no placeholders on the page", not re.search(r"\[[^\]]*\]", visible_text(out)))
    check("live: headline and lede pass the invented-claim guard",
          not portfolio.unsupported(live["headline"] + " " + live["lede"],
                                    json.dumps(bank, ensure_ascii=False) + " " + job["company"] + " " + job["title"]))
    check("live: page and resume copy saved", (Path(built["folder"]) / "index.html").exists()
          and any(p.name.endswith("_Resume.pdf") or p.name.endswith("_Resume.docx") for p in Path(built["folder"]).iterdir()))
    print("\nLIVE PAGE:", built["folder"])
    print("headline:", live["headline"])
    print("asks:", " | ".join(m["ask"] for m in live["matches"]))
    print("projects:", ", ".join(live["project_ids"]))
    print("gaps:", " | ".join(live["gaps"]) or "none")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
