# Smart Job Assistant — Design notes

A Streamlit app for job hunting. It reads your resume, suggests the job titles
you should search for, pulls real postings with full descriptions from Indeed,
LinkedIn and ~300 company career boards (Greenhouse, Lever, Ashby), ranks them
against your resume, writes tailored cover letters and resume tweaks, and
auto-fills application forms in a browser for you to review and submit.
The AI runs on Groq.

## 1. How it works

```
resume (PDF/DOCX/TXT, parsed in memory)
  -> AI reads it into a profile: seniority, years, skills, roles, 3-5 target titles
  -> search Indeed + LinkedIn for those titles (python-jobspy, full descriptions)
  -> search the company boards in companies.txt (public Greenhouse/Lever/Ashby APIs)
  -> de-duplicate (same title + company) and apply remote / job-type filters
  -> free local pre-rank: skill + title keyword overlap for every job
  -> AI-score the top N against the full posting, one thread per Groq model:
       0-100 score, verdict, matching skills, missing skills, hard blockers
  -> results table + CSV export + ranked job cards
  -> per job, on click: cover letter, resume tweaks, Fill application
       (autofill.py: visible browser, saved answers + AI answers, you click Submit)
       and Portfolio page (portfolio.py: fact bank -> AI picks the proof -> checked -> HTML -> GitHub Pages link)
```

## 2. Files

```
job_assistant/
├── app.py               Streamlit UI and session state
├── scraper.py           Indeed + LinkedIn search via python-jobspy; cleaning, dedupe, trimming
├── boards.py            company career boards (Greenhouse/Lever/Ashby public APIs), location filters
├── companies.txt        the boards to search (ats:slug  Name); searches append new ones
├── autofill.py          browser helper: fills application forms, never submits
├── applicant.py         your saved application answers + resume copy (my_data/)
├── matcher.py           resume -> profile (LLM), keyword pre-rank, job-fit scoring (LLM)
├── tailor.py            cover letter + resume tweaks (LLM)
├── portfolio.py         one-page portfolio website per job (LLM picks proof from your fact bank; GitHub Pages)
├── check_portfolio.py   acceptance checks for portfolio pages (offline; `live` adds one Groq call)
├── llm.py               Groq model factory, structured output, model list, error messages
├── utils.py             in-memory resume parsing, safe markdown/URL/CSV helpers
├── requirements.txt     pinned versions (see the 2-step install)
├── .env.example         template: copy to .env and add your key
├── .env                 GROQ_API_KEY (never commit or share)
├── .gitignore           keeps .env, venv, resumes, exports and my_data/ out of git
├── my_data/             created on first Save: applicant.json, resume copy, browser profile, logs
└── .streamlit/config.toml   localhost-only, 5 MB upload cap, no telemetry
```

## 3. Environment

- Python 3.14.3 (3.10+ works). Windows / PowerShell.
- Searching needs no browser. Auto-fill uses Playwright (1.63, in requirements.txt) with your installed
  Chrome, falling back to Playwright's Chromium (`venv\Scripts\python -m playwright install chromium`).

## 4. Setup

```powershell
cd smart-job-assistant
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pip install --no-deps python-jobspy==1.1.82
copy .env.example .env      # then paste your Groq key into .env
venv\Scripts\streamlit run app.py
```

`python-jobspy` pins numpy 1.26 and pandas 2. Those can't be installed on
Python 3.14 (there's no prebuilt numpy, and building it needs a C compiler),
so it is installed with `--no-deps` on top of the versions in
requirements.txt. That combination was tested live.

## 5. .env

```
GROQ_API_KEY=gsk_your_key_here
# optional: GROQ_MODEL=openai/gpt-oss-120b
```

Get a key at https://console.groq.com/keys. The app re-reads `.env` on every
AI call, so a new key works without restarting.

## 6. Key decisions

- **Job source: python-jobspy (Indeed + LinkedIn).** The old Bing Jobs scraper
  had no job descriptions, so the "description" the AI matched against was
  just "Title at Company. Location: X". Match scores were effectively guesses.
  jobspy uses the portals' own endpoints and returns the full posting
  (typically 4–10K characters). Tested live: Indeed (USA, India, UK) and
  LinkedIn work. Glassdoor (400), ZipRecruiter (403), Naukri (reCAPTCHA) and
  Google (0 results) were blocking, so they aren't offered.
- **Indeed filter quirk.** Indeed ignores its own remote and job-type filters
  when a "posted within" window is set. The app adds "remote" to the query and
  enforces both filters locally.
- **Two-stage ranking** to stay within Groq's free tier. Every job gets a free
  keyword score; only the top N (default 10) get an LLM call. An "AI-score the
  next 5" button extends the set.
- **Matching uses a compact profile, not the whole resume** (~350 tokens vs
  1–3K), plus the posting with EEO and legal boilerplate removed, capped at
  4,000 characters. Cover letters and tweaks use the full resume.
- **Structured output.** gpt-oss models use Groq's strict JSON-schema mode, so
  output always parses. Other models use tool calling.
  `reasoning_effort="low"` on gpt-oss cuts hidden reasoning tokens.
- **Rate limits.** `max_retries=6`, and the Groq SDK honours `retry-after` on
  429s. Fatal errors (bad key, forbidden, unknown model) stop the batch and
  show a clear message instead of failing every job.
- **Streamlit state.** Results live in `st.session_state`, so clicking a
  button doesn't wipe them. Cover letters and tweaks are generated only when
  clicked, inside `st.fragment`. The old app generated a letter plus tweaks
  for every job on every run: up to 60 LLM calls per search.
- **Career directions.** Scoring runs through a direction (role, caution, fits/avoid conditions, proof,
  search titles, good/bad posting phrases). The model judges company stage/context before the score, and
  code takes 15 points off 'outside' jobs (it used to cap them at 40, which left almost nothing for narrow
  directions like 'seed-stage only'). Outside jobs stay visible with a warning unless you hide them.
- **Company career boards (2026-10-09).** Google and Bing can't be scraped directly: a plain request gets
  Google's "enable JavaScript" wall, Bing returned no results, and Microsoft retired the Bing Search API in
  2025. jobspy's Glassdoor (400), ZipRecruiter (403) and Google (0) are still blocked. Instead, boards.py
  reads the public Greenhouse, Lever and Ashby job-board APIs: free, no blocking, full descriptions. The
  starter companies.txt has 299 boards checked live (~26K open jobs). A 'Product Manager' search found 174
  US/remote matches from the last 14 days. Board titles must match more strictly (2/3 of the words) than
  Indeed's (1/2), because boards list every job a company has.
- **Parallel scoring.** Groq's free limits are per model (8K tokens/min, 1K requests/day each, checked from
  the response headers). score_jobs() runs one worker thread per model (gpt-oss-120b, gpt-oss-20b,
  qwen3.8-27b), so about 3x the jobs per minute and per day. Threads never call st.*; results come back
  through a queue.
- **Auto-fill.** Measured on a live 88-job search: LinkedIn jobs (41) apply inside LinkedIn, and their terms
  ban bots, so they are never filled. Indeed jobs link out to Greenhouse/Ashby/SmartRecruiters (~20%),
  Workday (~13%) or custom sites. The helper fills by field labels: saved answers first (contact, links,
  visa/sponsorship, relocation, office days, 'interviewed here before', EEO), then one AI call for the
  remaining questions, using only the resume and saved facts. Colours: green = saved answer, orange = AI
  answer to check, red = yours (legal agreements, unknowns). It never clicks Submit and leaves CAPTCHAs to
  you; it doesn't try to get around them.
- **Portfolio pages (2026-10-17).** One website per job, so the hiring team sees the projects that matter
  for their posting. Claims come only from `my_data/portfolio_bank.json` (facts + projects you verified),
  copied word for word: the AI returns fact ids and project ids, never its own description of your work.
  `clean_plan()` drops unknown ids and duplicate asks, refuses a page with fewer than 3 provable asks, and
  falls back to your own headline/intro when the AI's names a number, tool or employer the bank doesn't state
  (`unsupported()`). Asks no fact proves are shown to you as gaps, never on the page. The page is rendered
  from a fixed template (all text HTML-escaped, only http(s) links), carries `noindex`, and ships with your
  resume (the tailored .docx if saved for that job). Publishing uses the GitHub contents API with a
  fine-grained token limited to one repo, or else the GitHub CLI's login (`gh auth token`, read per process,
  never stored); GitHub Pages serves it (repo H2hofficially/portfolio with `.nojekyll`, set up 2026-10-17; a
  test page went live 31 s after publishing). Netlify was ruled out: its free plan charges
  15 credits per production deploy and pauses every site at zero.
- **No placeholders (2026-10-17).** The cover letter and resume prompts used to ask for `[X%]`-style
  placeholders. Applicants send what the app writes, so the prompts now leave unknown numbers out
  and sign letters with the saved name. The Save button still refuses any bracket text as a safety net.

## 7. Security and privacy

| Issue in the old build | Fix |
|---|---|
| An API key once pasted in plaintext in the build notes | Removed and revoked; keys live only in `.env`, which git ignores |
| No `.gitignore`, so `.env` could be committed | `.gitignore` covers `.env`, venv, resumes (`*.pdf`, `*.docx`) and CSV exports |
| Resume written to `temp_resume.pdf` in the working folder and never deleted | Parsed in memory (`BytesIO`); nothing written to disk |
| Streamlit listened on all network interfaces, so anyone on your Wi-Fi could open the app, see your resume and spend your Groq quota | `server.address = "localhost"` |
| Unlimited upload size and PDF pages | 5 MB cap, first 10 PDF pages, 20K characters of text |
| Scraped titles/companies rendered as raw markdown (link injection, `$` → LaTeX, `:red[...]` directives) | `md_escape()` on all scraped fields. `untrusted_md()` strips images (tracking pixels) and escapes `$` in descriptions and LLM output. Only `http(s)` links become buttons. |
| Job postings are untrusted text inside the LLM prompt (prompt injection) | Wrapped in `<job_posting>` tags; the system prompt says to treat them as data and ignore instructions in them |
| CSV export of scraped text (formula injection when opened in Excel) | `csv_safe()` prefixes cells starting with `= + - @` |
| Streamlit telemetry on | `gatherUsageStats = false` |

What still leaves your machine: resume text goes to Groq for analysis, and
search terms go to Indeed and LinkedIn. Board searches only download public job lists.

Auto-fill stores your answers, a copy of your resume, generated cover letters, the helper's browser profile
(so Workday logins persist) and a log in `my_data/` (git-ignored), only after you press Save. Form
questions and the job posting go to Groq when the AI answers open questions; page text is treated as
untrusted (same prompt-injection guard as the matcher).

## 8. Groq free-tier limits

Per model: 8K tokens/min and 1K requests/day (gpt-oss-120b, gpt-oss-20b and
qwen3.8-27b, read from Groq's rate-limit headers on 2026-10-09). One match call
is about 2K tokens, so roughly 4 jobs/min per model, ~12/min with all three in
parallel. One auto-fill AI call is about 5K tokens.

## 9. Verification

```powershell
cd smart-job-assistant
venv\Scripts\python scraper.py     # live search, no key needed: expect ~10 jobs with descriptions
venv\Scripts\python matcher.py     # needs key: prints a profile, keyword score and AI match
venv\Scripts\python tailor.py      # needs key: prints a cover letter and tweaks
venv\Scripts\streamlit run app.py  # http://localhost:8501
```

Tested for this build:

- Live search on Indeed (USA, India, UK) and LinkedIn.
- Every AI call run against a local mock Groq server: strict schemas valid,
  429 retried, both the tool-calling and JSON-schema paths.
- The full UI clicked through in a browser: upload → profile → search →
  30 jobs → 10 scored → cover letter and tweaks.

- Live on Groq (`openai/gpt-oss-120b`): resume profile, 6 real jobs scored in
  13 s with one real 429 handled automatically (4 s wait), cover letter and
  tweaks. The first live run showed the model inventing metrics and courses
  from a thin resume. The tailor prompts now leave unknown numbers out
  (they used `[placeholders]` until 2026-10-17), move missing skills to a "Gaps to close" list, and run at
  temperature 0.3 with medium reasoning. Still read everything before sending.

Portfolio pages (2026-10-17): `venv\Scripts\python check_portfolio.py live` → 43/43, including a live build
for Hello There Collective's AI & Automation Specialist posting (5 provable asks, 4 projects, 5 gaps listed).
Screenshots at 1280 px (light) and 390 px (dark): no sideways scroll.

Auto-fill (2026-10-09), dry runs on live forms with every non-read request blocked, so nothing was sent:
Anthropic/Greenhouse 15 saved answers + 1 AI answer, 4 left for you (relocation set to 'Depends', AI-policy
and arbitration agreements); OpenAI/Ashby 12 filled (location typeahead, yes/no buttons, EEO radios);
Pigment/Lever 8 filled. Helper loop (queue -> open -> fill -> heartbeat) and the "Fill this page" button
tested on a local form.

## 10. Known limitations

- LinkedIn throttles heavy use. Around 50+ results per search it may return
  fewer jobs, and the app shows its warning.
- Scanned (image-only) PDFs have no text. Export a text PDF or DOCX instead.
- The AI score is a screening aid. Always read the posting before applying.
- Auto-fill is reliable on Greenhouse, Lever and Ashby. Workday and custom career sites are partial: press
  "Fill this page" on each step and finish the rest by hand. Workday needs an account per company.
- Some forms check for bots (reCAPTCHA, hCaptcha). The helper uses a normal visible browser and you submit,
  so challenges are rare; solve any that appear yourself.
