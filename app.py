"""
Tender Radar - a single-file Streamlit app for listing and matching tenders.

Public side : browse/filter tenders, AI "match my company", email alert signup.
Admin side  : dashboard, manage tenders, AI extraction from pasted text / PDF,
              URL sources (fetch + AI extract), subscribers, settings, backup.

Secrets (st.secrets or environment variables):
    GEMINI_API_KEY   - Google Gemini API key
    ADMIN_PASSWORD   - password for the Admin page
"""

import datetime as dt
import hashlib
import hmac
import io
import json
import os
import re
import sqlite3
from contextlib import closing
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Tender Radar", page_icon="📑", layout="wide")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DB_PATH = os.getenv("TENDER_DB_PATH", "tenders.db")
DEFAULT_MODEL = "gemini-2.5-flash"
USER_AGENT = "TenderRadarBot/1.0 (+contact: set-your-email@example.com)"
MAX_MATCH_CALLS_PER_SESSION = 5

CATEGORIES = [
    "Construction & Civil", "IT & Software", "Medical & Healthcare",
    "Supplies & Equipment", "Consultancy & Services", "Education & Training",
    "Energy & Utilities", "Transport & Logistics", "Agriculture & Food",
    "Security & Facility", "Other",
]
STATUSES = ["active", "draft", "closed"]
FIELDS = [
    "title", "authority", "category", "state", "estimated_value", "emd",
    "published_date", "deadline", "tender_url", "reference_no",
    "description", "eligibility", "ai_summary", "source", "status",
]


def secret(name, default=None):
    """Read from st.secrets, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def q(sql, params=()):
    with closing(db()) as c:
        return pd.read_sql_query(sql, c, params=params)


def ex(sql, params=()):
    with closing(db()) as c:
        cur = c.execute(sql, params)
        c.commit()
        return cur.rowcount


def init_db():
    with closing(db()) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS tenders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dedupe_key TEXT UNIQUE,
                title TEXT, authority TEXT, category TEXT, state TEXT,
                estimated_value TEXT, emd TEXT,
                published_date TEXT, deadline TEXT,
                tender_url TEXT, reference_no TEXT,
                description TEXT, eligibility TEXT, ai_summary TEXT,
                source TEXT, status TEXT DEFAULT 'active', created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS sources(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, url TEXT UNIQUE,
                enabled INTEGER DEFAULT 1, last_run TEXT, last_status TEXT
            );
            CREATE TABLE IF NOT EXISTS subscribers(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE, keywords TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            """
        )
        c.commit()
    # Auto-close tenders whose deadline has passed
    ex("UPDATE tenders SET status='closed' WHERE status='active' "
       "AND deadline IS NOT NULL AND deadline != '' AND deadline < date('now')")


def get_setting(key, default=""):
    df = q("SELECT value FROM settings WHERE key=?", (key,))
    return df.iloc[0, 0] if not df.empty else default


def set_setting(key, value):
    ex("INSERT INTO settings(key,value) VALUES(?,?) "
       "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


# --------------------------------------------------------------------------- #
# Record cleaning / saving
# --------------------------------------------------------------------------- #
def s(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def norm_date(v):
    v = s(v)
    if not v:
        return ""
    d = pd.to_datetime(v, dayfirst=True, errors="coerce")
    return "" if pd.isna(d) else d.date().isoformat()


def clean_record(r, source="", status=None):
    out = {f: s(r.get(f)) for f in FIELDS}
    out["published_date"] = norm_date(out["published_date"])
    out["deadline"] = norm_date(out["deadline"])
    if out["category"] not in CATEGORIES:
        out["category"] = "Other"
    if source:
        out["source"] = source
    out["status"] = status or out["status"] or "active"
    if out["status"] not in STATUSES:
        out["status"] = "active"
    return out


def dedupe_key(r):
    base = "|".join(s(r.get(k)).lower() for k in ("reference_no", "title", "authority"))
    return hashlib.sha1(base.encode()).hexdigest()


def insert_tender(rec):
    """Insert a cleaned record. Returns True if added, False if duplicate/empty."""
    if not rec["title"]:
        return False
    cols = FIELDS + ["dedupe_key", "created_at"]
    vals = [rec[f] for f in FIELDS] + [dedupe_key(rec), dt.datetime.now().isoformat(timespec="seconds")]
    sql = f"INSERT OR IGNORE INTO tenders({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
    return ex(sql, vals) > 0


def days_left(d):
    try:
        return (dt.date.fromisoformat(d) - dt.date.today()).days
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Gemini helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


def ai_json(prompt):
    """Call Gemini and return parsed JSON."""
    key = secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (add it to Streamlit secrets).")
    from google.genai import types
    resp = gemini_client(key).models.generate_content(
        model=get_setting("model", DEFAULT_MODEL),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.1
        ),
    )
    text = (resp.text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)


EXTRACT_PROMPT = """You extract tender / procurement notices from raw text.
Return ONLY a JSON array. One object per distinct tender found. Never invent
tenders or details; use null when a value is not in the text.
Keys: title, authority (issuing organisation), category, state (Indian state or
region), estimated_value (as written, e.g. "Rs 45 lakh"), emd (earnest money),
published_date (YYYY-MM-DD), deadline (YYYY-MM-DD, last date for submission),
tender_url (full URL if present), reference_no, description (1-2 sentences of
scope), eligibility (key eligibility conditions, short).
"category" must be exactly one of: __CATS__.
If the text contains no tenders, return [].

TEXT:
__TEXT__
"""


def extract_tenders(text):
    """Chunk long text, run Gemini extraction on each chunk, return list of dicts."""
    text = text.strip()
    chunks = [text[i:i + 15000] for i in range(0, min(len(text), 45000), 15000)]
    found = []
    for ch in chunks:
        prompt = EXTRACT_PROMPT.replace("__CATS__", ", ".join(CATEGORIES)).replace("__TEXT__", ch)
        data = ai_json(prompt)
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if lists else [data]
        found.extend(x for x in data if isinstance(x, dict))
    return found


def summarize_tender(row):
    prompt = (
        "Summarise this tender for a small business owner in 3 short bullet points "
        "(what is needed, who can bid, key deadline/cost). Return JSON: "
        '{"summary": "<text with bullets separated by newlines>"}\n\n'
        + json.dumps({k: s(row.get(k)) for k in
                      ("title", "authority", "category", "state", "estimated_value",
                       "emd", "deadline", "description", "eligibility")})
    )
    return s(ai_json(prompt).get("summary"))


# --------------------------------------------------------------------------- #
# Fetching sources / PDFs
# --------------------------------------------------------------------------- #
def allowed_by_robots(url):
    try:
        p = urlparse(url)
        r = requests.get(f"{p.scheme}://{p.netloc}/robots.txt", timeout=10,
                         headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return True
        rp = robotparser.RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def fetch_page_text(url):
    if not allowed_by_robots(url):
        raise RuntimeError("Blocked by robots.txt - skipped.")
    r = requests.get(url, timeout=25, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "footer"]):
        t.decompose()
    for a in soup.find_all("a", href=True):
        a.replace_with(f"{a.get_text(' ', strip=True)} ({urljoin(url, a['href'])})")
    return soup.get_text("\n", strip=True)


def pdf_text(file):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file.read()))
    return "\n".join((p.extract_text() or "") for p in reader.pages)[:45000]


def run_source(src):
    """Fetch one source, AI-extract tenders, store them. Returns (found, added)."""
    status = "active" if get_setting("auto_publish", "0") == "1" else "draft"
    try:
        items = extract_tenders(fetch_page_text(src["url"]))
        added = sum(insert_tender(clean_record(i, source=src["name"], status=status)) for i in items)
        msg = f"OK: {len(items)} found, {added} new"
        ex("UPDATE sources SET last_run=?, last_status=? WHERE id=?",
           (dt.datetime.now().isoformat(timespec="seconds"), msg, int(src["id"])))
        return len(items), added
    except Exception as e:
        ex("UPDATE sources SET last_run=?, last_status=? WHERE id=?",
           (dt.datetime.now().isoformat(timespec="seconds"), f"ERROR: {str(e)[:150]}", int(src["id"])))
        raise


# --------------------------------------------------------------------------- #
# Public pages
# --------------------------------------------------------------------------- #
def render_tender_card(r):
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.markdown(f"**{r['title']}**")
        left.caption(" · ".join(x for x in (r["authority"], r["category"], r["state"]) if x))
        dl = days_left(r["deadline"])
        if r["status"] == "closed":
            right.markdown("🔒 Closed")
        elif dl is not None:
            icon = "🔴" if dl <= 3 else "🟠" if dl <= 10 else "🟢"
            right.markdown(f"{icon} **{dl} days left**")
            right.caption(f"Due {r['deadline']}")
        if r["estimated_value"]:
            left.caption(f"Value: {r['estimated_value']}" + (f" · EMD: {r['emd']}" if r["emd"] else ""))
        with st.expander("Details"):
            if r["ai_summary"]:
                st.markdown(r["ai_summary"])
            if r["description"]:
                st.write(r["description"])
            if r["eligibility"]:
                st.markdown(f"**Eligibility:** {r['eligibility']}")
            if r["reference_no"]:
                st.caption(f"Ref: {r['reference_no']}")
            if r["tender_url"]:
                st.link_button("Open tender notice", r["tender_url"])


def page_browse():
    st.title("📑 Tender Radar")
    st.caption("Find tenders before your competitors do.")

    df = q("SELECT * FROM tenders WHERE status IN ('active','closed')")
    if df.empty:
        st.info("No tenders published yet. Check back soon.")
        return
    df = df.fillna("")

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    text = c1.text_input("Search", placeholder="e.g. dental equipment, road, software")
    cats = c2.multiselect("Category", sorted(df["category"].unique()))
    states = c3.multiselect("State", sorted(x for x in df["state"].unique() if x))
    window = c4.selectbox("Deadline", ["Any", "Next 7 days", "Next 30 days"])
    show_closed = st.checkbox("Include closed tenders", value=False)

    if not show_closed:
        df = df[df["status"] == "active"]
    if text:
        pat = re.escape(text.lower())
        hay = (df["title"] + " " + df["authority"] + " " + df["description"]).str.lower()
        df = df[hay.str.contains(pat, regex=True)]
    if cats:
        df = df[df["category"].isin(cats)]
    if states:
        df = df[df["state"].isin(states)]
    if window != "Any":
        n = 7 if "7" in window else 30
        df = df[df["deadline"].map(lambda d: (days_left(d) is not None) and 0 <= days_left(d) <= n)]

    df = df.assign(_k=df["deadline"].replace("", "9999-12-31")).sort_values("_k")
    st.write(f"**{len(df)}** tenders")
    limit = st.session_state.setdefault("limit", 25)
    for r in df.head(limit).to_dict("records"):
        render_tender_card(r)
    if len(df) > limit and st.button("Show more"):
        st.session_state["limit"] += 25
        st.rerun()


def page_match():
    st.title("🎯 Match tenders to my company")
    st.caption("Describe your business and AI will rank open tenders by fit.")
    profile = st.text_area(
        "Your company profile", height=160,
        placeholder="What you supply/do, locations you serve, turnover range, "
                    "certifications (GST, MSME, ISO), past similar work...")
    used = st.session_state.setdefault("match_calls", 0)
    if st.button("Find my matches", type="primary", disabled=not profile.strip()):
        if used >= MAX_MATCH_CALLS_PER_SESSION:
            st.warning("Match limit reached for this session. Please try later.")
            return
        df = q("SELECT * FROM tenders WHERE status='active' ORDER BY "
               "CASE WHEN deadline='' THEN '9999' ELSE deadline END LIMIT 40").fillna("")
        if df.empty:
            st.info("No active tenders to match yet.")
            return
        lines = [f"{r['id']} | {r['title']} | {r['category']} | {r['state']} | "
                 f"{r['estimated_value']} | elig: {r['eligibility'][:200]} | {r['description'][:250]}"
                 for r in df.to_dict("records")]
        prompt = (
            "You match tenders to a company. Company profile:\n" + profile.strip()[:2000] +
            "\n\nTenders (id | title | category | state | value | eligibility | description):\n" +
            "\n".join(lines) +
            "\n\nReturn ONLY a JSON array of up to 10 best matches, each as an object with keys "
            "id (integer), score (0-100 fit), reason (one sentence). Omit tenders scoring below 40."
        )
        with st.spinner("Matching..."):
            try:
                results = ai_json(prompt)
            except Exception as e:
                st.error(f"AI error: {e}")
                return
        st.session_state["match_calls"] = used + 1
        by_id = {int(r["id"]): r for r in df.to_dict("records")}
        shown = 0
        for m in sorted(results, key=lambda x: -int(x.get("score", 0))):
            r = by_id.get(int(m.get("id", -1)))
            if r:
                st.markdown(f"**Fit score: {int(m['score'])}/100** - {m.get('reason', '')}")
                render_tender_card(r)
                shown += 1
        if not shown:
            st.info("No strong matches right now. Try again as new tenders are added.")


def sidebar_subscribe():
    with st.sidebar.expander("🔔 Get tender alerts"):
        with st.form("sub", clear_on_submit=True):
            email = st.text_input("Email")
            kw = st.text_input("Keywords", placeholder="dental, medical equipment")
            if st.form_submit_button("Subscribe"):
                if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                    ex("INSERT OR REPLACE INTO subscribers(email,keywords,created_at) VALUES(?,?,?)",
                       (email.strip().lower(), kw.strip(), dt.datetime.now().isoformat(timespec="seconds")))
                    st.success("Subscribed!")
                else:
                    st.error("Enter a valid email.")


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
def admin_login():
    pw = secret("ADMIN_PASSWORD")
    if not pw:
        st.error("Set ADMIN_PASSWORD in Streamlit secrets to enable the admin area.")
        return False
    if st.session_state.get("is_admin"):
        return True
    st.title("🔐 Admin login")
    entered = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


def admin_dashboard():
    t = q("SELECT status, category, source, deadline FROM tenders")
    subs = q("SELECT COUNT(*) n FROM subscribers").iloc[0, 0]
    soon = sum(1 for d in t[t["status"] == "active"]["deadline"].fillna("")
               if (days_left(d) is not None and 0 <= days_left(d) <= 7))
    m = st.columns(5)
    m[0].metric("Total", len(t))
    m[1].metric("Active", int((t["status"] == "active").sum()))
    m[2].metric("Drafts to review", int((t["status"] == "draft").sum()))
    m[3].metric("Closing in 7 days", soon)
    m[4].metric("Subscribers", int(subs))
    if not t.empty:
        a, b = st.columns(2)
        a.subheader("By category")
        a.bar_chart(t["category"].value_counts())
        b.subheader("By source")
        b.bar_chart(t["source"].replace("", "manual").value_counts())
    st.caption(f"Gemini key: {'✅ set' if secret('GEMINI_API_KEY') else '❌ missing'} · "
               f"Model: {get_setting('model', DEFAULT_MODEL)}")


def admin_tenders():
    flt = st.selectbox("Show", ["all"] + STATUSES, index=0)
    df = q("SELECT id,status,title,authority,category,state,estimated_value,deadline,tender_url "
           "FROM tenders ORDER BY id DESC LIMIT 300").fillna("")
    if flt != "all":
        df = df[df["status"] == flt]
    df.insert(0, "delete", False)
    edited = st.data_editor(
        df, key="tender_editor", hide_index=True, disabled=["id"],
        column_config={
            "status": st.column_config.SelectboxColumn(options=STATUSES),
            "category": st.column_config.SelectboxColumn(options=CATEGORIES),
        },
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("💾 Save changes", type="primary"):
        for r in edited.to_dict("records"):
            if r["delete"]:
                ex("DELETE FROM tenders WHERE id=?", (int(r["id"]),))
            else:
                ex("UPDATE tenders SET status=?,title=?,authority=?,category=?,state=?,"
                   "estimated_value=?,deadline=?,tender_url=? WHERE id=?",
                   (r["status"], r["title"], r["authority"], r["category"], r["state"],
                    r["estimated_value"], norm_date(r["deadline"]), r["tender_url"], int(r["id"])))
        st.success("Saved.")
        st.rerun()
    if c2.button("✅ Publish all drafts"):
        n = ex("UPDATE tenders SET status='active' WHERE status='draft'")
        st.success(f"Published {n} drafts.")
        st.rerun()
    if c3.button("✨ AI-summarise 10 tenders without summary"):
        todo = q("SELECT * FROM tenders WHERE (ai_summary IS NULL OR ai_summary='') "
                 "AND status IN ('active','draft') LIMIT 10").fillna("")
        done = 0
        with st.spinner("Summarising..."):
            for r in todo.to_dict("records"):
                try:
                    ex("UPDATE tenders SET ai_summary=? WHERE id=?", (summarize_tender(r), int(r["id"])))
                    done += 1
                except Exception as e:
                    st.error(f"Stopped: {e}")
                    break
        st.success(f"Summarised {done}.")


def admin_add():
    mode = st.radio("How do you want to add tenders?",
                    ["Paste notice text (AI)", "Upload PDF (AI)", "Manual form"], horizontal=True)

    if mode == "Manual form":
        with st.form("manual", clear_on_submit=True):
            c1, c2 = st.columns(2)
            title = c1.text_input("Title*")
            authority = c2.text_input("Authority")
            category = c1.selectbox("Category", CATEGORIES)
            state = c2.text_input("State")
            value = c1.text_input("Estimated value")
            emd = c2.text_input("EMD")
            deadline = c1.date_input("Deadline", value=None)
            ref = c2.text_input("Reference no.")
            url = st.text_input("Tender URL")
            desc = st.text_area("Description")
            elig = st.text_area("Eligibility")
            status = st.selectbox("Status", STATUSES)
            if st.form_submit_button("Add tender", type="primary"):
                rec = clean_record({
                    "title": title, "authority": authority, "category": category, "state": state,
                    "estimated_value": value, "emd": emd, "deadline": deadline, "tender_url": url,
                    "reference_no": ref, "description": desc, "eligibility": elig,
                }, source="manual", status=status)
                st.success("Added.") if insert_tender(rec) else st.warning("Missing title or duplicate.")
        return

    raw = ""
    if mode == "Paste notice text (AI)":
        raw = st.text_area("Paste tender notice(s) text", height=250)
    else:
        f = st.file_uploader("Tender PDF", type=["pdf"])
        if f and st.button("Read PDF"):
            try:
                st.session_state["pdf_raw"] = pdf_text(f)
            except Exception as e:
                st.error(f"Could not read PDF: {e}")
        raw = st.session_state.get("pdf_raw", "")
        if raw:
            st.caption(f"PDF text loaded ({len(raw):,} characters).")

    if st.button("✨ Extract with Gemini", disabled=not raw.strip()):
        with st.spinner("Extracting..."):
            try:
                items = [clean_record(i, source="ai-import", status="draft") for i in extract_tenders(raw)]
                st.session_state["extracted"] = pd.DataFrame(items) if items else None
                if not items:
                    st.warning("No tenders found in that text.")
            except Exception as e:
                st.error(f"AI error: {e}")

    ext = st.session_state.get("extracted")
    if ext is not None and not ext.empty:
        st.subheader("Review before saving")
        ext = ext.copy()
        ext.insert(0, "save", True)
        edited = st.data_editor(
            ext, key="extract_editor", hide_index=True,
            column_config={"category": st.column_config.SelectboxColumn(options=CATEGORIES),
                           "status": st.column_config.SelectboxColumn(options=STATUSES)})
        if st.button("Save selected", type="primary"):
            n = sum(insert_tender(clean_record(r)) for r in edited.to_dict("records") if r["save"])
            st.session_state["extracted"] = None
            st.success(f"Saved {n} new tenders (duplicates skipped).")
            st.rerun()


def admin_sources():
    st.caption("Add public tender listing pages. Gemini reads each page and extracts tenders. "
               "robots.txt is respected; check each site's terms before scraping.")
    with st.form("add_source", clear_on_submit=True):
        c1, c2 = st.columns([1, 3])
        name = c1.text_input("Name")
        url = c2.text_input("Listing page URL")
        if st.form_submit_button("Add source") and name and url.startswith("http"):
            ex("INSERT OR IGNORE INTO sources(name,url) VALUES(?,?)", (name.strip(), url.strip()))
            st.rerun()

    df = q("SELECT id,name,url,enabled,last_run,last_status FROM sources")
    if df.empty:
        st.info("No sources yet.")
        return
    df["enabled"] = df["enabled"].astype(bool)
    df.insert(0, "delete", False)
    edited = st.data_editor(df, key="src_editor", hide_index=True,
                            disabled=["id", "last_run", "last_status"])
    c1, c2 = st.columns(2)
    if c1.button("💾 Save sources"):
        for r in edited.to_dict("records"):
            if r["delete"]:
                ex("DELETE FROM sources WHERE id=?", (int(r["id"]),))
            else:
                ex("UPDATE sources SET name=?,url=?,enabled=? WHERE id=?",
                   (r["name"], r["url"], int(r["enabled"]), int(r["id"])))
        st.rerun()
    if c2.button("▶️ Run all enabled sources", type="primary"):
        with st.spinner("Fetching and extracting..."):
            for r in edited.to_dict("records"):
                if r["enabled"] and not r["delete"]:
                    try:
                        found, added = run_source(r)
                        st.success(f"{r['name']}: {found} found, {added} new")
                    except Exception as e:
                        st.error(f"{r['name']}: {e}")
        st.caption("New tenders land as drafts unless auto-publish is on (Settings tab).")


def admin_subscribers():
    df = q("SELECT email, keywords, created_at FROM subscribers ORDER BY id DESC")
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "subscribers.csv", "text/csv")
    st.caption("Next step: send these people a weekly digest of matching tenders.")


def admin_settings():
    model = st.text_input("Gemini model", get_setting("model", DEFAULT_MODEL))
    auto = st.checkbox("Auto-publish tenders found by sources (skip draft review)",
                       value=get_setting("auto_publish", "0") == "1")
    if st.button("Save settings", type="primary"):
        set_setting("model", model.strip() or DEFAULT_MODEL)
        set_setting("auto_publish", "1" if auto else "0")
        st.success("Saved.")

    st.divider()
    st.subheader("Backup & restore")
    st.warning("Streamlit Community Cloud wipes local files on restart. "
               "Download a backup regularly, or move to a hosted database (e.g. Supabase) later.")
    all_t = q("SELECT * FROM tenders")
    st.download_button("Download tenders CSV", all_t.to_csv(index=False), "tenders_backup.csv", "text/csv")
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as fh:
            st.download_button("Download SQLite database", fh.read(), "tenders.db")
    up = st.file_uploader("Restore from tenders CSV", type=["csv"])
    if up and st.button("Import CSV"):
        imp = pd.read_csv(up)
        n = sum(insert_tender(clean_record(r)) for r in imp.to_dict("records"))
        st.success(f"Imported {n} new tenders.")


def page_admin():
    if not admin_login():
        return
    st.title("🛠️ Admin dashboard")
    if st.sidebar.button("Log out"):
        st.session_state["is_admin"] = False
        st.rerun()
    tabs = st.tabs(["Dashboard", "Tenders", "Add tenders", "Sources", "Subscribers", "Settings & backup"])
    for tab, fn in zip(tabs, [admin_dashboard, admin_tenders, admin_add,
                              admin_sources, admin_subscribers, admin_settings]):
        with tab:
            fn()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    init_db()
    st.sidebar.title("📑 Tender Radar")
    page = st.sidebar.radio("Menu", ["Browse tenders", "Match my company", "Admin"])
    if page == "Browse tenders":
        page_browse()
    elif page == "Match my company":
        page_match()
    else:
        page_admin()
    sidebar_subscribe()


main()
