# 02_Woo_Fetch_Latest_Order_By_Name_fast.py
import io
import time
import re
from urllib.parse import urljoin
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Woo fetch latest order by Name", page_icon="🧾", layout="wide")
st.markdown(
    """
    <div style="display:block;align-items:center;gap:12px;">
        <img src="https://alternatehealthclub.com/wp-content/uploads/2025/08/AHC-New-V1.png" style="height:56px;image-rendering:-webkit-optimize-contrast;">
        <h2 style="margin:0;">Woo fetch latest order by Name</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Secrets
if "woo" not in st.secrets:
    st.error("Woo secrets missing. Add [woo] url, ck, cs in .streamlit/secrets.toml")
    st.stop()

BASE_URL = st.secrets["woo"]["url"].rstrip("/") + "/"
CK = st.secrets["woo"]["ck"]
CS = st.secrets["woo"]["cs"]

# Single session for keep-alive
SESSION = requests.Session()

# --------------- Text and name utils ---------------
def norm_text(s: str, lower=True, strip_punct=True, collapse_ws=True) -> str:
    if s is None:
        return ""
    out = str(s)
    if lower:
        out = out.lower()
    if strip_punct:
        out = re.sub(r"[^a-z0-9\s]", " ", out)
    if collapse_ws:
        out = re.sub(r"\s+", " ", out)
    return out.strip()

def tokens(s: str) -> list[str]:
    return [t for t in norm_text(s).split() if t]

def dedupe_repeated_sequence(tok: list[str]) -> list[str]:
    while len(tok) >= 2 and len(tok) % 2 == 0 and tok[: len(tok)//2] == tok[len(tok)//2 :]:
        tok = tok[: len(tok)//2]
    return tok

def canonical_name_simple(name: str) -> str:
    t = tokens(name)
    t = dedupe_repeated_sequence(t)
    return " ".join(t)

def canonical_name_from_parts(first: str, last: str) -> str:
    t_first = tokens(first)
    t_last = tokens(last)
    combined = dedupe_repeated_sequence(t_first + t_last)
    if t_first and t_first == t_last:
        combined = t_first
    combined = dedupe_repeated_sequence(combined)
    return " ".join(combined)

def name_ratio(a: str, b: str) -> float:
    a_c = canonical_name_simple(a)
    b_c = canonical_name_simple(b)
    if not a_c or not b_c:
        return 0.0
    return SequenceMatcher(None, a_c, b_c).ratio()

def split_first_last(name: str) -> tuple[str | None, str | None]:
    t = tokens(name)
    t = dedupe_repeated_sequence(t)
    if not t:
        return None, None
    if len(t) == 1:
        return t[0], None
    return t[0], t[-1]

def contains_token(candidate_name: str, token: str | None) -> bool:
    if not token:
        return False
    return token in set(tokens(candidate_name))

def match_class(candidate_name: str, target_full: str) -> str:
    f, l = split_first_last(target_full)
    cand = canonical_name_simple(candidate_name)
    has_f = contains_token(cand, f)
    has_l = contains_token(cand, l)
    if f and l and has_f and has_l:
        return "both"
    if l and has_l:
        return "last"
    if f and has_f:
        return "first"
    return "none"

def class_priority(c: str) -> int:
    order = {"both": 3, "last": 2, "first": 1, "none": 0}
    return order.get(c, 0)

# --------------- Address utils ---------------
def jaccard(a: str, b: str) -> float:
    A = set(tokens(a))
    B = set(tokens(b))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0

def address_concat(parts: dict) -> str:
    segs = [
        parts.get("address_1") or "",
        parts.get("address_2") or "",
        parts.get("city") or "",
        parts.get("state") or "",
        parts.get("postcode") or "",
        parts.get("country") or "",
    ]
    return " ".join([s for s in segs if s]).strip()

def extract_numeric_tokens(s: str) -> list[str]:
    return re.findall(r"\d+", s or "")

def address_match_metrics(excel_addr: str, wc_shipping: dict) -> dict:
    wc_addr_str = address_concat(wc_shipping or {})
    jac = jaccard(excel_addr, wc_addr_str)
    seqr = name_ratio(excel_addr, wc_addr_str)
    excel_nums = set(extract_numeric_tokens(excel_addr))
    wc_nums = set(extract_numeric_tokens(wc_addr_str))
    num_overlap = len(excel_nums & wc_nums)
    postcode = (wc_shipping or {}).get("postcode") or ""
    zip_match = bool(postcode and postcode in excel_nums)
    house_num_match = False
    addr1 = (wc_shipping or {}).get("address_1") or ""
    house_nums_wc = extract_numeric_tokens(addr1)
    if house_nums_wc:
        house_num_match = any(n == house_nums_wc[0] for n in excel_nums)
    return {
        "wc_joined": wc_addr_str,
        "jaccard": round(jac, 3),
        "seq_ratio": round(seqr, 3),
        "zip_match": zip_match,
        "house_num_match": house_num_match,
        "num_overlap": int(num_overlap),
    }

def address_accept(metrics: dict, th_jaccard: float, th_seq: float) -> bool:
    if metrics["jaccard"] >= th_jaccard:
        return True
    if metrics["seq_ratio"] >= th_seq:
        return True
    if metrics["zip_match"] and metrics["jaccard"] >= 0.20:
        return True
    if metrics["house_num_match"] and metrics["jaccard"] >= 0.25:
        return True
    return False

# --------------- Woo API helpers with caching ---------------
def _status_csv(statuses):
    if not statuses:
        return None
    return ",".join(statuses)

def woo_get(path: str, params: dict) -> requests.Response:
    url = urljoin(BASE_URL, path)
    resp = SESSION.get(url, params=params, auth=(CK, CS), timeout=30)
    resp.raise_for_status()
    return resp

@st.cache_data(ttl=1800, show_spinner=False)
def cached_customers_search(q: str, per_page: int) -> list:
    params = {
        "search": q,
        "per_page": per_page,
        "orderby": "id",
        "order": "desc",
    }
    r = woo_get("wp-json/wc/v3/customers", params)
    return r.json()

@st.cache_data(ttl=1800, show_spinner=False)
def cached_orders_page(q: str, page: int, per_page: int, status_csv: str | None, after_iso: str | None) -> list:
    params = {
        "search": q,
        "per_page": per_page,
        "page": page,
        "orderby": "date",
        "order": "desc",
    }
    if status_csv:
        params["status"] = status_csv
    if after_iso:
        params["after"] = after_iso
    r = woo_get("wp-json/wc/v3/orders", params)
    return r.json()

def fetch_latest_order_for_customer(customer_id: int, allowed_statuses: list[str] | None = None) -> dict | None:
    params = {
        "customer": customer_id,
        "per_page": 5,
        "orderby": "date",
        "order": "desc",
    }
    if allowed_statuses:
        params["status"] = _status_csv(allowed_statuses)
    r = woo_get("wp-json/wc/v3/orders", params)
    arr = r.json()
    if not arr:
        return None
    def _dt(o):
        return o.get("date_created_gmt") or o.get("date_created") or ""
    arr.sort(key=_dt, reverse=True)
    return arr[0]

def fetch_recent_orders_search(q: str, pages: int, per_page: int, allowed_statuses: list[str] | None, after_iso: str | None) -> list:
    results = []
    status_csv = _status_csv(allowed_statuses)
    for page in range(1, pages + 1):
        arr = cached_orders_page(q, page, per_page, status_csv, after_iso)
        if not arr:
            break
        results.extend(arr)
    return results

# --------------- Candidate selection with strict priority ---------------
def pick_best_customer_with_priority(customers: list, target_full: str, threshold: float) -> dict | None:
    cands = []
    for c in customers:
        full = canonical_name_from_parts(c.get("first_name", ""), c.get("last_name", ""))
        cls = match_class(full, target_full)
        score = name_ratio(full, target_full)
        cands.append((c, cls, score))
    if not cands:
        return None
    for cls_want in ("both", "last", "first"):
        tier = [(c, s) for c, cls, s in cands if cls == cls_want and s >= threshold]
        if tier:
            tier.sort(key=lambda x: x[1], reverse=True)
            return tier[0][0]
    return None

def pick_latest_order_by_billing_name_with_priority(orders: list, target_full: str, threshold: float) -> dict | None:
    cands = []
    for o in orders:
        b = o.get("billing") or {}
        full = canonical_name_from_parts(b.get("first_name", ""), b.get("last_name", ""))
        cls = match_class(full, target_full)
        score = name_ratio(full, target_full)
        cands.append((o, cls, score))
    if not cands:
        return None
    def dt_key(o):
        return o.get("date_created_gmt") or o.get("date_created") or ""
    for cls_want in ("both", "last", "first"):
        tier = [(o, s) for o, cls, s in cands if cls == cls_want and s >= threshold]
        if tier:
            tier.sort(key=lambda x: (dt_key(x[0]), x[1]), reverse=True)
            return tier[0][0]
    return None

# --------------- UI ---------------
st.subheader("Upload Excel with Name and Address")
uploaded = st.file_uploader("Excel file", type=["xlsx", "xls"])
if uploaded is None:
    st.stop()

try:
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", xls.sheet_names)
    df = xls.parse(sheet_name=sheet)
except Exception as e:
    st.error(f"Failed to read Excel. {e}")
    st.stop()

if df.empty:
    st.error("Selected sheet is empty")
    st.stop()

left, right = st.columns(2)
with left:
    name_col = st.selectbox("Name column", list(df.columns))
with right:
    addr_col = st.selectbox("Address column", list(df.columns))

st.subheader("Matching options")
c1, c2, c3 = st.columns(3)
with c1:
    name_conf_threshold = st.slider("Minimum name similarity to accept", 0.50, 0.99, 0.85, 0.01)
with c2:
    order_search_pages = st.number_input("Order search pages", min_value=1, max_value=20, value=3, step=1)
with c3:
    per_page_orders = st.number_input("Orders per page", min_value=20, max_value=100, value=100, step=10)

c4, c5, c6 = st.columns(3)
with c4:
    addr_accept_jaccard = st.slider("Accept address if Jaccard ≥", 0.10, 1.0, 0.40, 0.05)
with c5:
    addr_accept_ratio = st.slider("Accept address if Sequence ratio ≥", 0.10, 1.0, 0.60, 0.05)
with c6:
    pause_ms = st.number_input("Pause between rows ms", min_value=0, max_value=1000, value=0, step=50)

with st.expander("Advanced speed settings"):
    preload = st.checkbox("Preload cache for all unique queries", value=True)
    max_workers = st.number_input("Concurrent workers for preload", min_value=1, max_value=16, value=6, step=1)
    use_filter = st.checkbox("Filter by allowed statuses", value=True)
    allowed_statuses = ["processing", "completed", "on-hold", "pending", "failed"] if use_filter else None
    use_after = st.checkbox("Limit orders to recent period", value=True)
    default_after = (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z"
    after_iso = st.text_input("After ISO8601 UTC", value=default_after if use_after else "")

run = st.button("Fetch from Woo and build final sheet")
if not run:
    st.stop()

# --------------- Optional preload to warm caches ---------------
def build_prioritized_queries(fullname: str) -> list[tuple[str, str]]:
    f_tok, l_tok = split_first_last(fullname)
    queries = []
    full_q = canonical_name_simple(fullname)
    if full_q:
        queries.append(("both", full_q))
    if l_tok:
        queries.append(("last", l_tok))
    if f_tok:
        queries.append(("first", f_tok))
    return queries

if preload:
    unique_qs = set()
    for _, r in df.iterrows():
        n = canonical_name_simple(str(r.get(name_col, "") or ""))
        for _, q in build_prioritized_queries(n):
            if q:
                unique_qs.add(q)
    status_csv = _status_csv(allowed_statuses)
    st.info(f"Preloading {len(unique_qs)} unique queries into cache")
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for q in unique_qs:
            tasks.append(ex.submit(cached_customers_search, q, 100))
            for page in range(1, order_search_pages + 1):
                tasks.append(ex.submit(cached_orders_page, q, page, per_page_orders, status_csv, after_iso or None))
        for _ in as_completed(tasks):
            pass  # just warm the cache

# --------------- Processing ---------------
rows = []
progress = st.progress(0)
status = st.empty()
n = len(df)

for idx, row in df.iterrows():
    name_raw = str(row.get(name_col, "") or "")
    addr_raw = str(row.get(addr_col, "") or "")
    target_full = canonical_name_simple(name_raw)

    result = {
        "Name": name_raw,
        "InputAddress": addr_raw,
        "OrderID": None,
        "BillingEmail": None,
        "BillingPhone": None,
        "WooShippingAddress": None,
        "AddrJaccard": None,
        "AddrSeqRatio": None,
        "ZipMatch": None,
        "HouseNumMatch": None,
        "NumOverlap": None,
        "NameSimUsed": None,
        "MatchSource": None,
        "MatchNote": None,
    }

    try:
        # Priority queries full then last then first and early stop
        queries = build_prioritized_queries(target_full)

        order = None
        best_sim = 0.0

        # customers first by priority
        for cls_want, q in queries:
            customers = cached_customers_search(q, 100)
            cand = pick_best_customer_with_priority(customers, target_full, name_conf_threshold)
            if cand:
                full = canonical_name_from_parts(cand.get("first_name", ""), cand.get("last_name", ""))
                sim = name_ratio(full, target_full)
                if class_priority(match_class(full, target_full)) >= class_priority(cls_want) and sim >= name_conf_threshold:
                    order = fetch_latest_order_for_customer(cand["id"], allowed_statuses=allowed_statuses)
                    result["MatchSource"] = "customer"
                    best_sim = sim
                    break

        # fallback orders search by priority
        if order is None:
            for cls_want, q in queries:
                orders = fetch_recent_orders_search(q, order_search_pages, per_page_orders, allowed_statuses, after_iso or None)
                cand = pick_latest_order_by_billing_name_with_priority(orders, target_full, name_conf_threshold)
                if cand:
                    b = cand.get("billing") or {}
                    full = canonical_name_from_parts(b.get("first_name", ""), b.get("last_name", ""))
                    mcls = match_class(full, target_full)
                    sim = name_ratio(full, target_full)
                    if class_priority(mcls) >= class_priority(cls_want) and sim >= name_conf_threshold:
                        order = cand
                        result["MatchSource"] = "orders"
                        best_sim = sim
                        break

        if order:
            billing = order.get("billing") or {}
            shipping = order.get("shipping") or {}
            result["OrderID"] = order.get("id")
            result["BillingEmail"] = billing.get("email")
            result["BillingPhone"] = billing.get("phone")

            m = address_match_metrics(addr_raw, shipping)
            result["WooShippingAddress"] = m["wc_joined"]
            result["AddrJaccard"] = m["jaccard"]
            result["AddrSeqRatio"] = m["seq_ratio"]
            result["ZipMatch"] = m["zip_match"]
            result["HouseNumMatch"] = m["house_num_match"]
            result["NumOverlap"] = m["num_overlap"]

            ok = address_accept(m, addr_accept_jaccard, addr_accept_ratio)
            result["NameSimUsed"] = round(best_sim, 3)
            result["MatchNote"] = "OK" if ok else "Address differs"
        else:
            result["MatchNote"] = "No match"

    except requests.HTTPError as http_err:
        result["MatchNote"] = f"HTTP {http_err.response.status_code}"
    except Exception as e:
        result["MatchNote"] = f"Error {type(e).__name__}"

    rows.append(result)
    progress.progress(int(((idx + 1) / n) * 100))
    status.text(f"Processed {idx + 1} of {n}")
    if pause_ms:
        time.sleep(pause_ms / 1000.0)

progress.progress(100)
status.text("Done")

out_df = pd.DataFrame(rows)
st.subheader("Preview")
st.dataframe(out_df.head(30), use_container_width=True)

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    out_df.to_excel(writer, index=False, sheet_name="Output")
st.download_button(
    "Download final sheet",
    data=buf.getvalue(),
    file_name="woo_latest_order_enriched.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
