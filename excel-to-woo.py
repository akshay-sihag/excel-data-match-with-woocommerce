# 02_Woo_Fetch_Last_Order_By_Name.py
import io
import time
import re
from urllib.parse import urljoin
from difflib import SequenceMatcher

import requests
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(page_title="Woo fetch latest order by Name", page_icon="🧾", layout="wide")
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;">
        <img src="https://alternatehealthclub.com/wp-content/uploads/2025/08/AHC-New-V1.png" style="height:56px;image-rendering:-webkit-optimize-contrast;">
        <br><br>
        <h2 style="margin:0;">Woo fetch latest order by Name</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Secrets expected at .streamlit/secrets.toml
# [woo]
# url = "https://yourdomain.com"
# ck = "ck_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# cs = "cs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
if "woo" not in st.secrets:
    st.error("Woo secrets missing. Add [woo] url, ck, cs in .streamlit/secrets.toml")
    st.stop()

BASE_URL = st.secrets["woo"]["url"].rstrip("/") + "/"
CK = st.secrets["woo"]["ck"]
CS = st.secrets["woo"]["cs"]

# ---------------------------
# Text and name utilities
# ---------------------------
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
    """
    If the sequence is exactly repeated (e.g., ['haley','mcquire','haley','mcquire']),
    compress it to the first half. Repeat until stable.
    """
    while len(tok) >= 2 and len(tok) % 2 == 0:
        half = len(tok) // 2
        if tok[:half] == tok[half:]:
            tok = tok[:half]
        else:
            break
    return tok

def canonical_name_from_parts(first: str, last: str) -> str:
    """
    Build a canonical person name from Woo first and last names,
    resilient to the user accidentally entering full name in both fields
    or duplicating the full name twice.
    """
    t_first = tokens(first)
    t_last = tokens(last)
    combined = t_first + t_last
    combined = dedupe_repeated_sequence(combined)

    # If first and last are identical sets, keep unique order
    if t_first and t_first == t_last:
        combined = t_first

    # If still looks like a doubled full name, run once more
    combined = dedupe_repeated_sequence(combined)
    return " ".join(combined)

def canonical_name_simple(name: str) -> str:
    """
    Canonicalize a single full name string and compress doubled patterns.
    """
    t = tokens(name)
    t = dedupe_repeated_sequence(t)
    return " ".join(t)

def name_score(a: str, b: str) -> float:
    a_c = canonical_name_simple(a)
    b_c = canonical_name_simple(b)
    if not a_c or not b_c:
        return 0.0
    return SequenceMatcher(None, a_c, b_c).ratio()

def search_queries_for_name(name: str) -> list[str]:
    """
    Build a small set of queries to try
    1. full canonical name
    2. first token
    3. last token
    """
    t = tokens(name)
    t = dedupe_repeated_sequence(t)
    if not t:
        return []
    queries = [" ".join(t)]
    if len(t) >= 1:
        queries.append(t[0])
    if len(t) >= 2:
        queries.append(t[-1])
    # Uniquify preserving order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out

# ---------------------------
# Address utilities
# ---------------------------
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
    seqr = name_score(excel_addr, wc_addr_str)  # reuse sequence ratio on normalized strings
    # Numeric hints
    excel_nums = set(extract_numeric_tokens(excel_addr))
    wc_nums = set(extract_numeric_tokens(wc_addr_str))
    num_overlap = len(excel_nums & wc_nums)
    postcode = (wc_shipping or {}).get("postcode") or ""
    zip_match = postcode and postcode in excel_nums
    # House number naive check first numeric token
    house_num_match = False
    addr1 = (wc_shipping or {}).get("address_1") or ""
    house_nums_wc = extract_numeric_tokens(addr1)
    if house_nums_wc:
        house_num_match = any(n == house_nums_wc[0] for n in excel_nums)

    return {
        "wc_joined": wc_addr_str,
        "jaccard": round(jac, 3),
        "seq_ratio": round(seqr, 3),
        "zip_match": bool(zip_match),
        "house_num_match": bool(house_num_match),
        "num_overlap": int(num_overlap),
    }

def address_accept(metrics: dict, th_jaccard: float, th_seq: float) -> bool:
    if metrics["jaccard"] >= th_jaccard:
        return True
    if metrics["seq_ratio"] >= th_seq:
        return True
    # small boosters
    if metrics["zip_match"] and metrics["jaccard"] >= 0.20:
        return True
    if metrics["house_num_match"] and metrics["jaccard"] >= 0.25:
        return True
    return False

# ---------------------------
# Woo API helpers
# ---------------------------
def woo_get(path: str, params: dict) -> requests.Response:
    url = urljoin(BASE_URL, path)
    resp = requests.get(url, params=params, auth=(CK, CS), timeout=30)
    resp.raise_for_status()
    return resp

def fetch_customers_by_search(q: str, per_page=100) -> list:
    params = {
        "search": q,
        "per_page": per_page,
        "orderby": "id",
        "order": "desc",
    }
    r = woo_get("wp-json/wc/v3/customers", params)
    return r.json()

def fetch_latest_order_for_customer(customer_id: int, allowed_statuses: list[str] | None = None) -> dict | None:
    params = {
        "customer": customer_id,
        "per_page": 5,
        "orderby": "date",
        "order": "desc",
    }
    if allowed_statuses:
        params["status"] = ",".join(allowed_statuses)
    r = woo_get("wp-json/wc/v3/orders", params)
    arr = r.json()
    if not arr:
        return None
    # Pick newest by date_created_gmt or date_created
    def _dt(o):
        return o.get("date_created_gmt") or o.get("date_created") or ""
    arr.sort(key=_dt, reverse=True)
    return arr[0]

def fetch_recent_orders_search(q: str, pages=3, per_page=100, allowed_statuses: list[str] | None = None) -> list:
    results = []
    for page in range(1, pages + 1):
        params = {
            "search": q,
            "per_page": per_page,
            "page": page,
            "orderby": "date",
            "order": "desc",
        }
        if allowed_statuses:
            params["status"] = ",".join(allowed_statuses)
        r = woo_get("wp-json/wc/v3/orders", params)
        arr = r.json()
        if not arr:
            break
        results.extend(arr)
        time.sleep(0.15)
    return results

def pick_best_customer(customers: list, target_name: str, threshold: float) -> dict | None:
    best = None
    best_score = 0.0
    tn = canonical_name_simple(target_name)
    for c in customers:
        full = canonical_name_from_parts(c.get("first_name", ""), c.get("last_name", ""))
        s = name_score(full, tn)
        if s > best_score:
            best = c
            best_score = s
    return best if best and best_score >= threshold else None

def pick_latest_order_by_billing_name(orders: list, target_name: str, threshold: float) -> dict | None:
    tn = canonical_name_simple(target_name)
    candidates = []
    for o in orders:
        b = o.get("billing") or {}
        full = canonical_name_from_parts(b.get("first_name", ""), b.get("last_name", ""))
        if name_score(full, tn) >= threshold:
            candidates.append(o)
    if not candidates:
        return None
    def _dt(o):
        return o.get("date_created_gmt") or o.get("date_created") or ""
    candidates.sort(key=_dt, reverse=True)
    return candidates[0]

# ---------------------------
# UI inputs
# ---------------------------
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
    name_conf_threshold = st.slider("Minimum name similarity to accept", 0.50, 0.99, 0.80, 0.01)
with c2:
    order_search_pages = st.number_input("Fallback order search pages", min_value=1, max_value=20, value=5, step=1)
with c3:
    per_page_orders = st.number_input("Orders per page when searching", min_value=20, max_value=100, value=100, step=10)

c4, c5, c6 = st.columns(3)
with c4:
    addr_accept_jaccard = st.slider("Accept address if Jaccard ≥", 0.10, 1.0, 0.40, 0.05)
with c5:
    addr_accept_ratio = st.slider("Accept address if Sequence ratio ≥", 0.10, 1.0, 0.60, 0.05)
with c6:
    pause_ms = st.number_input("Pause between API calls ms", min_value=0, max_value=2000, value=100, step=50)

with st.expander("Order status filter optional"):
    use_filter = st.checkbox("Filter by allowed statuses", value=False)
    allowed_statuses = ["processing", "completed", "on-hold", "pending", "failed"]
    if use_filter:
        st.caption("Using default set processing, completed, on-hold, pending, failed")
    else:
        allowed_statuses = None

run = st.button("Fetch from Woo and build final sheet")
if not run:
    st.stop()

# ---------------------------
# Processing
# ---------------------------
rows = []
progress = st.progress(0)
status = st.empty()
n = len(df)

for idx, row in df.iterrows():
    name_raw = str(row.get(name_col, "") or "")
    addr_raw = str(row.get(addr_col, "") or "")
    target_name = canonical_name_simple(name_raw)

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
        "MatchSource": None,   # "customer" or "orders"
        "MatchNote": None
    }

    try:
        order = None

        # 1 try customer search with a few queries
        queries = search_queries_for_name(target_name)
        best_customer = None
        best_score = 0.0
        for q in queries:
            if not q:
                continue
            customers = fetch_customers_by_search(q)
            cand = pick_best_customer(customers, target_name, name_conf_threshold)
            if cand:
                # pick the best by similarity if multiple queries yield candidates
                full = canonical_name_from_parts(cand.get("first_name", ""), cand.get("last_name", ""))
                sc = name_score(full, target_name)
                if sc > best_score:
                    best_score = sc
                    best_customer = cand
            time.sleep(pause_ms / 1000.0)

        if best_customer:
            order = fetch_latest_order_for_customer(best_customer["id"], allowed_statuses=allowed_statuses)
            result["MatchSource"] = "customer"
            result["NameSimUsed"] = round(best_score, 3)

        # 2 fallback search orders by queries if still not found
        if order is None:
            best_order = None
            best_sim = 0.0
            for q in queries:
                orders = fetch_recent_orders_search(q, pages=order_search_pages, per_page=per_page_orders, allowed_statuses=allowed_statuses)
                cand = pick_latest_order_by_billing_name(orders, target_name, name_conf_threshold)
                if cand:
                    b = cand.get("billing") or {}
                    full = canonical_name_from_parts(b.get("first_name", ""), b.get("last_name", ""))
                    sc = name_score(full, target_name)
                    if sc > best_sim:
                        best_sim = sc
                        best_order = cand
                time.sleep(pause_ms / 1000.0)
            if best_order:
                order = best_order
                result["MatchSource"] = "orders"
                result["NameSimUsed"] = round(best_sim, 3)

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
            result["MatchNote"] = "OK" if ok else "Address differs"
        else:
            result["MatchNote"] = "No match"

    except requests.HTTPError as http_err:
        result["MatchNote"] = f"HTTP {http_err.response.status_code}"
    except Exception as e:
        result["MatchNote"] = f"Error {type(e).__name__}"

    rows.append(result)

    pct = int(((idx + 1) / n) * 100)
    progress.progress(min(pct, 100))
    status.text(f"Processed {idx + 1} of {n}")
    time.sleep(pause_ms / 1000.0)

progress.progress(100)
status.text("Done")

out_df = pd.DataFrame(rows)

st.subheader("Preview")
st.dataframe(out_df.head(30), use_container_width=True)

# Download
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    out_df.to_excel(writer, index=False, sheet_name="Output")
st.download_button(
    "Download final sheet",
    data=buf.getvalue(),
    file_name="woo_latest_order_enriched.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
