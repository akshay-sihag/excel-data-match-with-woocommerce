# file: 02_Woo_Fetch_Last_Order_By_Name.py
import io
import time
import re
import math
from datetime import datetime
from urllib.parse import urljoin
import requests
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# Page setup
st.set_page_config(page_title="Woo fetch last order by Name", page_icon="🧾", layout="wide")

# Header with crisp logo and title side by side
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;">
        <img src="https://alternatehealthclub.com/wp-content/uploads/2025/08/AHC-New-V1.png" style="height:56px;image-rendering:-webkit-optimize-contrast;">
        <h2 style="margin:0;">Woo fetch last order by Name</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Secrets
# Put this in .streamlit/secrets.toml
# [woo]
# url = "https://yourdomain.com/"
# ck = "ck_xxx"
# cs = "cs_xxx"

if "woo" not in st.secrets:
    st.error("Woo secrets missing. Add [woo] url, ck, cs in .streamlit/secrets.toml")
    st.stop()

BASE_URL = st.secrets["woo"]["url"].rstrip("/") + "/"
CK = st.secrets["woo"]["ck"]
CS = st.secrets["woo"]["cs"]

# Helpers
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

def name_score(a: str, b: str) -> float:
    a_n = norm_text(a)
    b_n = norm_text(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()

def tokens(s: str) -> set:
    return set([t for t in norm_text(s).split() if t])

def jaccard(a: str, b: str) -> float:
    A = tokens(a)
    B = tokens(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0

def address_concat(shipping: dict) -> str:
    parts = [
        shipping.get("address_1") or "",
        shipping.get("address_2") or "",
        shipping.get("city") or "",
        shipping.get("state") or "",
        shipping.get("postcode") or "",
        shipping.get("country") or "",
    ]
    return " ".join([p for p in parts if p]).strip()

def woo_get(path: str, params: dict) -> requests.Response:
    url = urljoin(BASE_URL, path)
    # Basic auth with consumer key and secret
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

def fetch_latest_order_for_customer(customer_id: int) -> dict | None:
    params = {
        "customer": customer_id,
        "per_page": 1,
        "orderby": "date",
        "order": "desc",
        "status": "any",
    }
    r = woo_get("wp-json/wc/v3/orders", params)
    arr = r.json()
    return arr[0] if arr else None

def fetch_recent_orders_search(q: str, pages=3, per_page=100) -> list:
    results = []
    for page in range(1, pages + 1):
        params = {
            "search": q,
            "per_page": per_page,
            "page": page,
            "orderby": "date",
            "order": "desc",
            "status": "any",
        }
        r = woo_get("wp-json/wc/v3/orders", params)
        arr = r.json()
        if not arr:
            break
        results.extend(arr)
        # small pause to be polite
        time.sleep(0.2)
    return results

def pick_best_customer(customers: list, target_name: str, threshold: float) -> dict | None:
    best = None
    best_score = 0.0
    for c in customers:
        full = f"{c.get('first_name','')} {c.get('last_name','')}".strip()
        s = name_score(full, target_name)
        if s > best_score:
            best = c
            best_score = s
    return best if best and best_score >= threshold else None

def pick_best_order_by_billing_name(orders: list, target_name: str, threshold: float) -> dict | None:
    best = None
    best_score = 0.0
    for o in orders:
        b = o.get("billing") or {}
        full = f"{b.get('first_name','')} {b.get('last_name','')}".strip()
        s = name_score(full, target_name)
        if s >= threshold and s >= best_score:
            best = o
            best_score = s
    return best

def address_match_metrics(excel_addr: str, wc_shipping: dict) -> tuple[float, float, str]:
    wc_addr_str = address_concat(wc_shipping or {})
    return jaccard(excel_addr, wc_addr_str), name_score(excel_addr, wc_addr_str), wc_addr_str

# UI
st.subheader("Upload Excel with Name and Address")
uploaded = st.file_uploader("Excel file", type=["xlsx", "xls"])

if uploaded is None:
    st.stop()

# Read and select columns
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
    addr_accept_jaccard = st.slider("Accept address if Jaccard ≥", 0.10, 1.0, 0.40, 0.05)

c4, c5 = st.columns(2)
with c4:
    addr_accept_ratio = st.slider("Accept address if Sequence ratio ≥", 0.10, 1.0, 0.60, 0.05)
with c5:
    pause_ms = st.number_input("Pause between API calls ms", min_value=0, max_value=2000, value=100, step=50)

run = st.button("Fetch from Woo and build final sheet")

if not run:
    st.stop()

# Process
rows = []
progress = st.progress(0)
status = st.empty()

for idx, row in df.iterrows():
    name_raw = str(row.get(name_col, "") or "")
    addr_raw = str(row.get(addr_col, "") or "")
    target_name = norm_text(name_raw)

    result = {
        "Name": name_raw,
        "InputAddress": addr_raw,
        "OrderID": None,
        "BillingEmail": None,
        "BillingPhone": None,
        "WooShippingAddress": None,
        "AddrJaccard": None,
        "AddrSeqRatio": None,
        "MatchSource": None,   # "customer" or "orders"
        "MatchNote": None
    }

    try:
        # First try customers search
        customers = fetch_customers_by_search(target_name) if target_name else []
        best_customer = pick_best_customer(customers, target_name, name_conf_threshold) if customers else None

        order = None
        if best_customer:
            order = fetch_latest_order_for_customer(best_customer["id"])
            result["MatchSource"] = "customer"
        else:
            # Fallback: search orders by name text
            orders = fetch_recent_orders_search(target_name, pages=order_search_pages) if target_name else []
            order = pick_best_order_by_billing_name(orders, target_name, name_conf_threshold)
            result["MatchSource"] = "orders" if order else None

        if order:
            billing = order.get("billing") or {}
            shipping = order.get("shipping") or {}
            result["OrderID"] = order.get("id")
            result["BillingEmail"] = billing.get("email")
            result["BillingPhone"] = billing.get("phone")

            jac, seqr, wc_addr_str = address_match_metrics(addr_raw, shipping)
            result["WooShippingAddress"] = wc_addr_str
            result["AddrJaccard"] = round(jac, 3)
            result["AddrSeqRatio"] = round(seqr, 3)

            addr_ok = (jac >= addr_accept_jaccard) or (seqr >= addr_accept_ratio)
            result["MatchNote"] = "OK" if addr_ok else "Address differs"
        else:
            result["MatchNote"] = "No match"

    except requests.HTTPError as http_err:
        result["MatchNote"] = f"HTTP {http_err.response.status_code}"
    except Exception as e:
        result["MatchNote"] = f"Error {type(e).__name__}"

    rows.append(result)

    # UI progress
    pct = int(((idx + 1) / len(df)) * 100)
    progress.progress(min(pct, 100))
    status.text(f"Processed {idx + 1} of {len(df)}")
    time.sleep(pause_ms / 1000.0)

progress.progress(100)
status.text("Done")

out_df = pd.DataFrame(rows)

st.subheader("Preview")
st.dataframe(out_df.head(30), use_container_width=True)

# Downloads
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    out_df.to_excel(writer, index=False, sheet_name="Output")
st.download_button(
    "Download final sheet",
    data=buf.getvalue(),
    file_name="woo_last_order_enriched.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
