# 02_Woo_Fetch_Latest_Order_By_Name_indexed.py
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

# Page
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

# --------------- Woo API ---------------
def woo_get(path: str, params: dict) -> requests.Response:
    url = urljoin(BASE_URL, path)
    resp = SESSION.get(url, params=params, auth=(CK, CS), timeout=30)
    resp.raise_for_status()
    return resp

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_customers(per_page: int = 100, max_pages: int = 2000) -> list:
    customers = []
    for page in range(1, max_pages + 1):
        r = woo_get("wp-json/wc/v3/customers", {
            "per_page": per_page,
            "page": page,
            "orderby": "id",
            "order": "desc",
        })
        arr = r.json()
        if not arr:
            break
        customers.extend(arr)
        # Stop if fewer than per_page which means last page
        if len(arr) < per_page:
            break
    return customers

@st.cache_data(ttl=3600, show_spinner=False)
def latest_order_for_customer_cached(customer_id: int, status_csv: str | None) -> dict | None:
    params = {
        "customer": customer_id,
        "per_page": 5,
        "orderby": "date",
        "order": "desc",
    }
    if status_csv:
        params["status"] = status_csv
    r = woo_get("wp-json/wc/v3/orders", params)
    arr = r.json()
    if not arr:
        return None
    def _dt(o):
        return o.get("date_created_gmt") or o.get("date_created") or ""
    arr.sort(key=_dt, reverse=True)
    return arr[0]

def status_csv(statuses):
    if not statuses:
        return None
    return ",".join(statuses)

# --------------- Customer index and matching ---------------
class CustomerIndex:
    def __init__(self, customers: list[dict]):
        rows = []
        for c in customers:
            first = c.get("first_name", "") or ""
            last = c.get("last_name", "") or ""
            full = canonical_name_from_parts(first, last)
            rows.append({
                "id": c.get("id"),
                "first": first,
                "last": last,
                "full": full,
                "first_tok": tokens(first)[0] if tokens(first) else None,
                "last_tok": tokens(last)[-1] if tokens(last) else None,
            })
        self.df = pd.DataFrame(rows)

        # Build simple dict indexes
        self.by_full = {}
        for _, r in self.df.iterrows():
            self.by_full.setdefault(r["full"], []).append(int(r["id"]))

        self.by_last = {}
        for _, r in self.df.iterrows():
            if r["last_tok"]:
                self.by_last.setdefault(r["last_tok"], []).append(int(r["id"]))

        self.by_first = {}
        for _, r in self.df.iterrows():
            if r["first_tok"]:
                self.by_first.setdefault(r["first_tok"], []).append(int(r["id"]))

    def get_candidates(self, target_full: str) -> list[int]:
        # Priority 1 exact full canonical match
        t_full = canonical_name_simple(target_full)
        if t_full in self.by_full:
            return list(self.by_full[t_full])

        # Priority 2 match by last token
        _, l = split_first_last(target_full)
        if l and l in self.by_last:
            return list(self.by_last[l])

        # Priority 3 match by first token
        f, _ = split_first_last(target_full)
        if f and f in self.by_first:
            return list(self.by_first[f])

        return []

    def best_customer(self, target_full: str, threshold: float) -> int | None:
        cand_ids = self.get_candidates(target_full)
        if not cand_ids:
            return None

        # Score and class priority
        t = canonical_name_simple(target_full)
        records = self.df[self.df["id"].isin(cand_ids)].copy()
        if records.empty:
            return None

        def row_score(row):
            cls = match_class(row["full"], t)
            sc = name_ratio(row["full"], t)
            return class_priority(cls), sc

        records["cls_pri"] = records.apply(lambda r: class_priority(match_class(r["full"], t)), axis=1)
        records["sim"] = records.apply(lambda r: name_ratio(r["full"], t), axis=1)
        # Filter by threshold
        records = records[records["sim"] >= threshold]
        if records.empty:
            return None
        # Sort by class priority then similarity
        records = records.sort_values(by=["cls_pri", "sim"], ascending=[False, False])
        return int(records.iloc[0]["id"])

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

st.subheader("Matching and speed options")
c1, c2, c3 = st.columns(3)
with c1:
    name_conf_threshold = st.slider("Minimum name similarity to accept", 0.50, 0.99, 0.85, 0.01)
with c2:
    max_workers = st.number_input("Concurrent order fetch workers", min_value=1, max_value=24, value=8, step=1)
with c3:
    use_status_filter = st.checkbox("Filter order statuses", value=True)

allowed_statuses = ["processing", "completed", "on-hold", "pending", "failed"] if use_status_filter else None

c4, c5 = st.columns(2)
with c4:
    use_order_fallback = st.checkbox("If no customer match, try order search by name", value=False)
with c5:
    recent_days = st.number_input("Limit order search to last N days", min_value=7, max_value=3650, value=365, step=7)

run = st.button("Build final sheet")
if not run:
    st.stop()

# --------------- Build customer index once ---------------
with st.spinner("Fetching customers and building index"):
    all_customers = fetch_all_customers(per_page=100)
    idx = CustomerIndex(all_customers)

# --------------- Phase 1 map each row to a customer id ---------------
name_to_customer = []
for _, r in df.iterrows():
    target_full = canonical_name_simple(str(r.get(name_col, "") or ""))
    cid = idx.best_customer(target_full, name_conf_threshold)
    name_to_customer.append(cid)

# --------------- Phase 2 fetch latest orders for unique customers with concurrency ---------------
unique_customers = sorted({cid for cid in name_to_customer if cid})
status_str = status_csv(allowed_statuses)

def fetch_latest_for_id(cid):
    return cid, latest_order_for_customer_cached(cid, status_str)

orders_map = {}
if unique_customers:
    with st.spinner(f"Fetching latest orders for {len(unique_customers)} customers"):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_latest_for_id, cid) for cid in unique_customers]
            for fut in as_completed(futures):
                cid, order = fut.result()
                orders_map[cid] = order

# Optional fallback for names without a customer match
def fetch_recent_orders_search(q: str, pages: int, per_page: int, status_csv: str | None, after_iso: str | None) -> list:
    results = []
    for page in range(1, pages + 1):
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
        arr = r.json()
        if not arr:
            break
        results.extend(arr)
    return results

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

after_iso = (datetime.utcnow() - timedelta(days=int(recent_days))).isoformat() + "Z" if use_order_fallback else None

# --------------- Phase 3 build output rows ---------------
rows = []
progress = st.progress(0)
n = len(df)

for i, r in enumerate(df.itertuples(index=False), start=1):
    name_raw = str(getattr(r, name_col) if hasattr(r, name_col) else "")
    addr_raw = str(getattr(r, addr_col) if hasattr(r, addr_col) else "")
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
        "MatchSource": None,
        "MatchNote": None,
    }

    cid = name_to_customer[i - 1]
    order = None

    if cid:
        order = orders_map.get(cid)
        result["MatchSource"] = "customer"

    # Optional fallback by order search when no customer id found
    if order is None and use_order_fallback and target_full:
        # strict priority full then last then first
        f_tok, l_tok = split_first_last(target_full)
        queries = []
        full_q = canonical_name_simple(target_full)
        if full_q:
            queries.append(("both", full_q))
        if l_tok:
            queries.append(("last", l_tok))
        if f_tok:
            queries.append(("first", f_tok))

        for cls_want, q in queries:
            orders = fetch_recent_orders_search(q, pages=2, per_page=100, status_csv=status_str, after_iso=after_iso)
            cand = pick_latest_order_by_billing_name_with_priority(orders, target_full, name_conf_threshold)
            if cand:
                order = cand
                result["MatchSource"] = "orders"
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

        ok = address_accept(m, th_jaccard=0.40, th_seq=0.60)
        result["MatchNote"] = "OK" if ok else "Address differs"
    else:
        result["MatchNote"] = "No match"

    rows.append(result)
    progress.progress(int((i / n) * 100))

progress.progress(100)

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
