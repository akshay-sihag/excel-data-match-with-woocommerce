# Combined WooCommerce Match + FedEx Export App
import io
import time
import re
from urllib.parse import urljoin
from difflib import SequenceMatcher
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

# Page
st.set_page_config(page_title="WooCommerce to FedEx Export", page_icon="📦", layout="wide")
st.markdown(
    """
    <div style="display:block;align-items:center;gap:12px;">
        <img src="https://alternatehealthclub.com/wp-content/uploads/2025/08/AHC-New-V1.png" style="height:56px;image-rendering:-webkit-optimize-contrast;">
        <h2 style="margin:0;">WooCommerce Match + FedEx Export</h2>
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

# --------------- Woo API helpers ---------------
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

# --------------- Candidate selection ---------------
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

def pick_latest_order_by_shipping_name_with_priority(orders: list, target_full: str, threshold: float) -> dict | None:
    cands = []
    for o in orders:
        s = o.get("shipping") or {}
        full = canonical_name_from_parts(s.get("first_name", ""), s.get("last_name", ""))
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

# --------------- FedEx Constants ---------------
CONSTANT_DATA = {
    'serviceType': 'FEDEX_2_DAY',
    'shipmentType': 'OUTBOUND',
    'senderContactName': 'JOHN KATZ',
    'senderCompany': 'AHC',
    'senderContactNumber': '9477777228',
    'senderLine1': '108 NEW SOUTH RD',
    'senderLine2': 'STE B',
    'senderPostcode': '11801',
    'senderCity': 'HICKSVILLE',
    'senderState': 'NY',
    'senderCountry': 'US',
    'senderEmail': 'SUPPORT@ALTERNATEHEALTHCLUB.COM'
}

RECIPIENT_HEADERS = [
    'recipientContactName',
    'recipientContactNumber',
    'recipientLine1',
    'recipientLine2',
    'recipientPostcode',
    'recipientCity',
    'recipientState'
]

EMPTY_HEADERS = [
    'recipientCountry',
    'recipientResidential',
    'numberOfPackages',
    'packageWeight',
    'weightUnits',
    'packageType',
    'currencyType',
    'oneRatePricing'
]

# ==================== STREAMLIT UI ====================

st.markdown("---")
st.header("Step 1: Upload Excel and Match with WooCommerce")

uploaded = st.file_uploader("Upload Excel file with Name and Address", type=["xlsx", "xls"])

if uploaded is None:
    st.info("👆 Please upload your Excel file to begin.")
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

# Auto-detect name and address columns
def auto_detect_column(columns, patterns):
    """Auto-detect column based on name patterns"""
    for idx, col in enumerate(columns):
        col_lower = str(col).lower()
        for pattern in patterns:
            if pattern in col_lower:
                return idx
    return 0

name_patterns = ['name', 'customer', 'recipient', 'client', 'contact']
address_patterns = ['address', 'street', 'location', 'addr', 'shipping', 'delivery']

columns_list = list(df.columns)
name_idx = auto_detect_column(columns_list, name_patterns)
addr_idx = auto_detect_column(columns_list, address_patterns)

left, right = st.columns(2)
with left:
    name_col = st.selectbox("Name column", columns_list, index=name_idx)
with right:
    addr_col = st.selectbox("Address column", columns_list, index=addr_idx)

st.subheader("Matching options")
c1, c2, c3 = st.columns(3)
with c1:
    name_conf_threshold = st.slider("Minimum name similarity", 0.50, 0.99, 0.85, 0.01)
with c2:
    order_search_pages = st.number_input("Fallback order search pages", min_value=1, max_value=20, value=5, step=1)
with c3:
    per_page_orders = st.number_input("Orders per page", min_value=20, max_value=100, value=100, step=10)

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
    if not use_filter:
        allowed_statuses = None

run = st.button("🔍 Fetch from WooCommerce", type="primary")

# Initialize session state
if 'woo_output_df' not in st.session_state:
    st.session_state.woo_output_df = None

if run:
    # Processing
    rows = []
    progress = st.progress(0)
    status = st.empty()
    n = len(df)

    for idx, row in df.iterrows():
        name_raw = str(row.get(name_col, "") or "")
        addr_raw = str(row.get(addr_col, "") or "")
        target_full = canonical_name_simple(name_raw)

        result = {
            "Name": None,
            "InputAddress": addr_raw,
            "Phone": None,
            "Address line 1": None,
            "Address line 2": None,
            "City": None,
            "Zipcode": None,
            "State": None,
            "OrderID": None,
            "MatchNote": None,
        }

        try:
            f_tok, l_tok = split_first_last(target_full)
            queries = []
            full_q = canonical_name_simple(target_full)
            if full_q:
                queries.append(("both", full_q))
            if l_tok:
                queries.append(("last", l_tok))
            if f_tok:
                queries.append(("first", f_tok))

            order = None
            best_sim = 0.0

            # Priority 1: search customers
            best_customer = None
            for cls_want, q in queries:
                customers = fetch_customers_by_search(q)
                cand = pick_best_customer_with_priority(customers, target_full, name_conf_threshold)
                if cand:
                    full = canonical_name_from_parts(cand.get("first_name", ""), cand.get("last_name", ""))
                    sim = name_ratio(full, target_full)
                    if class_priority(match_class(full, target_full)) >= class_priority(cls_want) and sim >= name_conf_threshold:
                        best_customer = cand
                        best_sim = sim
                        break
                time.sleep(pause_ms / 1000.0)

            if best_customer:
                order = fetch_latest_order_for_customer(best_customer["id"], allowed_statuses=allowed_statuses)

            # Priority 2: orders search by billing name
            if order is None:
                for cls_want, q in queries:
                    orders = fetch_recent_orders_search(q, pages=order_search_pages, per_page=per_page_orders, allowed_statuses=allowed_statuses)
                    cand = pick_latest_order_by_billing_name_with_priority(orders, target_full, name_conf_threshold)
                    if cand:
                        b = cand.get("billing") or {}
                        full = canonical_name_from_parts(b.get("first_name", ""), b.get("last_name", ""))
                        mcls = match_class(full, target_full)
                        sim = name_ratio(full, target_full)
                        if class_priority(mcls) >= class_priority(cls_want) and sim >= name_conf_threshold:
                            order = cand
                            best_sim = sim
                            break
                    time.sleep(pause_ms / 1000.0)

            # Priority 3: orders search by shipping name (in case billing name is different)
            if order is None:
                for cls_want, q in queries:
                    orders = fetch_recent_orders_search(q, pages=order_search_pages, per_page=per_page_orders, allowed_statuses=allowed_statuses)
                    cand = pick_latest_order_by_shipping_name_with_priority(orders, target_full, name_conf_threshold)
                    if cand:
                        s = cand.get("shipping") or {}
                        full = canonical_name_from_parts(s.get("first_name", ""), s.get("last_name", ""))
                        mcls = match_class(full, target_full)
                        sim = name_ratio(full, target_full)
                        if class_priority(mcls) >= class_priority(cls_want) and sim >= name_conf_threshold:
                            order = cand
                            best_sim = sim
                            break
                    time.sleep(pause_ms / 1000.0)

            if order:
                billing = order.get("billing") or {}
                shipping = order.get("shipping") or {}

                # Try to get name from shipping first, fall back to billing
                # Use canonical_name_from_parts to clean up duplicates and formatting
                shipping_first = shipping.get("first_name", "")
                shipping_last = shipping.get("last_name", "")
                shipping_canonical = canonical_name_from_parts(shipping_first, shipping_last)
                
                if shipping_canonical:
                    full_name = shipping_canonical
                else:
                    billing_first = billing.get("first_name", "")
                    billing_last = billing.get("last_name", "")
                    full_name = canonical_name_from_parts(billing_first, billing_last)
                
                # Capitalize properly for output
                if full_name:
                    full_name = ' '.join(word.capitalize() for word in full_name.split())
                
                result["Name"] = full_name if full_name else None
                
                # Get phone number - try billing first, then shipping as fallback
                phone = billing.get("phone") or shipping.get("phone") or ""
                if phone:
                    phone = str(phone).strip()
                    # Strip country code prefix (+1, +91, etc.)
                    phone = re.sub(r'^\+\d{0,3}\s*', '', phone)
                    # Remove all non-digit characters (dashes, spaces, parentheses, etc.)
                    phone = re.sub(r'\D', '', phone)
                result["Phone"] = phone if phone else None
                
                # Address fields from shipping only
                result["Address line 1"] = shipping.get("address_1")
                result["Address line 2"] = shipping.get("address_2")
                result["City"] = shipping.get("city")
                result["Zipcode"] = shipping.get("postcode")
                result["State"] = shipping.get("state")
                result["OrderID"] = order.get("id")

                m = address_match_metrics(addr_raw, shipping)
                ok = address_accept(m, addr_accept_jaccard, addr_accept_ratio)
                result["MatchNote"] = "OK" if ok else "Address differs"
            else:
                # No match found - preserve input name
                result["Name"] = name_raw
                result["MatchNote"] = "No match"

        except requests.HTTPError as http_err:
            result["Name"] = name_raw  # Preserve input name on error
            result["MatchNote"] = f"HTTP {http_err.response.status_code}"
        except Exception as e:
            result["Name"] = name_raw  # Preserve input name on error
            result["MatchNote"] = f"Error {type(e).__name__}"

        rows.append(result)
        progress.progress(int(((idx + 1) / n) * 100))
        status.text(f"Processed {idx + 1} of {n}")
        time.sleep(pause_ms / 1000.0)

    progress.progress(100)
    status.text("✅ Done!")

    out_df = pd.DataFrame(rows)
    st.session_state.woo_output_df = out_df

# Display results if available
if st.session_state.woo_output_df is not None:
    out_df = st.session_state.woo_output_df
    
    st.markdown("---")
    st.header("Step 2: WooCommerce Match Results")
    
    st.subheader("Preview")
    st.dataframe(out_df.head(30), use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Output")
    
    st.download_button(
        "⬇️ Download WooCommerce Matched Data",
        data=buf.getvalue(),
        file_name="woo_latest_order_enriched.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ==================== FedEx Export Section ====================
    st.markdown("---")
    st.header("Step 3: Generate FedEx Export File")
    
    # Option to use WooCommerce data or upload new file
    data_source = st.radio(
        "Choose data source for FedEx export:",
        ["Use WooCommerce data above", "Upload a different Excel file"],
        horizontal=True
    )
    
    # Determine which dataframe to use
    fedex_source_df = None
    if data_source == "Use WooCommerce data above":
        fedex_source_df = out_df
        st.info("📊 Using WooCommerce matched data from Step 2")
    else:
        uploaded_fedex = st.file_uploader("Upload Excel file for FedEx export", type=["xlsx", "xls"], key="fedex_upload")
        if uploaded_fedex:
            try:
                fedex_xls = pd.ExcelFile(uploaded_fedex)
                fedex_sheet = st.selectbox("Select sheet", fedex_xls.sheet_names, key="fedex_sheet")
                fedex_source_df = fedex_xls.parse(sheet_name=fedex_sheet)
                st.success(f"✅ Loaded {len(fedex_source_df)} rows from uploaded file")
                with st.expander("📋 Preview of Uploaded File"):
                    st.dataframe(fedex_source_df.head(10))
            except Exception as e:
                st.error(f"Failed to read Excel file: {e}")
        else:
            st.info("👆 Please upload an Excel file to continue")
    
    if fedex_source_df is not None:
        st.markdown("---")
        st.subheader("Map columns to FedEx recipient fields")
        
        # Get source columns
        source_columns = ['-- Leave Empty --'] + fedex_source_df.columns.tolist()
        
        # Auto-detect column mapping with aliases
        def find_column_index(col_names, target_name, aliases=None):
            """Find best matching column index"""
            if aliases is None:
                aliases = []
            
            search_terms = [target_name.lower()] + [a.lower() for a in aliases]
            
            for idx, col in enumerate(col_names):
                col_lower = str(col).lower()
                for term in search_terms:
                    if term in col_lower or col_lower in term:
                        return idx
            return 0
        
        # Create mapping interface
        column_mapping = {}
        
        map_col1, map_col2 = st.columns(2)
    
        with map_col1:
            st.markdown("#### Recipient Information")
            column_mapping['recipientContactName'] = st.selectbox(
                '📝 Recipient Contact Name',
                source_columns,
                index=find_column_index(source_columns, 'name', ['contact', 'customer']),
                key='map_name'
            )
            
            column_mapping['recipientContactNumber'] = st.selectbox(
                '📞 Recipient Contact Number',
                source_columns,
                index=find_column_index(source_columns, 'phone', ['tel', 'telephone', 'mobile', 'contact']),
                key='map_number'
            )
            
            column_mapping['recipientLine1'] = st.selectbox(
                '🏠 Recipient Address Line 1',
                source_columns,
                index=find_column_index(source_columns, 'address line 1', ['address1', 'addr1', 'street']),
                key='map_line1'
            )
            
            column_mapping['recipientLine2'] = st.selectbox(
                '🏢 Recipient Address Line 2',
                source_columns,
                index=find_column_index(source_columns, 'address line 2', ['address2', 'addr2', 'apt', 'suite']),
                key='map_line2'
            )
        
        with map_col2:
            st.markdown("#### Location Details")
            column_mapping['recipientPostcode'] = st.selectbox(
                '📮 Recipient Postcode',
                source_columns,
                index=find_column_index(source_columns, 'zipcode', ['postcode', 'zip', 'postal']),
                key='map_postcode'
            )
            
            column_mapping['recipientCity'] = st.selectbox(
                '🏙️ Recipient City',
                source_columns,
                index=find_column_index(source_columns, 'city', ['town', 'municipality']),
                key='map_city'
            )
            
            column_mapping['recipientState'] = st.selectbox(
                '📍 Recipient State',
                source_columns,
                index=find_column_index(source_columns, 'state', ['province', 'region']),
                key='map_state'
            )
        
        # Check if at least one mapping is set
        has_mapping = any(v != '-- Leave Empty --' for v in column_mapping.values())
        
        if not has_mapping:
            st.warning("⚠️ Please map at least one column to proceed.")
        
        # Process button for FedEx
        if st.button("🚀 Generate FedEx Export File", type="primary", disabled=not has_mapping):
            with st.spinner("Generating FedEx file..."):
                num_rows = len(fedex_source_df)
            
                # Create output dataframe
                output_data = {}
                
                # Fill constant data
                for header, value in CONSTANT_DATA.items():
                    output_data[header] = [value] * num_rows
                
                # Copy recipient data based on mapping
                for target_header in RECIPIENT_HEADERS:
                    source_column = column_mapping.get(target_header, '-- Leave Empty --')
                    
                    if source_column != '-- Leave Empty --' and source_column in fedex_source_df.columns:
                        output_data[target_header] = fedex_source_df[source_column].tolist()
                    else:
                        output_data[target_header] = [''] * num_rows
                
                # Add empty columns
                for header in EMPTY_HEADERS:
                    output_data[header] = [''] * num_rows
                
                # Create output dataframe
                df_fedex = pd.DataFrame(output_data)
                
                # Generate filename
                current_date = datetime.now().strftime("%d-%m-%y")
                fedex_filename = f"Dispense-list-fedex-{current_date}.xlsx"
                
                # Create Excel file
                fedex_output = io.BytesIO()
                with pd.ExcelWriter(fedex_output, engine='openpyxl') as writer:
                    df_fedex.to_excel(writer, index=False, sheet_name='FedEx Dispatch')
                
                fedex_output.seek(0)
                
                st.success(f"✅ FedEx file generated! {len(df_fedex)} rows ready for export.")
                
                # Show preview
                with st.expander("📋 Preview of FedEx Export File"):
                    st.dataframe(df_fedex.head(10))
                
                # Download button
                st.download_button(
                    label="⬇️ Download FedEx Export File",
                    data=fedex_output,
                    file_name=fedex_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
                
                st.balloons()
