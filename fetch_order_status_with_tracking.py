# Fast Order Status and Tracking Fetcher
import io
import concurrent.futures
import pandas as pd
import requests
import streamlit as st
from urllib.parse import urljoin

# Page configuration
st.set_page_config(page_title="WooCommerce Order Status & Tracking Fetcher", page_icon="📋", layout="wide")

st.markdown(
    """
    <div style="display:block;align-items:center;gap:12px;">
        <img src="https://alternatehealthclub.com/wp-content/uploads/2025/08/AHC-New-V1.png" style="height:56px;image-rendering:-webkit-optimize-contrast;">
        <h2 style="margin:0;">Fast Order Status & Tracking Fetcher</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("Upload an Excel file with Order IDs to quickly fetch their status and tracking numbers from WooCommerce")

# Check for WooCommerce secrets
if "woo" not in st.secrets:
    st.error("WooCommerce secrets missing. Add [woo] url, ck, cs in .streamlit/secrets.toml")
    st.stop()

BASE_URL = st.secrets["woo"]["url"].rstrip("/") + "/"
CK = st.secrets["woo"]["ck"]
CS = st.secrets["woo"]["cs"]

# Helper function to fetch tracking numbers for an order
def fetch_tracking_numbers(order_id):
    """Fetch tracking numbers for a single order ID"""
    try:
        # Using WooCommerce Shipment Tracking API endpoint
        url = urljoin(BASE_URL, f"wp-json/wc-shipment-tracking/v3/orders/{order_id}/shipment-trackings")
        resp = requests.get(url, auth=(CK, CS), timeout=10)
        
        if resp.status_code == 404:
            return []
        
        resp.raise_for_status()
        trackings = resp.json()
        
        # Extract tracking numbers and providers
        tracking_list = []
        if isinstance(trackings, list):
            for tracking in trackings:
                tracking_number = tracking.get("tracking_number", "")
                tracking_provider = tracking.get("tracking_provider", tracking.get("custom_tracking_provider", ""))
                date_shipped = tracking.get("date_shipped", "")
                
                tracking_info = tracking_number
                if tracking_provider:
                    tracking_info = f"{tracking_number} ({tracking_provider})"
                
                tracking_list.append({
                    "number": tracking_number,
                    "provider": tracking_provider,
                    "date": date_shipped,
                    "display": tracking_info
                })
        
        return tracking_list
        
    except Exception:
        # If tracking endpoint fails, return empty list
        return []

# Helper function to fetch a single order with tracking
def fetch_order_status(order_id):
    """Fetch order status and tracking for a single order ID"""
    try:
        order_id = str(order_id).strip()
        if not order_id or order_id.lower() == 'nan':
            return {
                "OrderID": order_id,
                "Status": None,
                "DateCreated": None,
                "DatePaid": None,
                "Total": None,
                "CustomerName": None,
                "TrackingNumbers": None,
                "TrackingProviders": None,
                "TrackingCount": 0,
                "Error": "Invalid Order ID"
            }
        
        url = urljoin(BASE_URL, f"wp-json/wc/v3/orders/{order_id}")
        resp = requests.get(url, auth=(CK, CS), timeout=10)
        
        if resp.status_code == 404:
            return {
                "OrderID": order_id,
                "Status": None,
                "DateCreated": None,
                "DatePaid": None,
                "Total": None,
                "CustomerName": None,
                "TrackingNumbers": None,
                "TrackingProviders": None,
                "TrackingCount": 0,
                "Error": "Order not found"
            }
        
        resp.raise_for_status()
        order = resp.json()
        
        # Extract relevant information
        billing = order.get("billing", {})
        first_name = billing.get("first_name", "")
        last_name = billing.get("last_name", "")
        customer_name = f"{first_name} {last_name}".strip()
        
        # Fetch tracking numbers
        tracking_list = fetch_tracking_numbers(order_id)
        
        # Format tracking information
        tracking_numbers = ", ".join([t["number"] for t in tracking_list]) if tracking_list else None
        tracking_providers = ", ".join([t["provider"] for t in tracking_list if t["provider"]]) if tracking_list else None
        tracking_count = len(tracking_list)
        
        return {
            "OrderID": order_id,
            "Status": order.get("status"),
            "DateCreated": order.get("date_created"),
            "DatePaid": order.get("date_paid"),
            "Total": order.get("total"),
            "CustomerName": customer_name if customer_name else None,
            "TrackingNumbers": tracking_numbers,
            "TrackingProviders": tracking_providers,
            "TrackingCount": tracking_count,
            "Error": None
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "OrderID": order_id,
            "Status": None,
            "DateCreated": None,
            "DatePaid": None,
            "Total": None,
            "CustomerName": None,
            "TrackingNumbers": None,
            "TrackingProviders": None,
            "TrackingCount": 0,
            "Error": f"Error: {str(e)}"
        }
    except Exception as e:
        return {
            "OrderID": order_id,
            "Status": None,
            "DateCreated": None,
            "DatePaid": None,
            "Total": None,
            "CustomerName": None,
            "TrackingNumbers": None,
            "TrackingProviders": None,
            "TrackingCount": 0,
            "Error": f"Error: {str(e)}"
        }

# File uploader
st.subheader("📁 Upload Excel File")
uploaded = st.file_uploader("Upload Excel file with Order IDs", type=["xlsx", "xls"])

if uploaded is None:
    st.info("👆 Please upload your Excel file with Order IDs to begin.")
    st.stop()

try:
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", xls.sheet_names)
    df = xls.parse(sheet_name=sheet)
except Exception as e:
    st.error(f"Failed to read Excel file: {e}")
    st.stop()

if df.empty:
    st.error("Selected sheet is empty")
    st.stop()

# Select Order ID column
order_id_col = st.selectbox("Select Order ID column", list(df.columns))

# Performance settings
st.subheader("⚙️ Performance Settings")
max_workers = st.slider("Concurrent requests (higher = faster, but may hit rate limits)", 1, 20, 10, 1)

# Fetch button
if st.button("🚀 Fetch Order Statuses & Tracking", type="primary"):
    order_ids = df[order_id_col].tolist()
    
    st.info(f"Fetching status and tracking for {len(order_ids)} orders...")
    
    progress = st.progress(0)
    status_text = st.empty()
    
    results = []
    completed = 0
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_order = {executor.submit(fetch_order_status, oid): oid for oid in order_ids}
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_order):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Update progress
            progress.progress(completed / len(order_ids))
            status_text.text(f"Processed {completed} of {len(order_ids)} orders")
    
    progress.progress(1.0)
    status_text.text(f"✅ Completed! Processed {len(order_ids)} orders")
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["OrderID", "Status", "CustomerName", "Total", "TrackingCount", "TrackingNumbers", "TrackingProviders", "DateCreated", "DatePaid", "Error"]
    output_df = output_df[cols]
    
    # Show summary
    st.subheader("📊 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Orders", len(output_df))
    with col2:
        found = len(output_df[output_df["Status"].notna()])
        st.metric("Found", found)
    with col3:
        not_found = len(output_df[output_df["Error"] == "Order not found"])
        st.metric("Not Found", not_found)
    with col4:
        with_tracking = len(output_df[output_df["TrackingCount"] > 0])
        st.metric("With Tracking", with_tracking)
    with col5:
        errors = len(output_df[(output_df["Error"].notna()) & (output_df["Error"] != "Order not found")])
        st.metric("Errors", errors)
    
    # Status breakdown
    if found > 0:
        st.subheader("📈 Status Breakdown")
        status_counts = output_df["Status"].value_counts()
        st.bar_chart(status_counts)
    
    # Show preview
    st.subheader("📋 Preview")
    st.dataframe(output_df, use_container_width=True)
    
    # Download button
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        output_df.to_excel(writer, index=False, sheet_name='Order Status & Tracking')
    
    output.seek(0)
    
    st.download_button(
        label="⬇️ Download Results",
        data=output,
        file_name="order_status_tracking_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
    
    st.balloons()

