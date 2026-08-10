import streamlit as st
import sys
import os
import base64
import matplotlib.pyplot as plt
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BG_IMAGE_PATH = os.path.join(ASSETS_DIR, "background.png")

sys.path.append(SRC_DIR)

from predict import predict_churn

# Configure Streamlit page
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert background image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = ""
if os.path.exists(BG_IMAGE_PATH):
    bg_base64 = get_base64_of_bin_file(BG_IMAGE_PATH)

# Custom Glassmorphic CSS Theme
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {{
        font-family: 'Outfit', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(180deg, rgba(10, 15, 30, 0.88), rgba(15, 23, 42, 0.94)),
                    url("data:image/png;base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
        color: #F8FAFC;
    }}
    
    /* Header Card */
    .hero-header {{
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}
    
    .hero-title {{
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .hero-subtitle {{
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
    }}
    
    /* Section Glass Cards */
    .glass-card {{
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}
    
    .section-header {{
        font-size: 1.2rem;
        font-weight: 600;
        color: #38BDF8;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    /* Inputs Styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }}
    
    .stSelectbox label, .stNumberInput label {{
        color: #CBD5E1 !important;
        font-weight: 500;
        font-size: 0.95rem;
    }}
    
    /* Glowing Action Button */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 14px 28px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.6);
        transform: translateY(-2px);
    }}
    
    /* Prediction Badges */
    .badge-stay {{
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        padding: 16px 24px;
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }}
    
    .badge-churn {{
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #F87171;
        padding: 16px 24px;
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }}
    
    /* Footer */
    .dashboard-footer {{
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }}
    </style>
""", unsafe_allow_html=True)

# Top Hero Banner
st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">⚡ Customer Churn Intelligence Dashboard</h1>
        <p class="hero-subtitle">Predict customer attrition risk using Machine Learning & LIME Explainable AI</p>
    </div>
""", unsafe_allow_html=True)

# Main Form Container
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], help="0: No, 1: Yes")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)

    with col2:
        st.markdown('<div class="section-header">📶 Telecom Services</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown('<div class="section-header">💳 Billing & Contract</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔮 Predict Customer Churn Risk")

# Prediction Processing & Results
if submit_button:
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    with st.spinner("Analyzing profile & computing LIME explanation..."):
        prediction, probability, explanation = predict_churn(customer_data)

    st.markdown("---")
    st.subheader("📊 Prediction Results & AI Insights")

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Risk Badge
        if prediction == 1:
            st.markdown(
                f'<div class="badge-churn">⚠️ HIGH RISK OF CHURN</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="badge-stay">✅ LOW RISK - LIKELY TO STAY</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="Estimated Churn Probability",
            value=f"{probability * 100:.1f}%",
            delta=f"{(probability - 0.5) * 100:+.1f}% vs Threshold",
            delta_color="inverse"
        )

        st.progress(float(probability))
        
        # Key Recommendations
        st.markdown("#### 💡 AI Recommendations")
        if probability >= 0.5:
            st.warning("• Offer long-term contract discounts (e.g. 1 or 2-year contract).\n"
                       "• Provide tech support / security add-on bundle.\n"
                       "• Incentivize automatic bank/credit card payment method.")
        else:
            st.info("• Customer engagement is stable.\n"
                    "• Consider upselling premium streaming packages or high-speed upgrades.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 LIME Feature Impact Analysis")
        st.caption("Top factors driving this prediction (Red = Increases Churn Risk, Green = Lowers Churn Risk)")

        # Prepare LIME plot data
        if explanation:
            features = [x[0] for x in explanation[:8]]
            weights = [x[1] for x in explanation[:8]]

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#EF4444' if w > 0 else '#10B981' for w in weights]
            
            # Matplotlib styling for dark theme
            fig.patch.set_facecolor('#0F172A')
            ax.set_facecolor('#0F172A')
            
            y_pos = range(len(features))
            ax.barh(y_pos, weights, color=colors, height=0.55, edgecolor='none')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, color='#E2E8F0', fontsize=9)
            ax.axvline(0, color='#64748B', linewidth=0.8, linestyle='--')
            ax.tick_params(colors='#CBD5E1', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.set_xlabel('Feature Weight (Impact)', color='#94A3B8', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="dashboard-footer">
        Customer Churn Prediction System • Powered by Machine Learning & Streamlit
    </div>
""", unsafe_allow_html=True)