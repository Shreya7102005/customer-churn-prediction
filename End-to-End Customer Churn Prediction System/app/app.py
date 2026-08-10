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
LOGO_IMAGE_PATH = os.path.join(ASSETS_DIR, "logo.png")

sys.path.append(SRC_DIR)

from predict import predict_churn

# Configure Streamlit page
st.set_page_config(
    page_title="ChurnGuard | AI Predictive Customer Retention Safeguard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert binary image file to base64 string
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = get_base64_of_bin_file(BG_IMAGE_PATH)
logo_base64 = get_base64_of_bin_file(LOGO_IMAGE_PATH)

# Custom Glassmorphic & ChurnGuard CSS Theme
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(180deg, rgba(8, 14, 26, 0.90), rgba(15, 23, 42, 0.95)),
                    url("data:image/png;base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
        color: #F8FAFC;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }}
    
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 8px 24px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }}
    
    .sidebar-logo {{
        width: 48px;
        height: 48px;
        border-radius: 12px;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
    }}
    
    .brand-title {{
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .brand-tag {{
        font-size: 0.72rem;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    /* Hero Header Card */
    .hero-header {{
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    .hero-left {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}
    
    .hero-logo {{
        width: 64px;
        height: 64px;
        border-radius: 16px;
        box-shadow: 0 0 24px rgba(99, 102, 241, 0.5);
    }}
    
    .hero-title {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .hero-subtitle {{
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 4px;
    }}
    
    .status-badge {{
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.3);
        color: #38BDF8;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }}
    
    /* Section Glass Cards */
    .glass-card {{
        background: rgba(15, 23, 42, 0.70);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }}
    
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    /* Input Control Overrides */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }}
    
    .stSelectbox label, .stNumberInput label {{
        color: #CBD5E1 !important;
        font-weight: 600;
        font-size: 0.92rem;
    }}
    
    /* Glowing Action Button */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 14px 28px;
        border-radius: 14px;
        border: none;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45);
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.65);
        transform: translateY(-2px);
    }}
    
    /* ChurnGuard Shields */
    .shield-safe {{
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }}
    
    .shield-danger {{
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #F87171;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }}
    
    /* Metric Card Styling */
    .metric-box {{
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }}
    
    .metric-val {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #38BDF8;
    }}
    
    .metric-lbl {{
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
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

# Session state initialization for presets
default_values = {
    'gender': 'Female',
    'senior': 0,
    'partner': 'No',
    'dependents': 'No',
    'tenure': 3,
    'phone_service': 'Yes',
    'multiple_lines': 'No',
    'internet_service': 'Fiber optic',
    'online_security': 'No',
    'online_backup': 'No',
    'device_protection': 'No',
    'tech_support': 'No',
    'streaming_tv': 'Yes',
    'streaming_movies': 'Yes',
    'contract': 'Month-to-month',
    'paperless': 'Yes',
    'payment_method': 'Electronic check',
    'monthly_charges': 95.5,
    'total_charges': 286.5
}

for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_preset(preset_type):
    if preset_type == "high_risk":
        st.session_state['gender'] = 'Female'
        st.session_state['senior'] = 1
        st.session_state['partner'] = 'No'
        st.session_state['dependents'] = 'No'
        st.session_state['tenure'] = 2
        st.session_state['phone_service'] = 'Yes'
        st.session_state['multiple_lines'] = 'Yes'
        st.session_state['internet_service'] = 'Fiber optic'
        st.session_state['online_security'] = 'No'
        st.session_state['online_backup'] = 'No'
        st.session_state['device_protection'] = 'No'
        st.session_state['tech_support'] = 'No'
        st.session_state['streaming_tv'] = 'Yes'
        st.session_state['streaming_movies'] = 'Yes'
        st.session_state['contract'] = 'Month-to-month'
        st.session_state['paperless'] = 'Yes'
        st.session_state['payment_method'] = 'Electronic check'
        st.session_state['monthly_charges'] = 98.50
        st.session_state['total_charges'] = 197.00
    elif preset_type == "low_risk":
        st.session_state['gender'] = 'Male'
        st.session_state['senior'] = 0
        st.session_state['partner'] = 'Yes'
        st.session_state['dependents'] = 'Yes'
        st.session_state['tenure'] = 65
        st.session_state['phone_service'] = 'Yes'
        st.session_state['multiple_lines'] = 'No'
        st.session_state['internet_service'] = 'DSL'
        st.session_state['online_security'] = 'Yes'
        st.session_state['online_backup'] = 'Yes'
        st.session_state['device_protection'] = 'Yes'
        st.session_state['tech_support'] = 'Yes'
        st.session_state['streaming_tv'] = 'No'
        st.session_state['streaming_movies'] = 'No'
        st.session_state['contract'] = 'Two year'
        st.session_state['paperless'] = 'No'
        st.session_state['payment_method'] = 'Credit card (automatic)'
        st.session_state['monthly_charges'] = 55.00
        st.session_state['total_charges'] = 3575.00

# Sidebar Section
with st.sidebar:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="sidebar-logo" />' if logo_base64 else '🛡️'
    st.markdown(f"""
        <div class="sidebar-brand">
            {logo_html}
            <div>
                <div class="brand-title">ChurnGuard</div>
                <div class="brand-tag">v2.0 Enterprise AI</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Demo Presets")
    st.caption("Load pre-configured customer profiles to test ChurnGuard AI")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🚨 High Risk", use_container_width=True):
            load_preset("high_risk")
            st.rerun()
    with col_p2:
        if st.button("🛡️ Low Risk", use_container_width=True):
            load_preset("low_risk")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown('<div class="metric-box"><div class="metric-val">81.4%</div><div class="metric-lbl">ACCURACY</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-box"><div class="metric-val">0.84</div><div class="metric-lbl">AUC SCORE</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("🛡️ **ChurnGuard AI** utilizes Random Forest classification integrated with LIME tabular explainers.")

# Main Hero Header
logo_hero_html = f'<img src="data:image/png;base64,{logo_base64}" class="hero-logo" />' if logo_base64 else ''
st.markdown(f"""
    <div class="hero-header">
        <div class="hero-left">
            {logo_hero_html}
            <div>
                <h1 class="hero-title">ChurnGuard AI</h1>
                <div class="hero-subtitle">Predictive Customer Churn Intelligence & Automated Retention Safeguard</div>
            </div>
        </div>
        <div>
            <span class="status-badge">● LIVE MODEL ONLINE</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main Form Container
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="section-header">👤 Customer Demographics</div>', unsafe_allow_html=True)
        
        gender = st.selectbox(
            "Gender", ["Male", "Female"],
            index=["Male", "Female"].index(st.session_state['gender'])
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1],
            index=[0, 1].index(st.session_state['senior']),
            help="0: No, 1: Yes"
        )
        partner = st.selectbox(
            "Partner", ["Yes", "No"],
            index=["Yes", "No"].index(st.session_state['partner'])
        )
        dependents = st.selectbox(
            "Dependents", ["Yes", "No"],
            index=["Yes", "No"].index(st.session_state['dependents'])
        )
        tenure = st.number_input(
            "Tenure (Months)", min_value=0, max_value=120,
            value=int(st.session_state['tenure'])
        )

    with col2:
        st.markdown('<div class="section-header">📶 Subscribed Services</div>', unsafe_allow_html=True)
        
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"],
            index=["Yes", "No"].index(st.session_state['phone_service'])
        )
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"],
            index=["Yes", "No", "No phone service"].index(st.session_state['multiple_lines'])
        )
        internet_service = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"],
            index=["Fiber optic", "DSL", "No"].index(st.session_state['internet_service'])
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(st.session_state['online_security'])
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(st.session_state['online_backup'])
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(st.session_state['device_protection'])
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(st.session_state['tech_support'])
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(st.session_state['streaming_tv'])
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(st.session_state['streaming_movies'])
        )

    with col3:
        st.markdown('<div class="section-header">💳 Account & Billing</div>', unsafe_allow_html=True)
        
        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(st.session_state['contract'])
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"],
            index=["Yes", "No"].index(st.session_state['paperless'])
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ].index(st.session_state['payment_method'])
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            value=float(st.session_state['monthly_charges']), step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            value=float(st.session_state['total_charges']), step=10.0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🛡️ Evaluate Customer Risk with ChurnGuard AI")

# Evaluation Processing & Results
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

    with st.spinner("🛡️ ChurnGuard AI analyzing profile & constructing LIME explanation..."):
        prediction, probability, explanation = predict_churn(customer_data)

    st.markdown("---")
    st.subheader("🛡️ ChurnGuard Risk Evaluation & Executive Briefing")

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Risk Badge & Gauge
        if prediction == 1:
            st.markdown(
                f'<div class="shield-danger">🚨 CRITICAL RISK: HIGH ATTRITION PROBABILITY</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="shield-safe">🛡️ CUSTOMER SAFE: LOW CHURN RISK</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="Evaluated Churn Probability",
            value=f"{probability * 100:.1f}%",
            delta=f"{(probability - 0.5) * 100:+.1f}% Risk Margin",
            delta_color="inverse"
        )

        st.progress(float(probability))
        
        # ChurnGuard Retention Playbook
        st.markdown("#### 📋 Automated Retention Playbook")
        if probability >= 0.5:
            st.error(
                "**Action Plan: Immediate Intervention Required**\n"
                "• **Contract Upgrade:** Offer a 15% discount on transitioning from Month-to-Month to a 1-Year or 2-Year Contract.\n"
                "• **Service Support:** Provide complimentary Tech Support and Online Security trial for 6 months.\n"
                "• **Billing Incentive:** Waive $5/mo fee for switching to Automatic Bank Transfer / Credit Card billing."
            )
        else:
            st.success(
                "**Action Plan: Growth & Loyalty Enhancement**\n"
                "• **Loyalty Reward:** Eligible for annual customer appreciation bonus.\n"
                "• **Cross-Sell Opportunity:** Offer discounted bundle upgrades for Streaming Services & Fiber speed boosts."
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 LIME Feature Contribution Impact")
        st.caption("Feature drivers computed by LIME (Red = Increases Churn Probability, Green = Lowers Churn Probability)")

        if explanation:
            features = [x[0] for x in explanation[:8]]
            weights = [x[1] for x in explanation[:8]]

            fig, ax = plt.subplots(figsize=(6, 4.2))
            colors = ['#EF4444' if w > 0 else '#10B981' for w in weights]
            
            fig.patch.set_facecolor('#0F172A')
            ax.set_facecolor('#0F172A')
            
            y_pos = range(len(features))
            ax.barh(y_pos, weights, color=colors, height=0.55, edgecolor='none')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, color='#E2E8F0', fontsize=9, fontweight='bold')
            ax.axvline(0, color='#64748B', linewidth=0.8, linestyle='--')
            ax.tick_params(colors='#CBD5E1', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.set_xlabel('LIME Feature Weight Impact', color='#94A3B8', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="dashboard-footer">
        🛡️ ChurnGuard AI Enterprise System • Powered by Machine Learning & Explainable AI
    </div>
""", unsafe_allow_html=True)