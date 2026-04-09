import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from fpdf import FPDF
from datetime import datetime

# High-end visualization & audio libraries
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Local imports
from src.risk_assessor import RiskAssessor
import src.config as config
from src.audio_processor import extract_voice_features

# Safe audio import
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None

# ==========================================
# PAGE CONFIG & RESPONSIVE CSS
# ==========================================
st.set_page_config(page_title="NeuroVision AI | Parkinson's Diagnostic", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main-title { font-size: 44px; font-weight: 800; color: var(--text-color); letter-spacing: -1px; margin-bottom: 0px; }
    .sub-text { font-size: 16px; color: var(--text-color); opacity: 0.7; margin-top: -5px; margin-bottom: 25px; letter-spacing: 0.5px;}
    .hero-container { background: linear-gradient(135deg, #0D6EFD 0%, #00E5FF 100%); padding: 50px 40px; border-radius: 20px; text-align: center; margin-bottom: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); color: white; }
    .hero-title { font-size: 42px; font-weight: 800; margin-bottom: 15px; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .info-card { background: var(--secondary-background-color); padding: 25px; border-radius: 16px; border: 1px solid var(--primary-color); text-align: center; height: 100%; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.05); color: var(--text-color); }
    .stButton>button { background: transparent; color: var(--primary-color); border: 2px solid var(--primary-color); border-radius: 8px; font-weight: 600; transition: all 0.3s ease-in-out; width: 100%; }
    .stButton>button:hover { background: var(--primary-color); color: var(--background-color); }
    .footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: var(--secondary-background-color); color: var(--text-color); text-align: center; padding: 15px 0; font-size: 13px; border-top: 1px solid var(--primary-color); z-index: 100; }
    .footer span { color: var(--primary-color); font-weight: 800; }
    .block-container { padding-bottom: 100px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE & DATABASE MANAGEMENT
# ==========================================
if 'patient_db' not in st.session_state:
    st.session_state['patient_db'] = pd.DataFrame(columns=["Timestamp", "Patient ID", "Age", "Gender", "Risk Level", "Confidence"])
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ==========================================
# HELPER FUNCTIONS & VISUALS
# ==========================================
def get_ensemble_prediction(scaled_data, loaded_models):
    model_probs = {name: model.predict_proba(scaled_data)[0][1] for name, model in loaded_models.items()}
    avg_prob = np.mean(list(model_probs.values()))
    final_pred = 1 if avg_prob >= 0.5 else 0
    return final_pred, avg_prob, model_probs

def plot_plotly_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = prob * 100, domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Confidence Score", 'font': {'size': 20}}, number = {'suffix': "%", 'font': {'size': 36}},
        gauge = {'axis': {'range': [0, 100], 'tickwidth': 1}, 'bar': {'color': "#0D6EFD"}, 'bgcolor': "rgba(0,0,0,0.1)", 'borderwidth': 0,
            'steps': [{'range': [0, 30], 'color': "rgba(16, 185, 129, 0.3)"}, {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.3)"}, {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': "Plus Jakarta Sans"}, height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
    ax.set_title('Acoustic Mel-Spectrogram Analysis', fontsize=12, color='gray')
    return fig

def generate_pdf_report(risk_label, probability, input_data_dict, model_probs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Clinical Diagnostic Report - Parkinson's Assessment", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, "Generated by: NeuroVision Diagnostic AI", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, f"System Diagnosis: {risk_label} ({probability*100:.2f}%)", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    for name, prob in model_probs.items(): pdf.cell(200, 8, f"  - {name} Module: {prob*100:.2f}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Biomarkers Analyzed:", ln=True)
    pdf.set_font("Arial", '', 10)
    for k, v in input_data_dict.items(): pdf.cell(200, 6, f"{k}: {v}", ln=True)
    filename = "NeuroVision_PD_Report.pdf"
    pdf.output(filename)
    return filename

def download_link(file):
    with open(file, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    return f'<a href="data:file/pdf;base64,{b64}" download="{file}" style="background-color:#0D6EFD; color:white; padding:10px 20px; text-decoration:none; border-radius:6px; font-weight:bold; display:block; text-align:center; margin-top:15px; width:100%;">📄 Download Official PDF Report</a>'

# ==========================================
# REUSABLE PREDICTION PIPELINE
# ==========================================
def process_and_predict(feature_dict, patient_id, patient_age, patient_gender, models, scaler):
    df = pd.DataFrame([feature_dict])
    # Ensure columns match training order exactly
    df = df[config.SELECTED_FEATURES] 
    
    scaled = scaler.transform(df)
    pred, prob, model_probs = get_ensemble_prediction(scaled, models)
    risk, color, explanation = RiskAssessor.calculate_risk_score(prob)
    
    new_record = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Patient ID": patient_id, "Age": patient_age, "Gender": patient_gender, "Risk Level": risk, "Confidence": f"{prob*100:.1f}%"}])
    st.session_state['patient_db'] = pd.concat([st.session_state['patient_db'], new_record], ignore_index=True) if not st.session_state['patient_db'].empty else new_record

    st.divider()
    st.markdown(f"<h2 style='color:{color}; text-align:center;'>Clinical Indication: {risk}</h2>", unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns([1.5, 1])
    with r_col1: st.plotly_chart(plot_plotly_gauge(prob), use_container_width=True)
    with r_col2:
        st.markdown("### Model Consensus")
        for name, p in model_probs.items(): st.progress(float(p), text=f"{name}: {p*100:.1f}%")
        pdf_file = generate_pdf_report(risk, prob, feature_dict, model_probs)
        st.markdown(download_link(pdf_file), unsafe_allow_html=True)

# ==========================================
# LOAD MODELS & ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        models = {
            "Random Forest": joblib.load(config.RF_MODEL_PATH),
            "SVM": joblib.load(config.SVM_MODEL_PATH),
            "XGBoost": joblib.load(config.XGB_MODEL_PATH)
        }
        scaler = joblib.load(config.SCALER_SAVE_PATH)
        return models, scaler, config.SELECTED_FEATURES
    except Exception as e:
        return None, None, None

models, scaler, feature_names = load_assets()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown(f'<p style="font-size:28px; font-weight:800; color:#0D6EFD; margin-bottom: 0;">🧬 NeuroVision</p>', unsafe_allow_html=True)
st.sidebar.write("---")
nav = st.sidebar.radio("SYSTEM NAVIGATION", ["System Overview", "Clinical Diagnostics", "Admin Portal"])
st.sidebar.write("---")
st.sidebar.markdown('<p style="color:#0D6EFD; font-weight:bold;">Active Patient Context</p>', unsafe_allow_html=True)
patient_id = st.sidebar.text_input("Patient ID (MRN)", value="PT-10042")
patient_age = st.sidebar.slider("Patient Age", 30, 90, 65)
patient_gender = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])

# ==========================================
# PAGES
# ==========================================
if nav == "System Overview":
    st.markdown("<div class='hero-container'><div class='hero-title'>High-Fidelity Neurological Diagnostics</div><p style='font-size:20px; max-width:800px; margin:0 auto;'>Leveraging multi-model ensemble intelligence and acoustic signal processing to detect vocal tremors associated with Parkinson's Disease.</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("<div class='info-card'><h1 style='font-size: 40px;'>📊</h1><h3 style='margin-bottom:5px;'>Signal Processing</h3><p style='font-size:14px;'>Transforms raw .wav data into Mel-Spectrograms and acoustic features.</p></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='info-card'><h1 style='font-size: 40px;'>🧠</h1><h3 style='margin-bottom:5px;'>Ensemble Engine</h3><p style='font-size:14px;'>Runs soft-voting across Random Forest, SVM, and XGBoost.</p></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='info-card'><h1 style='font-size: 40px;'>🔐</h1><h3 style='margin-bottom:5px;'>Enterprise Security</h3><p style='font-size:14px;'>Stateless processing ensures zero patient PII is retained.</p></div>", unsafe_allow_html=True)

elif nav == "Clinical Diagnostics":
    st.markdown('<p class="main-title">🩺 Diagnostic Interface</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-text">Analyzing data for Patient MRN: <strong>{patient_id}</strong></p>', unsafe_allow_html=True)
    
    if models is None or scaler is None:
        st.error("System Offline: ML Weights missing. Please run `python train_pipeline.py` first.")
    else:
        # ALL THREE TABS RESTORED
        tab1, tab2, tab3 = st.tabs(["🎙️ Audio Upload", "📝 Manual Entry", "🎤 Live Capture"])

        # TAB 1: AUDIO UPLOAD
        with tab1:
            st.markdown("### Upload Patient Recording")
            uploaded_audio = st.file_uploader("Upload .wav file", type=["wav"], label_visibility="collapsed")
            if uploaded_audio:
                temp_path = "temp_upload.wav"
                with open(temp_path, "wb") as f: f.write(uploaded_audio.read())
                
                st.audio(temp_path)
                st.pyplot(plot_spectrogram(temp_path))

                if st.button("Analyze Uploaded Audio", key="btn_up"):
                    with st.spinner("Extracting Acoustic Biomarkers..."):
                        features = extract_voice_features(temp_path)
                        if features:
                            # Map Praat outputs to our model features
                            feature_dict = {
                                'MDVP:Fo(Hz)': features[0],
                                'MDVP:Jitter(%)': features[1],
                                'MDVP:Shimmer': features[2],
                                'HNR': features[3]
                            }
                            process_and_predict(feature_dict, patient_id, patient_age, patient_gender, models, scaler)
                        else:
                            st.error("Failed to extract features from this audio file.")
                if os.path.exists(temp_path): os.remove(temp_path)

        # TAB 2: MANUAL ENTRY
        with tab2:
            st.markdown("### Direct Biomarker Entry")
            input_data = {}
            with st.form("manual_form", border=False):
                cols = st.columns(4)
                for i, f in enumerate(feature_names):
                    with cols[i % 4]: input_data[f] = st.number_input(f, value=0.0000, format="%.5f")
                submit = st.form_submit_button("Initialize Neural Pipeline", use_container_width=True)
            if submit:
                process_and_predict(input_data, patient_id, patient_age, patient_gender, models, scaler)

        # TAB 3: LIVE RECORDING
        with tab3:
            st.markdown("### Direct Clinical Capture")
            if audio_recorder:
                st.info("Instruct the patient to sustain the vowel 'Aaaa' for 3-5 seconds.")
                audio_bytes = audio_recorder(text="Click to Record", recording_color="#EF4444", neutral_color="#0D6EFD")
                if audio_bytes:
                    temp_live = "temp_live.wav"
                    with open(temp_live, "wb") as f: f.write(audio_bytes)
                    st.audio(temp_live)
                    
                    if st.button("Analyze Live Recording", key="btn_live"):
                        with st.spinner("Processing..."):
                            features = extract_voice_features(temp_live)
                            if features:
                                feature_dict = {
                                    'MDVP:Fo(Hz)': features[0],
                                    'MDVP:Jitter(%)': features[1],
                                    'MDVP:Shimmer': features[2],
                                    'HNR': features[3]
                                }
                                process_and_predict(feature_dict, patient_id, patient_age, patient_gender, models, scaler)
                            else:
                                st.error("Audio capture failed or was too noisy.")
                    if os.path.exists(temp_live): os.remove(temp_live)
            else:
                st.warning("Audio recorder module not detected. Please install it via requirements.")

elif nav == "Admin Portal":
    if not st.session_state['logged_in']:
        st.markdown("<br><br><h2 style='text-align: center;'>🔐 System Authentication</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1.2, 1]) 
        with col2:
            with st.form("login_form", border=True):
                user = st.text_input("Admin Badge ID")
                pwd = st.text_input("Secure Passcode", type="password")
                if st.form_submit_button("Authenticate", use_container_width=True):
                    if user == "admin" and pwd == "admin123": st.session_state['logged_in'] = True; st.rerun()
                    else: st.error("Authentication Denied.")
    else:
        st.header("⚙️ Central Command Hub")
        if st.button("End Secure Session"): st.session_state['logged_in'] = False; st.rerun()
        if not st.session_state['patient_db'].empty: st.dataframe(st.session_state['patient_db'], use_container_width=True)
        else: st.info("No patient diagnostics have been run in this session yet.")

st.markdown("<div class='footer'>NeuroVision Clinical Software © 2026 | Developed by <span>Team 17</span></div>", unsafe_allow_html=True)