# 🧬 NeuroVision AI — Parkinson's Disease Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/ML-Ensemble-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/College-ACE%20Engineering-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **Mini Project** submitted in partial fulfillment of the requirements for the award of the **Bachelor of Technology (B.Tech) in Computer Science and Engineering**
> **ACE Engineering College**, Affiliated to Jawaharlal Nehru Technological University, Hyderabad — *April 2026*

---

## 👨‍💻 Team

| Name | Roll Number |
|---|---|
| R. Satya Hemesh | 23AG1A05B2 |
| K. Naga Harsha | 23AG1A0590 |
| P. Umesh | 23AG1A05A8 |

**Internal Guide:** Mrs. T. Ratnamala, Assistant Professor, Dept. of CSE

---

## 📌 Table of Contents

- [Abstract](#-abstract)
- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Objective & Scope](#-objective--scope)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [ML Models & Ensemble Engine](#-ml-models--ensemble-engine)
- [Project Structure](#-project-structure)
- [Modules](#-modules)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Technical Details](#-technical-details)
- [Testing](#-testing)
- [Performance Evaluation](#-performance-evaluation)
- [Comparison with Existing Systems](#-comparison-with-existing-systems)
- [Outputs & Screenshots](#-outputs--screenshots)
- [Future Enhancements](#-future-enhancements)
- [References](#-references)

---

## 📄 Abstract

The **Parkinson's Disease Detection using Machine Learning & Voice Analysis** project is an innovative solution designed to facilitate the early diagnosis of neurological disorders by analyzing biomedical voice measurements. The project addresses a major limitation in traditional healthcare: the heavy reliance on subjective, expensive, and late-stage clinical evaluations for diagnosing Parkinson's Disease (PD).

Using advanced Machine Learning algorithms and acoustic signal processing, the system extracts critical voice biomarkers — such as fundamental frequency (Fo), Jitter, Shimmer, and Harmonics-to-Noise Ratio (HNR) — to identify the microscopic vocal tremors that precede physical symptoms. The system leverages the **Parselmouth** library for precise feature extraction and utilizes robust ensemble models (Random Forest, SVM, and XGBoost) for highly accurate classification. The **Streamlit** framework integrates the Python-based ML back-end with an interactive web front-end, allowing seamless user interaction and automated PDF report generation.

**Keywords:** Machine Learning, XGBoost, Parselmouth, Acoustic Analysis

---

## 🎯 Motivation & Problem Statement

Parkinson's Disease (PD) is a progressive nervous system disorder affecting movement. Clinical studies indicate that **vocal impairment (dysarthria) affects nearly 90% of PD patients** well before physical tremors become prominent, making voice a critical early biomarker.

Traditional diagnostic methods face significant challenges:
- **Delayed Diagnosis** — Patients are typically evaluated only after severe physical symptoms manifest, missing the early window for neuroprotective therapies.
- **High Diagnostic Costs** — Heavy reliance on expensive neurological hardware (MRI, DaTscans) makes preliminary screening financially inaccessible.
- **Subjectivity** — Standard clinical evaluations (like UPDRS) are inherently subjective and prone to practitioner bias.

This project provides an **automated, objective, non-invasive, and cost-effective** screening mechanism that reliably identifies early disease markers without requiring extensive hospital visits.

---

## 🏆 Objective & Scope

**Objective:** Provide a Machine Learning-based solution for the early detection of Parkinson's Disease using biomedical voice measurements without the need for invasive clinical tests.

**Scope:**
- **Clinical Integration** — Seamlessly integrates into preliminary medical screening workflows, allowing doctors to analyze acoustic data rapidly.
- **Patient Experience** — Offers a non-invasive, stress-free testing method using standard microphones.
- **Cost-Effectiveness** — Eliminates the immediate need for expensive biological scans by running on standard consumer devices.
- **Future Expansion** — Adaptable for mobile applications and integration with Deep Learning networks (CNNs, LSTMs) for even higher precision.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Audio Upload│  │ Manual Entry │  │Live Capture│  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │
└─────────┼───────────────┼────────────────┼──────────┘
          │               │                │
          ▼               ▼                ▼
┌─────────────────────────────────────────────────────┐
│              Feature Extraction Layer                │
│   Praat (parselmouth) → [Fo, Jitter, Shimmer, HNR]  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│           StandardScaler (Normalization)             │
└──────────────────────────┬──────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌──────────┐  ┌──────────────┐
   │Random Forest│  │   SVM    │  │   XGBoost    │
   └──────┬──────┘  └────┬─────┘  └──────┬───────┘
          └──────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │  Soft-Vote Ensemble │
              │  Risk Assessor      │
              └─────────┬───────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   Plotly Dashboard             PDF Clinical Report
```

**Data Flow (DFD Level 1):**
`Voice Sample → Extract Features → ML Model → Result → Admin Dashboard`

Each stage in the pipeline stores data: voice samples, extracted features, and prediction results are logged for admin review.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **File:** `data/parkinsons.data`
- **Records:** 195 voice measurements from 31 subjects (23 with Parkinson's)
- **Original Features:** 23 acoustic attributes
- **Target:** `status` (1 = Parkinson's, 0 = Healthy)

### Selected Biomarkers (4 Features Used)

| Feature | Description |
|---|---|
| `MDVP:Fo(Hz)` | Average vocal fundamental frequency |
| `MDVP:Jitter(%)` | Cycle-to-cycle variation in fundamental frequency (pitch instability) |
| `MDVP:Shimmer` | Cycle-to-cycle variation in amplitude (volume instability) |
| `HNR` | Harmonics-to-Noise Ratio (voice clarity) |

> To download the dataset automatically: `python download_data.py`

---

## 🤖 ML Models & Ensemble Engine

The system uses **soft-voting across three classifiers** — averaging their probability outputs for a consensus-based prediction.

### Random Forest
- `n_estimators`: 100 | `criterion`: entropy | `max_depth`: 15 | `random_state`: 42

### Support Vector Machine (SVM)
- `probability`: True | `random_state`: 42

### XGBoost
- `eval_metric`: logloss | `random_state`: 42

### Risk Assessment Logic

| Probability | Risk Level | Clinical Meaning |
|---|---|---|
| < 40% | ✅ Low Risk | Biomarkers align with healthy control group |
| 40% – 75% | ⚠️ Medium Risk | Subtle vocal impairments detected; follow-up recommended |
| > 75% | 🔴 High Risk | Strong indicators of Parkinsonian dysphonia detected |

---

## 📁 Project Structure

```
Parkinson_Disease_ML/
│
├── app.py                    # Main Streamlit application (UI + prediction pipeline)
├── train_pipeline.py         # ML training orchestration script
├── download_data.py          # UCI dataset auto-downloader
├── requirements.txt          # Python dependencies
│
├── data/
│   └── parkinsons.data       # UCI Parkinson's voice dataset (195 samples, 24 cols)
│
├── models/                   # Saved model artifacts (auto-generated by train_pipeline.py)
│   ├── rf_model.joblib       # Trained Random Forest
│   ├── svm_model.joblib      # Trained SVM
│   ├── xgb_model.joblib      # Trained XGBoost
│   ├── scaler.joblib         # Fitted StandardScaler
│   └── feature_names.csv     # Feature list used during training
│
└── src/
    ├── __init__.py
    ├── config.py             # Centralized paths, hyperparameters, feature list
    ├── preprocessing.py      # DataHandler: load, split, scale data
    ├── model_engine.py       # ParkinsonPredictor: train, evaluate, save
    ├── audio_processor.py    # Praat-based voice feature extraction
    └── risk_assessor.py      # Maps probability to risk label + color
```

---

## 🧩 Modules

### 1. Admin Module
Secure login giving access to the Central Command Hub, where all patient diagnostic sessions are tracked in real-time. Admins can manage datasets and trigger ML retraining pipelines.

### 2. User (Patient/Clinician) Module
Provides the diagnostic interface with three input methods: audio file upload, manual biomarker entry, and live in-browser voice capture.

### 3. Audio Feature Extraction Module (`src/audio_processor.py`)
Uses **Praat via Parselmouth** to extract:
- Mean Pitch (Fo) via pitch analysis
- Local Jitter via PointProcess (periodic, cc) method
- Local Shimmer via cross-correlation method
- HNR via autocorrelation harmonicity

### 4. Machine Learning Prediction Module
- Scales incoming features using a pre-fitted `StandardScaler`
- Passes data through all three pre-trained models
- Applies soft-vote probability averaging with 0.5 threshold for binary classification

### 5. Report Generation Module
Uses `fpdf` to auto-generate a clinical PDF containing patient context, ensemble confidence breakdown, and all biomarker values.

### 6. Configuration Module (`src/config.py`)
Centralizes all file paths, hyperparameters, and the feature list (`SELECTED_FEATURES`).

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher (developed on 3.12)
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/SatyaHemesh/Parkinson-s-Disease-Detection.git
cd Parkinson-s-Disease-Detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **macOS:** `brew install libsndfile` may be required.
> **Linux:** `sudo apt-get install libsndfile1` may be required.

### 4. Download the Dataset

```bash
python download_data.py
```

### Hardware Requirements
- **Processor:** i3 / Intel or higher
- **RAM:** 8 GB minimum (recommended for ML processing)
- **Storage:** 160 GB minimum

### Software Requirements
- **OS:** Windows 10/11 or Linux
- **IDE:** VS Code / PyCharm / Jupyter Notebook
- **Language:** Python 3.8+

---

## 🚀 Running the Application

### Step 1 — Train the Models (First Time Only)

Pre-trained model files are already included in `models/`. To retrain from scratch:

```bash
python train_pipeline.py
```

Expected output:
```
🚀 Starting Parkinson's Detection System Training Pipeline...
✅ Data loaded successfully. Shape: (195, 24)
✅ Random Forest training complete.
✅ SVM training complete.
✅ XGBoost training complete.
📊 System Accuracy: XX.XX%
✅ All 3 models saved successfully to the models/ folder.
✅ Feature names saved.
```

### Step 2 — Launch the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Troubleshooting

| Issue | Fix |
|---|---|
| Model not loading | Ensure `.joblib` files exist in `models/` — run `train_pipeline.py` |
| Audio not working | `pip install librosa soundfile` |
| Streamlit not running | `pip install streamlit` |

---

## 🖥️ Usage Guide

### Sidebar — Patient Context
Set the patient's **MRN**, **Age**, and **Biological Sex** before running any diagnostic. These attach to every session record.

### Tab 1 — 🎙️ Audio Upload
1. Upload a `.wav` voice recording (limit: 200MB)
2. Acoustic Mel-Spectrogram renders automatically
3. Click **Analyze Uploaded Audio**

### Tab 2 — 📝 Manual Entry
1. Enter the four acoustic biomarker values directly
2. Click **Initialize Neural Pipeline**

### Tab 3 — 🎤 Live Capture
1. Instruct patient to sustain vowel "Aaaa" for 3–5 seconds
2. Click the microphone button to record
3. Click **Analyze Live Recording**

### Results Panel
- Color-coded clinical risk label
- Interactive Plotly AI Confidence Score gauge
- Per-model probability bars (RF, SVM, XGBoost)
- **Download Official PDF Report** button

### 🔐 Admin Portal
```
Admin Badge ID : admin
Secure Passcode: admin123
```
> ⚠️ Change these before any shared or production deployment.

---

## 🛠️ Technical Details

### Technologies Used

| Layer | Technology |
|---|---|
| Frontend | Streamlit, HTML/CSS (embedded) |
| Backend | Python 3.8+ |
| ML | scikit-learn, XGBoost, joblib |
| Audio | librosa, parselmouth (Praat) |
| Visualization | Plotly, matplotlib |
| Report | fpdf |
| Live Audio | audio-recorder-streamlit |
| Data | pandas, numpy |

### Key Concepts

- **Acoustic Signal Processing** — Extracts neurological biomarkers (Jitter, Shimmer, HNR) from `.wav` audio and renders Mel-Spectrograms via librosa.
- **Soft-Vote Ensemble** — Averages `predict_proba` outputs across RF, SVM, and XGBoost for a consensus-based, bias-reduced confidence score.
- **Stateless Session Management** — Uses Streamlit's `session_state` for patient record tracking; zero PII is retained after the session closes.
- **Real-Time Visualization** — Plotly gauge charts transform raw probabilities into clinically interpretable visuals.
- **Dynamic PDF Reports** — `fpdf` compiles biomarker data, model consensus, and risk label into a downloadable clinical report.

---

## 🧪 Testing

Both **Black Box** and **White Box** testing methodologies were applied.

### Black Box Test Cases

| Test ID | Test Name | Objective | Status |
|---|---|---|---|
| TC-01 | Upload Voice Sample | Verify `.wav` upload works correctly | ✅ Pass |
| TC-02 | Predict Parkinson Disease | Verify system prediction correctness | ✅ Pass |
| TC-03 | Download Report | Verify PDF report download | ✅ Pass |

### White Box Test Cases

| Test ID | Test Name | Objective | Status |
|---|---|---|---|
| TC-04 | Feature Extraction | Verify Praat extracts pitch, jitter, shimmer correctly | ✅ Pass |
| TC-05 | Model Prediction Logic | Verify ML model predicts Positive/Negative correctly | ✅ Pass |
| TC-06 | Generate Report | Verify fpdf generates a valid downloadable PDF | ✅ Pass |

### Testing Techniques
- **Manual Testing** — UI flow, upload, prediction, and result display verified
- **Automated Testing** — Python unit validation for model inference functions
- **Performance Testing** — System tested with multiple inputs to ensure fast prediction
- **Security Testing** — File upload validation; unauthorized admin access blocked

---

## 📈 Performance Evaluation

### Metrics Used

| Metric | Description |
|---|---|
| Accuracy | Proportion of correct predictions overall |
| Precision | Of predicted positives, how many are truly positive |
| Recall | Of actual positives, how many were correctly identified |
| F1-Score | Harmonic mean of Precision and Recall |
| Latency | Time taken to process input and return prediction |

### Evaluation Code

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Replace with actual model outputs
y_true = [1, 0, 1, 1, 0]
y_pred  = [1, 0, 1, 0, 0]

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
```

---

## ⚖️ Comparison with Existing Systems

| Feature | NeuroVision (Proposed) | Existing Systems |
|---|---|---|
| **Cost** | Cost-effective (software-based ML) | Expensive (MRI, DaTscans, lab tests) |
| **Accuracy** | High (RF + SVM + XGBoost ensemble) | Moderate (manual diagnosis varies) |
| **Real-Time Performance** | Fast (instant Streamlit prediction) | Slow (hospital visits, lab processing) |
| **Scalability** | Highly scalable (web-based) | Complex and time-consuming |
| **User Experience** | Easy, user-friendly interface | Complex clinical process |
| **Invasiveness** | Non-invasive (voice recording only) | Invasive or requires specialized equipment |

---

## 🖼️ Outputs & Screenshots

The application produces the following key screens during a session:

1. **System Overview Page** — Hero banner with three capability cards (Signal Processing, Ensemble Engine, Enterprise Security)
2. **Audio Upload Page** — Drag-and-drop `.wav` uploader with Mel-Spectrogram visualization
3. **Manual Entry Page** — Direct biomarker input form with 4 numeric fields
4. **Live Voice Capture Page** — In-browser microphone recording interface
5. **Admin Authentication Page** — Secure login form (Badge ID + Passcode)
6. **Admin Dashboard (Empty)** — Central Command Hub before any diagnostics are run
7. **Diagnostic Result — Low Risk** — Gauge at ~27.1%, green clinical label
8. **Diagnostic Result — Medium Risk** — Gauge at ~44.6%, orange clinical label
9. **Diagnostic Result — High Risk** — Gauge at ~88.9%, red clinical label with visual alerts
10. **Acoustic Spectrogram** — Mel-Spectrogram visualization of uploaded audio
11. **Admin Dashboard (After Testing)** — Session table showing MRN, age, gender, risk level, and confidence for all diagnostics run

---

## 🔮 Future Enhancements

1. **Deep Learning Integration** — CNN or LSTM models to process raw audio spectrograms directly, eliminating manual feature extraction
2. **Mobile App Development** — Android/iOS applications for remote patient voice capture and monitoring
3. **Real-Time Monitoring** — Continuous tracking of vocal degradation over time via historical session data
4. **Hospital System Integration** — Direct connection with hospital management systems and doctor dashboards
5. **Cloud Storage** — Secure cloud persistence for patient records and longitudinal tracking
6. **Multi-Language Support** — Interface localization for broader patient accessibility
7. **Security Enhancements** — Data encryption for patient PII and formal compliance hardening
8. **IoT / Wearable Integration** — Collect additional health data from smart devices to improve prediction accuracy

---

## 📚 References

1. Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A., & Moroz, I.M. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.*
2. Tsanas, A., Little, M.A., McSharry, P.E., & Ramig, L.O. (2010). *Accurate Telemonitoring of Parkinson's Disease Progression Using Nonlinear Speech Signal Processing.*
3. Sakar, B.E., et al. (2013). *Collection and Analysis of a Parkinson Speech Dataset with Multiple Types of Sound Recordings.*
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
6. Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks.*
7. Breiman, L. (2001). *Random Forests.*
8. McKinney, W. (2010). *Data Structures for Statistical Computing in Python (Pandas).*
9. McFee, B., et al. (2015). *Librosa: Audio and Music Signal Analysis in Python.*
10. Streamlit Inc. (2020). *Streamlit: The Fastest Way to Build Data Apps.*

---

## 📦 Dependencies

```
streamlit                 # Web application framework
numpy                     # Numerical computing
pandas                    # Data manipulation
scikit-learn              # ML algorithms and preprocessing
matplotlib                # Static plotting
seaborn                   # Statistical visualization
librosa                   # Audio signal processing & Mel-Spectrograms
soundfile                 # Audio file I/O
numba                     # JIT compilation (librosa dependency)
xgboost                   # Extreme Gradient Boosting
praat-parselmouth          # Praat acoustic feature extraction
joblib                    # Model serialization/deserialization
scipy                     # Scientific computing
audio-recorder-streamlit  # In-browser live audio capture
fpdf                      # PDF clinical report generation
plotly                    # Interactive gauge and confidence charts
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

> This software is developed as an **academic mini-project** for educational purposes only. It is **not a certified medical device** and should not be used as a substitute for professional clinical diagnosis. All results must be reviewed and validated by a licensed medical professional before any clinical decisions are made.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <strong>NeuroVision Clinical Software © 2026</strong><br>
  Developed by <strong>Team 17</strong> — R. Satya Hemesh · K. Naga Harsha · P. Umesh<br>
  Department of Computer Science and Engineering, ACE Engineering College
</p>
