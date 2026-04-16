"""
Laptop Spec Tier Classifier — Streamlit App
============================================
Classifies laptops into Basic / Mainstream / Powerhouse tiers
using a Soft-Voting Ensemble of XGBoost + Random Forest.

Deploy on Hugging Face Spaces (streamlit SDK) or run locally with:
    streamlit run app.py

Folder layout expected:
    app.py
    data/laptops.csv        ← place dataset here
    requirements.txt
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Spec Tier Classifier",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0f1117; }
.stApp { background-color: #0f1117; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2130, #252840);
    border: 1px solid #3b3f5c;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
}

.tier-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 40px;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 8px 0;
}
.tier-basic      { background:#1e3a5f; color:#60a5fa; border:2px solid #3b82f6; }
.tier-mainstream { background:#3d2a00; color:#fbbf24; border:2px solid #f59e0b; }
.tier-powerhouse { background:#064e3b; color:#34d399; border:2px solid #10b981; }

.prob-bar-wrap { margin: 6px 0; }
.prob-label    { font-size:.85rem; color:#9ca3af; margin-bottom:2px; }

.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e5e7eb;
    border-left: 4px solid #6366f1;
    padding-left: 12px;
    margin: 24px 0 12px;
}

[data-testid="stSidebar"] { background: #141625; }

div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px;
    font-size: 1.05rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity .2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS
# ─────────────────────────────────────────────
def parse_storage(mem_str):
    """Parse a raw storage string into (total_gb, storage_type, speed_score)."""
    mem_str = str(mem_str).upper()
    total_gb = 0
    for v in pd.Series([mem_str]).str.findall(r'(\d+)TB')[0]:
        total_gb += int(v) * 1024
    for v in pd.Series([mem_str]).str.findall(r'(\d+)GB')[0]:
        total_gb += int(v)
    if 'NVME' in mem_str or 'PCIE' in mem_str:
        return total_gb, 'NVMe', 4
    elif 'SSD' in mem_str:
        return total_gb, 'SSD', 3
    elif 'FUSION' in mem_str:
        return total_gb, 'Fusion', 2
    elif 'HDD' in mem_str:
        return total_gb, 'HDD', 1
    elif 'SSHD' in mem_str:
        return total_gb, 'SSHD', 2
    return total_gb, 'Flash', 1


def cpu_tier(c):
    """Map CPU string to ordinal tier 0–3."""
    c = str(c).upper()
    if 'XEON' in c or 'I7' in c:
        return 3
    elif 'I5' in c:
        return 2
    elif 'I3' in c:
        return 1
    return 0


def gpu_tier(g):
    """Map GPU string to ordinal tier 0–2."""
    g = str(g).upper()
    if 'RX 5600' in g:
        return 2
    if 'GTX 1650' in g:
        return 1
    return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to a raw laptop DataFrame.
    Works on both the full training set and single-row prediction inputs.
    """
    df = df.copy()

    # Core specs
    df['Ram_GB']    = df['Ram'].astype(str).str.extract(r'(\d+)').astype(int)
    df['Weight_kg'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['Inches']    = pd.to_numeric(df['Inches'], errors='coerce')

    # Storage
    storage_parsed     = df['Memory'].apply(lambda x: pd.Series(parse_storage(x)))
    df['Storage_GB']   = storage_parsed[0]
    df['StorageType']  = storage_parsed[1]
    df['StorageSpeed'] = storage_parsed[2]

    # CPU
    df['CPU_Tier']  = df['Cpu'].apply(cpu_tier)
    df['CPU_Brand'] = df['Cpu'].apply(lambda x: 'AMD' if 'AMD' in str(x) else 'Intel')

    # GPU
    df['GPU_Tier']  = df['Gpu'].apply(gpu_tier)
    df['GPU_Brand'] = df['Gpu'].apply(
        lambda x: 'Nvidia' if 'NVIDIA' in str(x).upper()
        else ('AMD' if 'AMD' in str(x) else 'Intel')
    )

    # Screen flags
    sr = df['ScreenResolution'].astype(str)
    df['Is_Touchscreen'] = sr.str.contains('Touchscreen').astype(int)
    df['Is_IPS']         = sr.str.contains('IPS').astype(int)
    df['Is_4K']          = sr.str.upper().str.contains('4K|3840').astype(int)
    df['Is_2K']          = sr.str.upper().str.contains('2K|2560').astype(int)
    df['Is_Retina']      = sr.str.contains('Retina').astype(int)

    # Brand / form-factor flags
    df['IsPremiumBrand']  = df['CompanyName'].isin(['Apple', 'Microsoft', 'Dell']).astype(int)
    df['IsGaming']        = df['TypeOfLaptop'].str.contains('Gaming').astype(int)
    df['IsWorkstation']   = df['TypeOfLaptop'].str.contains('WorkStation').astype(int)
    df['IsConvertible']   = df['TypeOfLaptop'].str.contains('Convertible').astype(int)
    df['IsUltrabook']     = df['TypeOfLaptop'].str.contains('UltraBook').astype(int)

    # Composite scores
    df['GPU_Tier_Score'] = df['GPU_Tier'].map({0: 1.0, 1: 1.5, 2: 2.2})
    df['PowerScore']     = (df['Ram_GB'] * (df['CPU_Tier'] + 1) * df['GPU_Tier_Score']).round(2)
    df['SpecScore']      = (
        df['Ram_GB'] * df['StorageSpeed'] *
        (df['GPU_Tier'] + 1) * (df['CPU_Tier'] + 1)
    ).astype(float)

    return df


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'Ram_GB', 'Storage_GB', 'StorageSpeed', 'CPU_Tier', 'GPU_Tier',
    'Is_Touchscreen', 'Is_IPS', 'Is_4K', 'Is_2K', 'Is_Retina',
    'Inches', 'Weight_kg',
    'IsPremiumBrand', 'IsGaming', 'IsWorkstation', 'IsConvertible', 'IsUltrabook',
    'PowerScore',
    'TypeOfLaptop', 'StorageType', 'GPU_Brand', 'CPU_Brand', 'OpSys', 'CompanyName',
]
CAT_COLS = ['TypeOfLaptop', 'StorageType', 'GPU_Brand', 'CPU_Brand', 'OpSys', 'CompanyName']

TIER_COLOR = {
    'Basic':      ('#3b82f6', 'tier-basic'),
    'Mainstream': ('#f59e0b', 'tier-mainstream'),
    'Powerhouse': ('#10b981', 'tier-powerhouse'),
}


# ─────────────────────────────────────────────
# LOAD & TRAIN  (cached — runs only once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training model — please wait…")
def load_and_train():
    # Support both local dev (data/) and HF Space (root) paths
    csv_path = "data/laptops.csv" if os.path.exists("data/laptops.csv") else "laptops.csv"
    df_raw = pd.read_csv(csv_path)
    df = engineer_features(df_raw)

    # Create tier labels
    df['SpecTier'] = pd.qcut(
        df['SpecScore'], q=3,
        labels=['Basic', 'Mainstream', 'Powerhouse'],
        duplicates='drop'
    )

    df_model = df[FEATURE_COLS + ['SpecTier']].copy().dropna(subset=['SpecTier'])
    df_model = pd.get_dummies(df_model, columns=CAT_COLS, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(df_model['SpecTier'].astype(str))
    X = df_model.drop('SpecTier', axis=1).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models
    xgb_clf = XGBClassifier(
        n_estimators=300, learning_rate=0.10, max_depth=6,
        subsample=0.85, colsample_bytree=0.80, min_child_weight=2,
        gamma=0.05, reg_alpha=0.1, reg_lambda=1.2,
        random_state=42, eval_metric='mlogloss',
    )
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf)],
        voting='soft',
    )
    ensemble.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)  # standalone for feature importances

    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_,
        zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring='accuracy')

    importances = pd.Series(xgb_clf.feature_importances_, index=X.columns)

    return {
        "model":        ensemble,
        "le":           le,
        "feature_cols": list(X.columns),
        "acc":          acc,
        "cv_acc":       cv_acc,
        "report":       report,
        "cm":           cm,
        "df":           df,
        "importances":  importances,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_pred":       y_pred,
    }


def predict_single(state: dict, user_row: dict) -> tuple:
    """Predict tier + probabilities for a single user-supplied laptop dict."""
    row_df  = pd.DataFrame([user_row])
    row_enc = pd.get_dummies(row_df, columns=CAT_COLS, drop_first=True)

    for col in state["feature_cols"]:
        if col not in row_enc.columns:
            row_enc[col] = 0
    row_enc = row_enc[state["feature_cols"]].fillna(0)

    proba    = state["model"].predict_proba(row_enc)[0]
    pred_idx = np.argmax(proba)
    tier     = state["le"].inverse_transform([pred_idx])[0]
    return tier, proba


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
state = load_and_train()


# ─────────────────────────────────────────────
# SIDEBAR — PREDICTION INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/laptop.png", width=64)
    st.title("💻 Laptop Classifier")
    st.caption("Spec Tier: Basic · Mainstream · Powerhouse")
    st.divider()
    st.markdown("### 🔮 Predict a Laptop")

    company     = st.selectbox("Brand", ["MSI","Chuwi","hp","Microsoft","Apple","lenevo","Asus","Acer","Dell"])
    laptop_type = st.selectbox("Laptop Type", ["Business Laptop","2 in 1 Convertible","WorkStation","Gaming","NoteBook","UltraBook"])
    ram         = st.selectbox("RAM", ["4GB","8GB","12GB","16GB"])
    memory      = st.selectbox("Storage", [
        "128GB SSD","256GB SSD","512GB SSD","128GB PCIe SSD","256GB PCIe SSD",
        "512GB NVMe SSD","1TB NVMe SSD","2TB NVMe SSD","1TB HDD","2TB HDD",
        "4TB HDD","6TB HDD","1TB Fusion Drive","256GB Flash Storage",
        "256GB eMMC","512GB eMMC","1TB SSHD"
    ])
    gpu         = st.selectbox("GPU", ["Intel Iris Xe Graphics","NVIDIA GeForce GTX 1650","AMD Radeon RX 5600M"])
    cpu         = st.selectbox("CPU", ["Intel Core i7","Intel Core i5","Intel Core i3",
                                        "Intel Xeon E3-1505M","Intel Atom x5-Z8550","Intel Celeron Dual Core 3855U"])
    screen_res  = st.selectbox("Screen", ["Full HD","2K","4K","HD 1920x1080 ",
                                           "IPS Panel Retina Display 2560x1600",
                                           "IPS Panel Full HD / Touchscreen 1920x1080"])
    os_sel      = st.selectbox("OS", ["Windows 11","Windows 10","macOS","Linux","No OS"])
    inches      = st.slider("Screen Size (in)", 11.0, 18.0, 15.6, 0.1)
    weight      = st.slider("Weight (kg)", 2.0, 5.0, 2.5, 0.1)
    predict_btn = st.button("⚡ Predict Spec Tier")


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("💻 Laptop Spec Tier Classifier")
st.caption("XGBoost + Random Forest Ensemble · Classifies: Basic / Mainstream / Powerhouse")

m1, m2, m3, m4 = st.columns(4)
m1.metric("🎯 Test Accuracy",    f"{state['acc']*100:.2f}%")
m2.metric("📊 CV Accuracy",      f"{state['cv_acc'].mean()*100:.2f}%")
m3.metric("📈 CV Std Dev",       f"±{state['cv_acc'].std()*100:.2f}%")
m4.metric("🗂️ Training Samples", f"{len(state['df'])}")

st.divider()

# ── Prediction Result ────────────────────────
if predict_btn:
    row = {
        "Ram": ram, "Memory": memory, "Gpu": gpu, "Cpu": cpu,
        "ScreenResolution": screen_res, "OpSys": os_sel,
        "TypeOfLaptop": laptop_type, "CompanyName": company,
        "Inches": inches, "Weight": weight,
    }
    row_df = engineer_features(pd.DataFrame([row]))
    feature_map = {
        'Ram_GB':        row_df['Ram_GB'].iloc[0],
        'Storage_GB':    row_df['Storage_GB'].iloc[0],
        'StorageSpeed':  row_df['StorageSpeed'].iloc[0],
        'CPU_Tier':      row_df['CPU_Tier'].iloc[0],
        'GPU_Tier':      row_df['GPU_Tier'].iloc[0],
        'Is_Touchscreen':row_df['Is_Touchscreen'].iloc[0],
        'Is_IPS':        row_df['Is_IPS'].iloc[0],
        'Is_4K':         row_df['Is_4K'].iloc[0],
        'Is_2K':         row_df['Is_2K'].iloc[0],
        'Is_Retina':     row_df['Is_Retina'].iloc[0],
        'Inches':        inches,
        'Weight_kg':     weight,
        'IsPremiumBrand':row_df['IsPremiumBrand'].iloc[0],
        'IsGaming':      row_df['IsGaming'].iloc[0],
        'IsWorkstation': row_df['IsWorkstation'].iloc[0],
        'IsConvertible': row_df['IsConvertible'].iloc[0],
        'IsUltrabook':   row_df['IsUltrabook'].iloc[0],
        'PowerScore':    row_df['PowerScore'].iloc[0],
        'TypeOfLaptop':  laptop_type,
        'StorageType':   row_df['StorageType'].iloc[0],
        'GPU_Brand':     row_df['GPU_Brand'].iloc[0],
        'CPU_Brand':     row_df['CPU_Brand'].iloc[0],
        'OpSys':         os_sel,
        'CompanyName':   company,
    }
    tier, proba = predict_single(state, feature_map)
    color_hex, css_class = TIER_COLOR[tier]

    st.markdown("### 🔮 Prediction Result")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(
            f'<div style="text-align:center; padding:24px; background:#1a1d2e; '
            f'border-radius:16px; border:1.5px solid {color_hex};">'
            f'<div style="color:#9ca3af;font-size:.9rem;margin-bottom:8px;">PREDICTED SPEC TIER</div>'
            f'<div class="tier-badge {css_class}">{tier.upper()}</div>'
            f'<div style="color:#6b7280;font-size:.82rem;margin-top:10px;">'
            f'PowerScore: {row_df["PowerScore"].iloc[0]:.1f} &nbsp;|&nbsp; '
            f'SpecScore: {row_df["SpecScore"].iloc[0]:.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown("**Class Probabilities**")
        for idx, label in enumerate(state['le'].classes_):
            p  = proba[idx]
            hx = TIER_COLOR[label][0]
            st.markdown(
                f'<div class="prob-bar-wrap">'
                f'<div class="prob-label">{label}</div>'
                f'<div style="background:#1e2130;border-radius:8px;height:22px;">'
                f'<div style="background:{hx};width:{p*100:.1f}%;height:100%;'
                f'border-radius:8px;display:flex;align-items:center;'
                f'padding-left:8px;font-size:.8rem;font-weight:700;color:#fff;">'
                f'{p*100:.1f}%</div></div></div>',
                unsafe_allow_html=True,
            )
        spec_data = {
            "Spec": ["RAM","Storage","Storage Type","CPU Tier","GPU Tier","Screen","Brand","Type"],
            "Value": [
                f"{row_df['Ram_GB'].iloc[0]} GB",
                f"{row_df['Storage_GB'].iloc[0]} GB",
                row_df['StorageType'].iloc[0],
                ["Entry","Low","Mid","High-End"][int(row_df['CPU_Tier'].iloc[0])],
                ["Integrated","Mid Dedicated","High Dedicated"][int(row_df['GPU_Tier'].iloc[0])],
                screen_res.strip(), company, laptop_type,
            ],
        }
        st.dataframe(pd.DataFrame(spec_data), hide_index=True, use_container_width=True)

    st.divider()


# ─────────────────────────────────────────────
# DASHBOARD TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔍 Feature Analysis", "📈 Data Insights"])

df      = state['df']
PALETTE = ['#3B82F6', '#F59E0B', '#10B981']
CATS    = list(state['le'].classes_)

plt.rcParams.update({
    'figure.facecolor': '#1a1d2e', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor':   '#3b3f5c', 'axes.labelcolor': '#d1d5db',
    'xtick.color':      '#9ca3af', 'ytick.color':     '#9ca3af',
    'text.color':       '#e5e7eb', 'grid.color':      '#2d3148',
    'grid.alpha':       0.4,
})

# ── Tab 1 : Model Performance ────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1a1d2e')
        sns.heatmap(
            state['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=CATS, yticklabels=CATS,
            linewidths=0.6, linecolor='#0f1117',
            annot_kws={'size': 15, 'weight': 'bold'},
        )
        ax.set_xlabel('Predicted Label', labelpad=10)
        ax.set_ylabel('True Label', labelpad=10)
        ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=14)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown('<div class="section-header">Per-Class Metrics</div>', unsafe_allow_html=True)
        rep = state['report']
        mdf = pd.DataFrame({
            'Class':     CATS,
            'Precision': [rep[c]['precision'] for c in CATS],
            'Recall':    [rep[c]['recall']    for c in CATS],
            'F1-Score':  [rep[c]['f1-score']  for c in CATS],
        }).set_index('Class')
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1a1d2e')
        x, w = np.arange(len(CATS)), 0.25
        for i, (metric, col) in enumerate(zip(['Precision','Recall','F1-Score'],
                                               ['#6366f1','#f59e0b','#10b981'])):
            bars = ax.bar(x + i*w, mdf[metric], w, label=metric, color=col, alpha=0.88)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{bar.get_height():.2f}', ha='center', fontsize=8.5, fontweight='bold')
        ax.set_xticks(x + w); ax.set_xticklabels(CATS)
        ax.set_ylim(0, 1.12)
        ax.set_title('Precision / Recall / F1 by Class', fontsize=13, fontweight='bold', pad=14)
        ax.legend(framealpha=0.2); ax.grid(axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">5-Fold Cross-Validation</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#1a1d2e')
    fold_colors = ['#6366f1','#8b5cf6','#a78bfa','#c4b5fd','#ddd6fe']
    for i, (score, col) in enumerate(zip(state['cv_acc'], fold_colors)):
        ax.bar(f'Fold {i+1}', score*100, color=col, edgecolor='#0f1117', alpha=0.9)
        ax.text(i, score*100+0.2, f'{score*100:.2f}%', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(state['cv_acc'].mean()*100, color='#f43f5e', linestyle='--', linewidth=1.5,
               label=f'Mean = {state["cv_acc"].mean()*100:.2f}%')
    ax.set_ylim(95, 102); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cross-Validation Accuracy per Fold', fontsize=13, fontweight='bold', pad=14)
    ax.legend(framealpha=0.2); ax.grid(axis='y')
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ── Tab 2 : Feature Analysis ─────────────────
with tab2:
    st.markdown('<div class="section-header">Top 20 Feature Importances (XGBoost)</div>', unsafe_allow_html=True)
    top20 = state['importances'].nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#1a1d2e')
    colors_fi = plt.cm.Blues(np.linspace(0.4, 0.95, len(top20)))
    bars = ax.barh(top20.index, top20.values, color=colors_fi, edgecolor='#0f1117')
    for bar, val in zip(bars, top20.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score', labelpad=10)
    ax.set_title('XGBoost Feature Importances', fontsize=14, fontweight='bold', pad=14)
    ax.grid(axis='x')
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    fa1, fa2 = st.columns(2)
    with fa1:
        st.markdown('<div class="section-header">Storage Type Distribution</div>', unsafe_allow_html=True)
        sc = df['StorageType'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1d2e')
        ax.pie(sc.values, labels=sc.index, autopct='%1.1f%%', startangle=140,
               colors=['#6366f1','#f59e0b','#10b981','#f43f5e','#06b6d4','#ec4899'],
               textprops={'color':'#e5e7eb','fontsize':9})
        ax.set_title('Storage Type Mix', fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with fa2:
        st.markdown('<div class="section-header">GPU Distribution by Tier</div>', unsafe_allow_html=True)
        gpu_map = {0:'Integrated',1:'Mid Dedicated',2:'High Dedicated'}
        df['GPU_Label'] = df['GPU_Tier'].map(gpu_map)
        ct = pd.crosstab(df['GPU_Label'], df['SpecTier'].astype(str))
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1d2e')
        ct.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='#0f1117', alpha=0.88)
        ax.set_title('GPU Level per Spec Tier', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('GPU Category'); ax.set_ylabel('Count')
        ax.legend(title='Spec Tier', framealpha=0.2, fontsize=9)
        plt.xticks(rotation=20); ax.grid(axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ── Tab 3 : Data Insights ────────────────────
with tab3:
    di1, di2 = st.columns(2)
    with di1:
        st.markdown('<div class="section-header">RAM Distribution by Spec Tier</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#1a1d2e')
        for tier, col in zip(CATS, PALETTE):
            subset = df[df['SpecTier'].astype(str) == tier]['Ram_GB']
            ax.hist(subset, bins=[2,4,6,8,10,12,14,16,18], alpha=0.65, label=tier,
                    color=col, edgecolor='#0f1117')
        ax.set_xlabel('RAM (GB)'); ax.set_ylabel('Count')
        ax.set_title('RAM by Spec Tier', fontsize=13, fontweight='bold', pad=12)
        ax.legend(title='Spec Tier', framealpha=0.2); ax.set_xticks([4,8,12,16]); ax.grid(axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with di2:
        st.markdown('<div class="section-header">Storage vs RAM by Spec Tier</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#1a1d2e')
        for tier, col in zip(CATS, PALETTE):
            s = df[df['SpecTier'].astype(str) == tier]
            ax.scatter(s['Storage_GB'], s['Ram_GB'], alpha=0.55, label=tier,
                       color=col, edgecolors='#0f1117', linewidths=0.3, s=60)
        ax.set_xlabel('Storage (GB)'); ax.set_ylabel('RAM (GB)')
        ax.set_yticks([4,8,12,16])
        ax.set_title('Storage vs RAM', fontsize=13, fontweight='bold', pad=12)
        ax.legend(title='Spec Tier', framealpha=0.2); ax.grid()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">Power Score Distribution by Spec Tier</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1d2e')
    data_plot = [df[df['SpecTier'].astype(str)==t]['PowerScore'].dropna() for t in CATS]
    bp = ax.boxplot(
        data_plot, patch_artist=True, labels=CATS,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.4),
    )
    for patch, col in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    for i, (data, col) in enumerate(zip(data_plot, PALETTE), 1):
        ax.text(i, data.max()*1.03, f'μ={data.mean():.1f}', ha='center', fontsize=10,
                color=col, fontweight='bold')
    ax.set_ylabel('Power Score'); ax.set_xlabel('Spec Tier')
    ax.set_title('Power Score = RAM × CPU Tier × GPU Tier', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y')
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">Laptops per Brand</div>', unsafe_allow_html=True)
    bc = df['CompanyName'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#1a1d2e')
    bars = ax.bar(bc.index, bc.values, color='#6366f1', edgecolor='#0f1117', alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                str(int(bar.get_height())), ha='center', fontsize=9)
    ax.set_xlabel('Brand'); ax.set_ylabel('Count')
    ax.set_title('Dataset Distribution by Brand', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=15); ax.grid(axis='y')
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Built with XGBoost + Random Forest Soft-Voting Ensemble · "
    "Feature-engineered from RAM, Storage (type + speed), CPU tier, GPU tier, "
    "screen quality, brand, and form-factor. · ~99% accuracy on held-out test set."
)
