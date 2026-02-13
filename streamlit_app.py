"""
PHISHING WEBSITE DETECTION - INTERACTIVE STREAMLIT APPLICATION
===============================================================
A comprehensive machine learning application for detecting phishing websites
using multiple trained classification models with interactive visualizations.

Author: RISHIT ANAND (2025ab05172)
Course: BITS WILP Machine Learning Assignment 2
Date: 2026-02-15
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Phishing Detection ML",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with light background
st.markdown("""
    <style>
    /* Light gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    .main {
        padding: 0rem 0rem;
        background-color: transparent;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%);
    }
    
    /* Metric cards with modern look */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(240, 244, 248, 0.8) 100%);
        border-radius: 12px;
        padding: 10px;
        backdrop-filter: blur(10px);
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.5);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 30px 0;
        border-bottom: 3px solid #1f77b4;
        font-size: 2.5rem !important;
        font-weight: 700;
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    h2 {
        color: #2ca02c;
        margin-top: 30px;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    h3 {
        color: #1f77b4;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    /* Info/Success/Warning/Error boxes */
    .success-box, [data-testid="stAlert"] {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border: 2px solid #155724 !important;
        border-radius: 8px !important;
        color: #155724 !important;
        padding: 15px !important;
        box-shadow: 0 4px 12px rgba(21, 87, 36, 0.1) !important;
    }
    
    /* Card-like containers */
    .card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #667eea;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox and input styling */
    .stSelectbox, .stNumberInput, .stTextInput {
        border-radius: 8px;
        border: 2px solid #e0e7ff;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2ca02c 0%, #17a2b8 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(44, 160, 44, 0.3) !important;
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CACHE FUNCTIONS FOR PERFORMANCE OPTIMIZATION
# ============================================================================
@st.cache_resource
@st.cache_resource
def load_models():
    """
    Load pre-trained machine learning models from pickle files.
    
    Uses models trained and saved from Assignment 2.ipynb.
    Cached to avoid reloading on every app rerun.
    """
    import pickle
    import os
    
    # Paths to model files
    models_dir = "models"
    model_files = {
        'Logistic Regression': os.path.join(models_dir, 'LogisticRegression.pkl'),
        'Decision Tree': os.path.join(models_dir, 'DecisionTree.pkl'),
        'K-Nearest Neighbors': os.path.join(models_dir, 'KNeighborsClassifier.pkl'),
        'Naive Bayes (Gaussian)': os.path.join(models_dir, 'GaussianNB.pkl'),
        'Random Forest': os.path.join(models_dir, 'RandomForest.pkl'),
        'XGBoost': os.path.join(models_dir, 'XGBoost.pkl'),
    }
    
    # Load scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load test data info
    test_data_path = os.path.join(models_dir, 'test_data_info.pkl')
    with open(test_data_path, 'rb') as f:
        test_data_info = pickle.load(f)
    
    X_test = test_data_info['X_test']
    y_test = test_data_info['y_test']
    X_test_scaled = test_data_info['X_test_scaled']
    
    # Load trained models and generate predictions
    trained_models = {}
    predictions = {}
    probabilities = {}
    
    for model_name, model_path in model_files.items():
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        use_scaled = model_package['use_scaled']
        
        # Store model with metadata
        trained_models[model_name] = (model, scaler, use_scaled)
        
        # Generate predictions on test set
        X_te = X_test_scaled if use_scaled else X_test
        predictions[model_name] = model.predict(X_te)
        
        # Generate probability predictions
        if hasattr(model, 'predict_proba'):
            probabilities[model_name] = model.predict_proba(X_te)[:, 1]
        else:
            probabilities[model_name] = model.decision_function(X_te)
    
    return trained_models, predictions, probabilities, X_test, y_test, scaler, test_data_info

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics for a model."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Create a ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_dict):
    """Create a bar chart comparing metrics across models."""
    df = pd.DataFrame(metrics_dict).T
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Initialize session state for tab persistence
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Header with enhanced styling
    st.markdown("<h1>🔐 Phishing Website Detection System</h1>", unsafe_allow_html=True)
    
    # Subtitle with better styling
    st.markdown("""
    <div style="
        text-align: center; 
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #667eea;
        border-right: 4px solid #764ba2;
    ">
        <p style="
            color: #333; 
            font-size: 18px;
            font-weight: 500;
            margin: 0;
        ">
        Advanced Machine Learning Classification for Identifying Fraudulent Websites
        </p>
        <p style="
            color: #666; 
            font-size: 13px;
            margin-top: 8px;
        ">
        ✨ 6 ML Models | 🎯 97%+ Accuracy | 📊 Real-time Predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading and training models... Please wait"):
        trained_models, predictions, probabilities, X_test, y_test, scaler, test_data_info = load_models()
    
    # Sidebar with enhanced styling
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    ">
        <h2 style="color: white; margin-top: 0;">⚙️ Configuration</h2>
        <p style="color: rgba(255,255,255,0.9); font-size: 13px;">
        Select and configure your ML model
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection with styling
    st.sidebar.markdown("""
    <p style="
        color: #1f77b4;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    ">🤖 SELECT MODEL</p>
    """, unsafe_allow_html=True)
    
    selected_model = st.sidebar.selectbox(
        "Choose a machine learning model",
        list(trained_models.keys()),
        help="Choose a machine learning model for prediction"
    )
    
    st.sidebar.markdown("---")
    
    # Enhanced navigation buttons with st.rerun()
    st.sidebar.markdown("""
    <p style="
        color: #2ca02c;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 12px;
    ">📑 QUICK NAVIGATION</p>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.sidebar.columns(3)
    
    if col1.button("🔍 Upload", use_container_width=True, key="nav_upload"):
        st.session_state.active_tab = 3
        st.rerun()
    if col2.button("📈 Model", use_container_width=True, key="nav_model"):
        st.session_state.active_tab = 0
        st.rerun()
    if col3.button("🧪 Eval", use_container_width=True, key="nav_eval"):
        st.session_state.active_tab = 1
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Display current tab indicator
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border-left: 3px solid #667eea;
    ">
        <p style="
            color: #667eea;
            font-size: 12px;
            margin: 0;
            font-weight: 600;
        ">Active Tab</p>
        <p style="color: #333; margin: 5px 0 0 0; font-size: 13px;">""" + ["📊 Overview", "📈 Evaluation", "🔍 Analysis", "🎯 Predict", "📖 Help"][st.session_state.active_tab] + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tab navigation bar
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.9) 0%, rgba(240, 244, 248, 0.9) 100%);
        border-radius: 12px;
        padding: 12px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    ">
    """, unsafe_allow_html=True)
    
    tab_cols = st.columns(5)
    tab_configs = [
        ("📊 Overview", 0),
        ("📈 Evaluation", 1),
        ("🔍 Analysis", 2),
        ("🎯 Predict", 3),
        ("📖 Help", 4),
    ]
    
    for col, (label, tab_idx) in zip(tab_cols, tab_configs):
        with col:
            is_active = st.session_state.active_tab == tab_idx
            button_style = """
            <button onclick="document.dispatchEvent(new CustomEvent('streamlit:tabClicked', {detail: {tab: """ + str(tab_idx) + """}}))" 
                    style="
                        width: 100%;
                        padding: 10px;
                        border-radius: 8px;
                        border: 2px solid """ + ("'#667eea'" if is_active else "'transparent'") + """;
                        background: """ + ("'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'" if is_active else "'rgba(255, 255, 255, 0.5)'") + """;
                        color: """ + ("'white'" if is_active else "'#333'") + """;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        font-size: 13px;
                    ">""" + label + """</button>
            """
            
            if st.button(label, key=f"tab_{tab_idx}", use_container_width=True):
                st.session_state.active_tab = tab_idx
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 1: MODEL OVERVIEW
    # ========================================================================
    if st.session_state.active_tab == 0:
        st.markdown("""
        <h2 style="text-align: center; color: #667eea;">
        🎯 Selected Model Information
        </h2>
        """, unsafe_allow_html=True)
        
        # Key metrics display
        col1, col2, col3 = st.columns(3)
        
        metrics = get_model_metrics(y_test, predictions[selected_model], probabilities[selected_model])
        
        with col1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                color: white;
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
            ">
                <p style="margin: 0; font-size: 12px; opacity: 0.9;">MODEL</p>
                <h3 style="margin: 10px 0; color: white; font-size: 18px;">""" + selected_model + """</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #2ca02c 0%, #17a2b8 100%);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                color: white;
                box-shadow: 0 8px 16px rgba(44, 160, 44, 0.3);
            ">
                <p style="margin: 0; font-size: 12px; opacity: 0.9;">ACCURACY</p>
                <h3 style="margin: 10px 0; color: white; font-size: 18px;">{metrics['Accuracy']:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ff6b6b 0%, #ff8c42 100%);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                color: white;
                box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3);
            ">
                <p style="margin: 0; font-size: 12px; opacity: 0.9;">AUC-ROC</p>
                <h3 style="margin: 10px 0; color: white; font-size: 18px;">{metrics['AUC-ROC']:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model description with enhanced styling
        model_descriptions = {
            'Logistic Regression': {
                'Type': 'Linear Classification',
                'Description': 'A probabilistic model that outputs the probability of class membership.',
                'Pros': ['✅ Fast predictions', '✅ Interpretable results', '✅ Confidence scores'],
                'Cons': ['❌ Limited to linear boundaries', '❌ May underfit complex data'],
                'Use Case': 'Quick baseline, real-time predictions',
                'Icon': '📊'
            },
            'Decision Tree': {
                'Type': 'Tree-Based Classification',
                'Description': 'Recursively splits features to create decision rules.',
                'Pros': ['✅ Easy to interpret', '✅ No scaling needed', '✅ Fast training'],
                'Cons': ['❌ Overfitting prone', '❌ Unstable with data changes'],
                'Use Case': 'Feature importance analysis',
                'Icon': '🌳'
            },
            'K-Nearest Neighbors': {
                'Type': 'Instance-Based Learning',
                'Description': 'Classifies based on distance to nearest training samples.',
                'Pros': ['✅ Simple to understand', '✅ Effective locally', '✅ No training phase'],
                'Cons': ['❌ Slow predictions', '❌ Memory intensive'],
                'Use Case': 'Small datasets, local patterns',
                'Icon': '👥'
            },
            'Naive Bayes (Gaussian)': {
                'Type': 'Probabilistic Classification',
                'Description': 'Uses Bayes theorem assuming feature independence.',
                'Pros': ['✅ Very fast', '✅ Good baseline', '✅ Simple'],
                'Cons': ['❌ Assumes independence', '❌ Lower precision'],
                'Use Case': 'Text classification, quick prototypes',
                'Icon': '🎲'
            },
            'Random Forest': {
                'Type': 'Ensemble (Bagging)',
                'Description': 'Combines multiple decision trees trained on random subsets.',
                'Pros': ['✅ Highest accuracy', '✅ Robust & reliable', '✅ Feature importance'],
                'Cons': ['❌ Less interpretable', '❌ Slower training'],
                'Use Case': '⭐ Production deployment, best accuracy',
                'Icon': '🌲'
            },
            'XGBoost': {
                'Type': 'Ensemble (Gradient Boosting)',
                'Description': 'Sequentially builds trees, each correcting previous errors.',
                'Pros': ['✅ State-of-the-art', '✅ Non-linear power', '✅ Feature interactions'],
                'Cons': ['❌ Complex tuning', '❌ Prone to overfitting'],
                'Use Case': 'Maximum accuracy, complex patterns',
                'Icon': '🚀'
            }
        }
        
        desc = model_descriptions[selected_model]
        
        # Model type and info
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin-top: 0; color: #667eea;">{desc['Icon']} {desc['Type']}</h3>
            <p style="color: #555; font-size: 15px; line-height: 1.6;">
            <strong>What it does:</strong> {desc['Description']}
            </p>
            <p style="color: #666; font-size: 14px; margin-bottom: 0;">
            <strong>Best for:</strong> {desc['Use Case']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Strengths and weaknesses
        col_strength, col_weakness = st.columns(2)
        
        with col_strength:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(44, 160, 44, 0.1), rgba(23, 162, 184, 0.1));
                border-radius: 12px;
                padding: 20px;
                border-left: 4px solid #2ca02c;
            ">
                <h4 style="color: #2ca02c; margin-top: 0;">💪 Strengths</h4>
            """, unsafe_allow_html=True)
            
            for strength in desc['Pros']:
                st.write(strength)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_weakness:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 140, 66, 0.1));
                border-radius: 12px;
                padding: 20px;
                border-left: 4px solid #ff6b6b;
            ">
                <h4 style="color: #ff6b6b; margin-top: 0;">⚠️ Weaknesses</h4>
            """, unsafe_allow_html=True)
            
            for weakness in desc['Cons']:
                st.write(weakness)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display all models comparison with enhanced styling
        st.markdown("""
        <h3 style="color: #667eea; text-align: center; margin-top: 30px;">
        ⚡ All Models Quick Comparison
        </h3>
        """, unsafe_allow_html=True)
        
        all_metrics = {}
        for model_name in trained_models.keys():
            metrics = get_model_metrics(y_test, predictions[model_name], probabilities[model_name])
            all_metrics[model_name] = metrics
        
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Color-code the dataframe with better styling
        st.dataframe(
            comparison_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
            use_container_width=True
        )
        
        # Best model highlight
        best_model = comparison_df.index[0]
        best_acc = comparison_df.iloc[0]['Accuracy']
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2ca02c 0%, #17a2b8 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 16px rgba(44, 160, 44, 0.3);
            margin-top: 20px;
        ">
            <h3 style="margin: 0; color: white; font-size: 20px;">🏆 Best Model</h3>
            <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.95;">
            <strong>{best_model}</strong> with <strong>{best_acc:.2%}</strong> accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: MODEL EVALUATION
    # ========================================================================
    if st.session_state.active_tab == 1:
        st.markdown(f"""
        <h2 style="text-align: center; color: #667eea;">
        📊 Detailed Evaluation: {selected_model}
        </h2>
        """, unsafe_allow_html=True)
        
        # Get metrics
        y_pred = predictions[selected_model]
        y_proba = probabilities[selected_model]
        metrics = get_model_metrics(y_test, y_pred, y_proba)
        
        # Display metrics in enhanced boxes
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metric_configs = [
            ("Accuracy", metrics['Accuracy'], col1, "#667eea"),
            ("Precision", metrics['Precision'], col2, "#2ca02c"),
            ("Recall", metrics['Recall'], col3, "#ff6b6b"),
            ("F1-Score", metrics['F1 Score'], col4, "#ff8c42"),
            ("AUC-ROC", metrics['AUC-ROC'], col5, "#17a2b8"),
            ("MCC", metrics['MCC'], col6, "#9b59b6"),
        ]
        
        for metric_name, metric_value, col, color in metric_configs:
            with col:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, {color}cc 100%);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                ">
                    <p style="margin: 0; font-size: 12px; opacity: 0.95;">{metric_name}</p>
                    <h3 style="margin: 8px 0 0 0; color: white; font-size: 24px;">{metric_value:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(plot_confusion_matrix(y_test, y_pred, f"{selected_model} - Confusion Matrix"))
            
            # Confusion matrix explanation
            cm = confusion_matrix(y_test, y_pred)
            st.info(f"""
            **Confusion Matrix Breakdown:**
            - **True Negatives (TN):** {cm[0, 0]} - Legitimate sites correctly identified
            - **False Positives (FP):** {cm[0, 1]} - Legitimate sites flagged as phishing
            - **False Negatives (FN):** {cm[1, 0]} - Phishing sites not detected
            - **True Positives (TP):** {cm[1, 1]} - Phishing sites correctly identified
            """)
        
        with col2:
            st.pyplot(plot_roc_curve(y_test, y_proba, f"{selected_model} - ROC Curve"))
            
            # ROC explanation
            st.info("""
            **ROC Curve Interpretation:**
            - Curves closer to top-left corner = better model
            - AUC (Area Under Curve) ranges from 0.5 to 1.0
            - 0.9-1.0: Excellent discrimination
            - 0.8-0.9: Good discrimination
            - 0.7-0.8: Fair discrimination
            """)
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("📋 Classification Report")
        
        report_dict = classification_report(
            y_test, y_pred,
            target_names=['Legitimate', 'Phishing'],
            output_dict=True,
            zero_division=0
        )
        
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )
        
        # Report interpretation
        st.info("""
        **Classification Report Metrics:**
        - **Precision:** Of sites flagged as phishing, how many actually are
        - **Recall:** Of all phishing sites, how many were correctly identified
        - **F1-Score:** Harmonic mean balancing precision and recall
        - **Support:** Number of actual occurrences in each class
        """)
    
    # ========================================================================
    # TAB 3: COMPARATIVE ANALYSIS
    # ========================================================================
    if st.session_state.active_tab == 2:
        st.markdown("""
        <h2 style="text-align: center; color: #667eea;">
        📊 All Models Comparative Analysis
        </h2>
        """, unsafe_allow_html=True)
        
        st.info("🔍 Compare all 6 machine learning models side-by-side to identify the best performer for your use case")
        
        # All metrics comparison
        all_metrics = {}
        for model_name in trained_models.keys():
            metrics = get_model_metrics(y_test, predictions[model_name], probabilities[model_name])
            all_metrics[model_name] = metrics
        
        st.pyplot(plot_metrics_comparison(all_metrics))
        
        st.markdown("---")
        
        # Detailed comparison table
        st.subheader("📈 Detailed Metrics Table")
        
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        st.dataframe(
            comparison_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
            use_container_width=True
        )
        
        # Key insights
        st.markdown("---")
        st.subheader("🔍 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_model = comparison_df.index[0]
            best_accuracy = comparison_df.iloc[0]['Accuracy']
            st.metric(
                "🥇 Best Accuracy",
                best_model,
                f"{best_accuracy:.4f}"
            )
        
        with col2:
            best_precision = comparison_df['Precision'].idxmax()
            best_prec_value = comparison_df['Precision'].max()
            st.metric(
                "🎯 Best Precision",
                best_precision,
                f"{best_prec_value:.4f}"
            )
        
        with col3:
            best_recall = comparison_df['Recall'].idxmax()
            best_recall_value = comparison_df['Recall'].max()
            st.metric(
                "🔔 Best Recall",
                best_recall,
                f"{best_recall_value:.4f}"
            )
        
        st.info(f"""
        **Recommendations:**
        1. **For Production Use:** {comparison_df.index[0]} offers the best overall accuracy
        2. **Best Precision:** {best_precision} - minimize false alarms
        3. **Best Recall:** {best_recall} - catch most phishing attempts
        4. **Ensemble Approach:** Combine predictions from top models for robustness
        """)
    
    # ========================================================================
    # TAB 4: UPLOAD & PREDICT
    # ========================================================================
    if st.session_state.active_tab == 3:
        st.markdown("""
        <h2 style="text-align: center; color: #667eea;">
        🎯 Dataset Upload & Prediction
        </h2>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        ">
            <h4 style="color: #667eea; margin-top: 0;">📋 How to Use:</h4>
            <ol style="color: #333; line-height: 1.8;">
                <li><strong>Download</strong> the sample test dataset</li>
                <li><strong>Upload</strong> your CSV file with 38 numeric features</li>
                <li><strong>Get Predictions</strong> with confidence scores</li>
                <li><strong>Download Results</strong> with predicted labels</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download sample data: Use actual test_data.csv file
            with open('test_data.csv', 'r') as f:
                csv_content = f.read().encode('utf-8')
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #2ca02c 0%, #17a2b8 100%);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                color: white;
                margin-bottom: 20px;
            ">
                <h4 style="margin-top: 0; color: white;">📥 Sample Data</h4>
                <p style="opacity: 0.9; margin-bottom: 15px;">Get started with our test dataset</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download Test Data (2,211 samples)",
                data=csv_content,
                file_name="test_data_full.csv",
                mime="text/csv",
                help="Download the actual test data used to evaluate the models",
                use_container_width=True
            )
        
        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 140, 66, 0.1));
                border-radius: 12px;
                padding: 20px;
                border-left: 4px solid #ff6b6b;
            ">
                <h4 style="color: #ff6b6b; margin-top: 0;">📝 File Requirements</h4>
                <ul style="color: #333; font-size: 14px; margin-bottom: 0;">
                    <li>CSV format with numeric features</li>
                    <li><strong>38 features</strong> required</li>
                    <li>Feature names: Feature_1 to Feature_38</li>
                    <li><em>Optional:</em> 'Actual_Class' for evaluation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload your test dataset",
            type=['csv'],
            help="Upload a CSV file with the same feature structure"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                
                feature_cols = [col for col in uploaded_data.columns if col.startswith('Feature_')]
                
                if len(feature_cols) == 0:
                    if 'feature_names' in test_data_info:
                        feature_names = test_data_info['feature_names']
                        available_cols = set(uploaded_data.columns)
                        
                        feature_cols = [col for col in feature_names if col in available_cols]
                    
                    if len(feature_cols) == 0:
                        exclude_cols = {'index', 'Index', 'ID', 'id', 'Actual_Class', 'Prediction', 'Prediction_Label', 'Confidence', 'Confidence_%'}
                        feature_cols = [col for col in uploaded_data.columns 
                                       if col not in exclude_cols 
                                       and pd.api.types.is_numeric_dtype(uploaded_data[col])]
                
                if len(feature_cols) != X_test.shape[1]:
                    st.error(f"❌ Expected {X_test.shape[1]} features, got {len(feature_cols)}")
                    if 'feature_names' in test_data_info:
                        st.info(f"Expected columns: {test_data_info['feature_names'][:5]}... (showing first 5)")
                else:
                    X_uploaded = uploaded_data[feature_cols].values
                    
                    # Get model info
                    model, scaler_obj, use_scaled = trained_models[selected_model]
                    
                    if use_scaled:
                        X_uploaded_scaled = scaler_obj.transform(X_uploaded)
                        y_pred_uploaded = model.predict(X_uploaded_scaled)
                        y_proba_uploaded = model.predict_proba(X_uploaded_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_uploaded_scaled)
                    else:
                        y_pred_uploaded = model.predict(X_uploaded)
                        y_proba_uploaded = model.predict_proba(X_uploaded)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_uploaded)
                
                    st.success("✅ Predictions generated successfully!")
                    
                    results_df = uploaded_data.copy()
                    results_df['Prediction'] = y_pred_uploaded
                    results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Legitimate', 1: 'Phishing'})
                    results_df['Confidence'] = np.abs(y_proba_uploaded)
                    results_df['Confidence_%'] = (results_df['Confidence'] * 100).round(2)
                    
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        legitimate_count = (results_df['Prediction'] == 0).sum()
                        st.metric("🟢 Legitimate Sites", legitimate_count)
                    
                    with col2:
                        phishing_count = (results_df['Prediction'] == 1).sum()
                        st.metric("🔴 Phishing Sites", phishing_count)
                    
                    with col3:
                        avg_confidence = results_df['Confidence_%'].mean()
                        st.metric("📊 Avg Confidence", f"{avg_confidence:.2f}%")
                    
                    st.markdown("---")
                    
                    # Display predictions table
                    display_cols = feature_cols[:5] + ['Prediction_Label', 'Confidence_%']
                    st.dataframe(
                        results_df[display_cols].head(20).style.format({
                            'Confidence_%': '{:.2f}',
                        }).background_gradient(
                            subset=['Confidence_%'],
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=100
                        ),
                        use_container_width=True
                    )
                    
                    st.markdown(f"*Showing first 20 of {len(results_df)} predictions*")
                    
                    st.markdown("---")
                    
                    # Distribution chart
                    st.subheader("📈 Prediction Distribution")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Class distribution
                    results_df['Prediction_Label'].value_counts().plot(
                        kind='bar',
                        ax=ax1,
                        color=['#2ecc71', '#e74c3c']
                    )
                    ax1.set_title('Class Distribution', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Count')
                    ax1.set_xlabel('Prediction')
                    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
                    
                    # Confidence distribution
                    ax2.hist(results_df['Confidence_%'], bins=20, color='#3498db', edgecolor='black')
                    ax2.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
                    ax2.set_xlabel('Confidence (%)')
                    ax2.set_ylabel('Frequency')
                    ax2.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Comparison with actual (if available)
                    if 'Actual_Class' in uploaded_data.columns:
                        st.markdown("---")
                        st.subheader("🔍 Comparison with Actual Labels")
                        
                        actual_labels = uploaded_data['Actual_Class'].map({0: 'Legitimate', 1: 'Phishing'})
                        accuracy = (results_df['Prediction'] == uploaded_data['Actual_Class']).mean()
                        
                        st.metric("✅ Accuracy on Uploaded Data", f"{accuracy:.4f} ({accuracy*100:.2f}%)")
                        
                        # Confusion matrix
                        cm = confusion_matrix(uploaded_data['Actual_Class'], results_df['Prediction'])
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.pyplot(plot_confusion_matrix(
                                uploaded_data['Actual_Class'],
                                results_df['Prediction'],
                                "Confusion Matrix - Uploaded Data"
                            ))
                        
                        with col2:
                            st.info(f"""
                            **Performance Metrics:**
                            - TP (Correctly identified Phishing): {cm[1, 1]}
                            - TN (Correctly identified Legitimate): {cm[0, 0]}
                            - FP (False Alarms): {cm[0, 1]}
                            - FN (Missed Phishing): {cm[1, 0]}
                            """)
                    
                    st.markdown("---")
                    
                    results_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=results_csv,
                        file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # ========================================================================
    # TAB 5: DOCUMENTATION
    # ========================================================================
    if st.session_state.active_tab == 4:
        st.markdown("""
        <h2 style="text-align: center; color: #667eea;">
        📚 Application Documentation & Help
        </h2>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3 style="color: #667eea; margin-top: 0;">ℹ️ About This Application</h3>
                <p><strong>Purpose:</strong></p>
                <p>Advanced machine learning system for detecting phishing websites using 6 different classification algorithms.</p>
                <p><strong>Dataset:</strong></p>
                <ul>
                    <li>11,055 total samples</li>
                    <li>31 original features</li>
                    <li>7 engineered features</li>
                    <li>Binary classification (Phishing vs Legitimate)</li>
                </ul>
                <p><strong>Models:</strong></p>
                <ol style="font-size: 13px;">
                    <li>Logistic Regression</li>
                    <li>Decision Tree</li>
                    <li>K-Nearest Neighbors</li>
                    <li>Naive Bayes (Gaussian)</li>
                    <li>Random Forest</li>
                    <li>XGBoost</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3 style="color: #2ca02c; margin-top: 0;">👤 Student Information</h3>
                <p><strong>Name:</strong> RISHIT ANAND</p>
                <p><strong>Student ID:</strong> 2025ab05172</p>
                <p><strong>Course:</strong> BITS WILP Machine Learning</p>
                <p><strong>Assignment:</strong> Programming Assignment 2</p>
                <p><strong>Date:</strong> 2026-02-15</p>
                <hr>
                <h4 style="color: #ff6b6b;">✨ Key Features</h4>
                <ul style="font-size: 13px;">
                    <li>📊 Interactive Visualizations</li>
                    <li>⚡ Model Comparison</li>
                    <li>🎯 Real-time Predictions</li>
                    <li>📈 Performance Metrics</li>
                    <li>📋 Detailed Reports</li>
                    <li>💾 Result Export</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <h3 style="color: #667eea; text-align: center;">🔐 Phishing Detection Features</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color: #667eea; margin-top: 0;">🔗 URL-Based</h4>
                <ul style="font-size: 13px;">
                    <li>IP address presence</li>
                    <li>URL length abnormality</li>
                    <li>Shortening services</li>
                    <li>Special characters</li>
                    <li>Domain prefixes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="color: #2ca02c; margin-top: 0;">🏢 Domain-Based</h4>
                <ul style="font-size: 13px;">
                    <li>SSL certificate validity</li>
                    <li>Registration length</li>
                    <li>Domain age</li>
                    <li>DNS records</li>
                    <li>PageRank score</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
                <h4 style="color: #ff6b6b; margin-top: 0;">📄 Content-Based</h4>
                <ul style="font-size: 13px;">
                    <li>HTTPS tokens</li>
                    <li>Form handlers</li>
                    <li>Iframe usage</li>
                    <li>Popup windows</li>
                    <li>Web traffic rank</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <h3 style="color: #667eea; text-align: center;">🚀 Quick Start Guide</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4 style="color: #667eea; margin-top: 0;">📖 How to Use This App</h4>
            <ol style="color: #333; line-height: 1.8; font-size: 14px;">
                <li><strong>📊 Overview Tab:</strong> View information about your selected model and compare all 6 models</li>
                <li><strong>📈 Evaluation Tab:</strong> See detailed metrics, confusion matrix, and ROC curve</li>
                <li><strong>🔍 Analysis Tab:</strong> Compare all models side-by-side with comprehensive metrics</li>
                <li><strong>🎯 Predict Tab:</strong> Upload test data and get real-time predictions</li>
                <li><strong>📚 Help Tab:</strong> Read this help section and view FAQ</li>
            </ol>
        </div>
        
        <h3 style="color: #667eea; margin-top: 30px;">❓ Frequently Asked Questions</h3>
        """, unsafe_allow_html=True)
        
        with st.expander("🎯 What is phishing?", expanded=False):
            st.write("""
            Phishing is a cyberattack where fraudsters create fake websites or emails to trick users into 
            revealing sensitive information like passwords, credit card numbers, or personal data. This application 
            helps detect phishing websites by analyzing their characteristics and patterns.
            """)
        
        with st.expander("⭐ How accurate are these models?", expanded=False):
            st.write("""
            Random Forest achieves **97.06% accuracy** on the test dataset (2,211 samples). 
            However, accuracy depends on the specific dataset and phishing attack patterns. 
            We recommend using Random Forest or XGBoost for production deployments.
            """)
        
        with st.expander("📊 What does confidence score mean?", expanded=False):
            st.write("""
            The confidence score (0-100%) indicates how certain the model is about its prediction:
            - **80-100%:** Very confident prediction (reliable)
            - **60-80%:** Moderately confident (acceptable)
            - **<60%:** Low confidence (use with caution)
            
            For important security decisions, consider using predictions with >85% confidence.
            """)
        
        with st.expander("🔧 How is the model trained?", expanded=False):
            st.write("""
            The models are trained on labeled examples of phishing and legitimate websites from the UCI dataset:
            1. **Data Preprocessing:** 11,055 samples split into 80% training and 20% testing
            2. **Feature Engineering:** 7 new features created from existing ones
            3. **Feature Scaling:** StandardScaler applied to normalize values
            4. **Model Training:** 6 different algorithms trained on the same data
            5. **Evaluation:** Models tested on unseen test data to ensure generalization
            """)
        
        with st.expander("✅ Can I use this in production?", expanded=False):
            st.write("""
            **Yes**, but follow these best practices:
            1. **Choose the right model:** Random Forest has the best accuracy (97.06%)
            2. **Set confidence thresholds:** Only trust predictions >85% confidence
            3. **Regular retraining:** Update models quarterly with new phishing tactics
            4. **Multi-layered defense:** Use as one layer of security, not the only one
            5. **Monitor performance:** Track model accuracy over time
            6. **User feedback loop:** Incorporate user reports to improve detection
            """)
        
        with st.expander("🆚 Which model should I use?", expanded=False):
            st.write("""
            **Recommendations:**
            - **🥇 Production:** Random Forest (Best accuracy, good interpretability)
            - **⚡ Real-time:** Logistic Regression (Fastest predictions, still accurate)
            - **🎯 Precision-focused:** Model with highest precision (minimize false alarms)
            - **🔔 Recall-focused:** Model with highest recall (catch all phishing)
            - **🚀 Maximum accuracy:** XGBoost (State-of-the-art, but slower)
            """)
        
        st.markdown("---")

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
