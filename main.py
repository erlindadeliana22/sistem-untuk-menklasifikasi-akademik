import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ===============================
# KONFIGURASI
# ===============================
KMEANS_MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_FILE_PATH = "Data Siswa.csv"

# ===============================
# FUNGSI BACKGROUND NEMO
# ===============================
def set_nemo_background():
    """Mengatur background dengan tema kartun Nemo"""
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #FFB347 0%, #FFCC33 25%, #87CEEB 50%, #1E90FF 75%, #006994 100%);
        background-size: 400% 400%;
        animation: oceanGradient 15s ease infinite;
    }
    
    @keyframes oceanGradient {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    .stApp {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 10px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        min-height: 100vh;
    }
    
    h1 {
        color: #FF6B35;
        text-align: center;
        font-family: 'Comic Sans MS', cursive;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #FF6B35, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    
    h2, h3 {
        color: #1E90FF;
        font-family: 'Comic Sans MS', cursive;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .stDataFrame {
        border: 3px solid #FF6B35;
        border-radius: 15px;
        background-color: #E6F3FF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FF8C42, #FF6B35, #1E90FF);
        color: white;
        border-right: 3px solid #FFD700;
    }
    
    .sidebar-header {
        color: white;
        font-weight: bold;
        font-size: 1.3em;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        font-family: 'Comic Sans MS', cursive;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF6B35, #FF8C42);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-family: 'Comic Sans MS', cursive;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(255,107,53,0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(255,107,53,0.4);
        background: linear-gradient(45deg, #FF8C42, #FF6B35);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(45deg, #1E90FF, #00BFFF);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-family: 'Comic Sans MS', cursive;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(30,144,255,0.3);
    }
    
    .stDownloadButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(30,144,255,0.4);
    }
    
    /* Bubble animations */
    .bubble {
        position: fixed;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 50%;
        animation: float 20s infinite ease-in-out;
        z-index: -1;
        border: 2px solid rgba(255, 255, 255, 0.6);
    }
    
    @keyframes float {
        0% { transform: translateY(100vh) scale(0); opacity: 0; }
        10% { opacity: 0.8; }
        90% { opacity: 0.8; }
        100% { transform: translateY(-100vh) scale(1); opacity: 0; }
    }
    
    /* Nemo fish decoration */
    .nemo-fish {
        position: fixed;
        font-size: 30px;
        animation: swim 25s infinite linear;
        z-index: -1;
        filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
    }
    
    @keyframes swim {
        0% { transform: translateX(-100px) translateY(50vh) scaleX(1); }
        50% { transform: translateX(50vw) translateY(30vh) scaleX(1); }
        51% { transform: translateX(50vw) translateY(30vh) scaleX(-1); }
        100% { transform: translateX(100vw) translateY(70vh) scaleX(-1); }
    }
    
    /* Coral reef decoration */
    .coral {
        position: fixed;
        bottom: 0;
        font-size: 40px;
        z-index: -1;
    }
    
    /* Background untuk konten utama */
    .block-container {
        background: linear-gradient(135deg, #E6F3FF, #F0F8FF);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }
    
    /* Style untuk metric cards */
    [data-testid="metric-container"] {
        background-color: #D9F0FF;
        border: 2px solid #99D6FF;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Dekorasi gelembung
    bubble_html = """
    <div class="bubble" style="width: 42px; height: 42px; left: 56%; animation-delay: 4s; animation-duration: 19s;"></div>
    <div class="bubble" style="width: 30px; height: 30px; left: 59%; animation-delay: 3s; animation-duration: 21s;"></div>
    <div class="bubble" style="width: 46px; height: 46px; left: 27%; animation-delay: 1s; animation-duration: 16s;"></div>
    <div class="bubble" style="width: 57px; height: 57px; left: 33%; animation-delay: 3s; animation-duration: 16s;"></div>
    <div class="bubble" style="width: 26px; height: 26px; left: 61%; animation-delay: 0s; animation-duration: 24s;"></div>
    <div class="bubble" style="width: 47px; height: 47px; left: 31%; animation-delay: 3s; animation-duration: 20s;"></div>
    <div class="bubble" style="width: 27px; height: 27px; left: 54%; animation-delay: 7s; animation-duration: 24s;"></div>
    <div class="bubble" style="width: 54px; height: 54px; left: 62%; animation-delay: 5s; animation-duration: 15s;"></div>
    <div class="bubble" style="width: 20px; height: 20px; left: 60%; animation-delay: 8s; animation-duration: 18s;"></div>
    <div class="bubble" style="width: 42px; height: 42px; left: 76%; animation-delay: 9s; animation-duration: 20s;"></div>
    <div class="bubble" style="width: 28px; height: 28px; left: 13%; animation-delay: 1s; animation-duration: 16s;"></div>
    <div class="bubble" style="width: 38px; height: 38px; left: 0%; animation-delay: 1s; animation-duration: 19s;"></div>
    <div class="bubble" style="width: 33px; height: 33px; left: 69%; animation-delay: 3s; animation-duration: 23s;"></div>
    <div class="bubble" style="width: 48px; height: 48px; left: 47%; animation-delay: 0s; animation-duration: 18s;"></div>
    """
    
    # Ikan dan dekorasi laut
    nemo_html = """
    <div class="nemo-fish" style="top: 20%; animation-delay: 0s;">🐠</div>
    <div class="nemo-fish" style="top: 60%; animation-delay: 5s; animation-duration: 30s;">🐡</div>
    <div class="nemo-fish" style="top: 80%; animation-delay: 10s; animation-duration: 20s;">🐟</div>
    <div class="nemo-fish" style="top: 40%; animation-delay: 15s; animation-duration: 25s;">🦈</div>
    """
    
    coral_html = """
    <div class="coral" style="left: 10%;">🌊</div>
    <div class="coral" style="left: 25%;">🐚</div>
    <div class="coral" style="left: 40%;">🌴</div>
    <div class="coral" style="left: 70%;">🌊</div>
    <div class="coral" style="left: 85%;">🐚</div>
    """
    
    st.markdown(bubble_html + nemo_html + coral_html, unsafe_allow_html=True)

# ===============================
# FUNGSI CLUSTERING
# ===============================
def perform_clustering(df_students):
    """Melakukan clustering menggunakan K-Means"""
    features = df_students[["Nilai Akademik", "Kehadiran(%)"]].copy()

    # Cek apakah model sudah ada
    if os.path.exists(KMEANS_MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(KMEANS_MODEL_PATH, 'rb') as f:
                kmeans = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            st.sidebar.success("✅ Model berhasil dimuat")
        except Exception as e:
            st.sidebar.warning(f"⚠ Gagal memuat model: {e}. Membuat model baru...")
            kmeans, scaler = create_new_model(features)
    else:
        kmeans, scaler = create_new_model(features)

    # Prediksi cluster
    features_scaled = scaler.transform(features)
    cluster_labels = kmeans.predict(features_scaled)
    df_students["Cluster"] = cluster_labels

    # Mapping cluster ke kategori
    cluster_mapping = map_clusters_to_categories(kmeans, scaler)
    df_students["Kategori"] = df_students["Cluster"].map(cluster_mapping)
    
    return df_students

def create_new_model(features):
    """Membuat model K-Means baru"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    
    # Simpan model
    with open(KMEANS_MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    st.sidebar.success("✅ Model baru berhasil dibuat dan disimpan")
    return kmeans, scaler

def map_clusters_to_categories(kmeans, scaler):
    """Mapping cluster ke kategori Tinggi, Sedang, Rendah"""
    centroids_scaled = kmeans.cluster_centers_
    centro
