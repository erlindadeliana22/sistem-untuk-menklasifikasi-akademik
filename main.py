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
        background: rgba(255, 255, 255, 0.92);
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
        transition: all 0
