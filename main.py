import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# === Gaya CSS Kustom dengan Tema Anak-Anak ===
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #ffebee, #e3f2fd, #e8f5e9);
    }
    h1 {
        color: #FF6B6B;
        text-align: center;
        font-family: 'Comic Sans MS', cursive;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #4ECDC4;
        font-family: 'Comic Sans MS', cursive;
    }
    .stDataFrame {
        border: 3px dashed #FFD93D;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FF9A8B, #FF6B6B, #FFD93D);
        color: white;
    }
    .sidebar-header {
        color: white;
        font-weight: bold;
        font-size: 1.3em;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FFD93D);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-family: 'Comic Sans MS', cursive;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
    }
    .stDownloadButton>button {
        background: linear-gradient(45deg, #4ECDC4, #6A89CC);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-family: 'Comic Sans MS', cursive;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# === Konfigurasi ===
st.set_page_config(
    page_title="Klasifikasi Performa Akademik Siswa - SDN 273 Gempol Sari",
    layout="wide",
    page_icon="📊"
)

KMEANS_MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_FILE_PATH = "Data Siswa.csv"

# === Fungsi Clustering (TIDAK DIUBAH) ===
def perform_clustering(df_students):
    features = df_students[["Nilai Akademik", "Kehadiran(%)"]].copy()

    if os.path.exists(KMEANS_MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(KMEANS_MODEL_PATH, 'rb') as f:
                kmeans = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            st.sidebar.success("✅ Model berhasil dimuat")
        except Exception as e:
            st.sidebar.warning(f"⚠ Gagal memuat model: {e}. Membuat model baru...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            with open(KMEANS_MODEL_PATH, 'wb') as f:
                pickle.dump(kmeans, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            st.sidebar.success("✅ Model baru berhasil dibuat")
    else:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        with open(KMEANS_MODEL_PATH, 'wb') as f:
            pickle.dump(kmeans, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        st.sidebar.success("✅ Model berhasil dibuat dan disimpan")

    features_scaled = scaler.transform(features)
    cluster_labels = kmeans.predict(features_scaled)
    df_students["Cluster"] = cluster_labels

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=["Nilai Akademik", "Kehadiran(%)"])
    centroids_df['Rata_Rata'] = (centroids_df['Nilai Akademik'] + centroids_df['Kehadiran(%)']) / 2
    sorted_clusters = centroids_df.sort_values('Rata_Rata', ascending=False).index

    cluster_mapping = {
        int(sorted_clusters[0]): "Tinggi",
        int(sorted_clusters[1]): "Sedang",
        int(sorted_clusters[2]): "Rendah"
    }
    df_students["Kategori"] = df_students["Cluster"].map(cluster_mapping)
    return df_students

# === SIDEBAR ===
with st.sidebar:
    st.markdown('<div class="sidebar-header">🎮 MENU ANALISIS SISWA</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**🔧 Pengaturan Analisis**")
    
    # Filter kategori untuk ditampilkan
    st.subheader("🎯 Filter Kategori")
    show_tinggi = st.checkbox("Tinggi 🟢", value=True)
    show_sedang = st.checkbox("Sedang 🟡", value=True)
    show_rendah = st.checkbox("Rendah 🔴", value=True)
    
    st.markdown("---")
    st.markdown("**📈 Pengaturan Visualisasi**")
    
    # Pengaturan chart
    chart_size = st.slider("Ukuran Chart", min_value=8, max_value=15, value=10)
    point_size = st.slider("Ukuran Titik pada Scatter Plot", min_value=50, max_value=200, value=120)
    
    st.markdown("---")
    st.markdown("**ℹ️ Informasi Aplikasi**")
    
    st.info("""
    Aplikasi ini menggunakan algoritma **K-Means Clustering** untuk mengelompokkan siswa berdasarkan:
    - 📝 Nilai Akademik
    - 🎯 Persentase Kehadiran
    """)
    
    st.markdown("---")
    st.markdown("**🏫 Tentang**")
    st.write("🎒 SDN 273 Gempol Sari")
    st.write("Sistem Pendukung Keputusan Akademik")

# === Judul ===
st.title("📊 Klasifikasi Performa Akademik Siswa")
st.markdown("### Menggunakan Algoritma **K-Means Clustering** Berdasarkan **Nilai Akademik** dan **Kehadiran**")

# === Baca Data ===
try:
    if not os.path.exists(DATA_FILE_PATH):
        st.error(f"❌ File '{DATA_FILE_PATH}' tidak ditemukan!")
        st.info("💡 Pastikan file 'Data Siswa.csv' berada di folder yang sama dengan aplikasi ini.")
    else:
        df = pd.read_csv(DATA_FILE_PATH, sep=';', engine='python')
        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')

        if len(df.columns) >= 3:
            df.columns = ["Nama Siswa", "Nilai Akademik", "Kehadiran(%)"]
        else:
            raise ValueError("File tidak memiliki cukup kolom. Pastikan ada kolom: Nama Siswa, Nilai Akademik, Kehadiran(%)")

        df = df.dropna(subset=["Nama Siswa"]).reset_index(drop=True)
        df["Nilai Akademik"] = pd.to_numeric(df["Nilai Akademik"], errors='coerce')
        df["Kehadiran(%)"] = pd.to_numeric(df["Kehadiran(%)"], errors='coerce')
        df = df.dropna(subset=["Nilai Akademik", "Kehadiran(%)"])

        if df.empty:
            raise ValueError("Tidak ada data valid setelah pembersihan.")

        # Tampilkan info data
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Statistik Cepat**")
        st.sidebar.write(f"Total Data: {len(df)} siswa")
        
        # Clustering
        with st.spinner("🔮 Sedang melakukan clustering... Mohon tunggu"):
            df = perform_clustering(df)

        # Tampilkan statistik kategori di sidebar
        with st.sidebar:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 Tinggi", len(df[df["Kategori"] == "Tinggi"]))
            with col2:
                st.metric("🟡 Sedang", len(df[df["Kategori"] == "Sedang"]))
            with col3:
                st.metric("🔴 Rendah", len(df[df["Kategori"] == "Rendah"]))

        # === TAMPILKAN HASIL DENGAN KATEGORI: TINGGI/SEDANG/RENDAH ===
        st.subheader("🎯 Hasil Klasifikasi Siswa")
        
        # Filter data berdasarkan pilihan di sidebar
        kategori_filter = []
        if show_tinggi:
            kategori_filter.append("Tinggi")
        if show_sedang:
            kategori_filter.append("Sedang")
        if show_rendah:
            kategori_filter.append("Rendah")
        
        df_filtered = df[df["Kategori"].isin(kategori_filter)]
        
        df_display = df_filtered[["Nama Siswa", "Nilai Akademik", "Kehadiran(%)", "Kategori"]].reset_index(drop=True)
        
        # Fungsi untuk styling
        def color_kategori(val):
            if val == "Tinggi":
                return "background-color: #c8e6c9; color: #2e7d32; font-weight: bold; font-family: 'Comic Sans MS';"
            elif val == "Sedang":
                return "background-color: #ffecb3; color: #f57c00; font-weight: bold; font-family: 'Comic Sans MS';"
            elif val == "Rendah":
                return "background-color: #ffcdd2; color: #c62828; font-weight: bold; font-family: 'Comic Sans MS';"
            else:
                return ""
        
        st.dataframe(
            df_display.style.map(color_kategori, subset=["Kategori"]),
            use_container_width=True
        )

        # === Visualisasi ===
        st.subheader("📈 Visualisasi Clustering")
        plt.rcParams.update({'font.family': 'Comic Sans MS', 'font.size': 11})
        fig, ax = plt.subplots(figsize=(chart_size, 6))
        color_map = {"Tinggi": "#4caf50", "Sedang": "#ff9800", "Rendah": "#f44336"}
        
        # Filter data untuk visualisasi juga
        df_viz = df[df["Kategori"].isin(kategori_filter)]
        
        for kategori in ["Tinggi", "Sedang", "Rendah"]:
            if kategori in kategori_filter:
                group = df_viz[df_viz["Kategori"] == kategori]
                if not group.empty:
                    ax.scatter(
                        group["Nilai Akademik"],
                        group["Kehadiran(%)"],
                        label=kategori,
                        color=color_map[kategori],
                        s=point_size,
                        edgecolor='black',
                        linewidth=1,
                        alpha=0.9
                    )
        ax.set_xlabel("Nilai Akademik", fontweight='bold')
        ax.set_ylabel("Kehadiran (%)", fontweight='bold')
        ax.set_title("Klasifikasi Siswa: Nilai vs Kehadiran", fontweight='bold')
        ax.legend(title="Kategori", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(60, 100)
        ax.set_ylim(60, 100)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # === Statistik ===
        st.subheader("📋 Ringkasan Statistik per Kategori")
        stats = df.groupby("Kategori").agg(
            Jumlah_Siswa=("Nama Siswa", "count"),
            Rata_rata_Nilai=("Nilai Akademik", "mean"),
            Rata_rata_Kehadiran=("Kehadiran(%)", "mean")
        ).round(2)
        
        # Hitung persentase
        total_siswa = len(df)
        stats['Persentase'] = (stats['Jumlah_Siswa'] / total_siswa * 100).round(1).astype(str) + '%'
        
        st.table(stats)

        # === DOWNLOAD HASIL ===
        st.subheader("📥 Download Hasil Analisis")
        
        # Convert dataframe to CSV
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(df[["Nama Siswa", "Nilai Akademik", "Kehadiran(%)", "Kategori"]])
        
        st.download_button(
            label="💾 Download CSV Hasil Klasifikasi",
            data=csv,
            file_name="hasil_klasifikasi_siswa.csv",
            mime="text/csv",
        )

        # === PENJELASAN PERFORMA ===
        st.markdown("### 📌 Interpretasi Kategori Performa Akademik")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🟢 **Tinggi** 
            **Performa Sangat Baik**  
            Siswa memiliki nilai akademik tinggi **dan** kehadiran sangat baik. Mereka konsisten dan berpotensi menjadi teladan.
            """)
        
        with col2:
            st.markdown("""
            #### 🟡 **Sedang** 
            **Performa Cukup**  
            Siswa memiliki kombinasi nilai dan kehadiran yang stabil, namun masih ada ruang untuk peningkatan.
            """)
        
        with col3:
            st.markdown("""
            #### 🔴 **Rendah** 
            **Performa Perlu Perhatian**  
            Siswa memiliki nilai dan/atau kehadiran yang relatif rendah. Disarankan untuk pendampingan akademik atau konseling kehadiran.
            """)

        if st.button("🎉 Selesai Analisis", type="primary"):
            st.balloons()
            st.success("Analisis berhasil diselesaikan!")

except Exception as e:
    st.error(f"❌ Terjadi kesalahan: {str(e)}")
    st.info("💡 Pastikan file 'Data Siswa.csv' tersedia dengan format yang benar.")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-family: Comic Sans MS;'>
    <p>Dibuat dengan ❤️ untuk SDN 273 Gempol Sari</p>
</div>
""", unsafe_allow_html=True)
