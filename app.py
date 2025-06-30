import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Set page config
st.set_page_config(page_title="Clustering Sosial Ekonomi", layout="wide", page_icon="📊")

# CSS untuk tampilan modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
    
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --danger: #e74c3c;
        --warning: #f39c12;
        --dark: #2c3e50;
        --light: #ecf0f1;
    }
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .card {
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    .header {
        color: var(--dark);
        text-align: center;
        margin-bottom: 30px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
    }
    
    .cluster-0 { background-color: rgba(46, 204, 113, 0.1); border-left: 4px solid #2ecc71; }
    .cluster-1 { background-color: rgba(231, 76, 60, 0.1); border-left: 4px solid #e74c3c; }
    .cluster-2 { background-color: rgba(243, 156, 18, 0.1); border-left: 4px solid #f39c12; }
    
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .feature-name {
        font-weight: 600;
        color: var(--dark);
        margin-bottom: 5px;
    }
    
    .feature-value {
        font-size: 1.1em;
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<h1 class="header">🧠 Clustering Sosial Ekonomi Kabupaten/Kota di Indonesia</h1>', unsafe_allow_html=True)

# Sidebar dengan informasi aplikasi
with st.sidebar:
    st.markdown("""
    <div class="card">
        <h3>📊 Tentang Aplikasi</h3>
        <p>Aplikasi ini memprediksi cluster untuk data sosial ekonomi kabupaten/kota di Indonesia menggunakan model unsupervised learning:</p>
        <ul>
            <li><b>Model</b>: Autoencoder + KMeans</li>
            <li><b>Jumlah Cluster</b>: 3</li>
            <li><b>Algoritma</b>:
                <ul>
                    <li>Autoencoder untuk reduksi dimensi</li>
                    <li>K-Means untuk clustering</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>📝 Panduan Penggunaan</h3>
        <ol>
            <li>Isi nilai untuk setiap indikator sosial ekonomi</li>
            <li>Klik tombol 'Proses Clustering'</li>
            <li>Hasil cluster akan ditampilkan</li>
            <li>Visualisasi menunjukkan posisi data input</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>ℹ️ Informasi Cluster</h3>
        <div class="cluster-0 card">
            <b>Cluster 0</b>: Kondisi sosial ekonomi relatif baik
        </div>
        <div class="cluster-1 card">
            <b>Cluster 1</b>: Tantangan ekonomi signifikan
        </div>
        <div class="cluster-2 card">
            <b>Cluster 2</b>: Potensi pengembangan lebih lanjut
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan versi TensorFlow untuk debugging
    st.write(f"TensorFlow version: {tf.__version__}")

# Load model dengan penanganan kompatibilitas
@st.cache_resource
def load_models():
    try:
        # Coba muat model encoder
        encoder = tf.keras.models.load_model("encoder_model.keras", compile=False)
    except Exception as e:
        st.error(f"Error loading encoder: {e}")
        st.stop()
    
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()
    
    try:
        kmeans = joblib.load("kmeans_model.pkl")
    except Exception as e:
        st.error(f"Error loading KMeans model: {e}")
        st.stop()
    
    return encoder, scaler, kmeans

try:
    encoder, scaler, kmeans = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.error("Pastikan model tersedia di direktori yang benar:")
    st.error("- encoder_model.keras")
    st.error("- scaler.pkl")
    st.error("- kmeans_model.pkl")
    st.stop()

# Kolom fitur sesuai dengan model
feature_names = [
    "Persentase Penduduk Miskin",
    "Rata-rata Lama Sekolah",
    "Pengeluaran per Kapita",
    "IPM",
    "Umur Harapan Hidup",
    "Akses Sanitasi",
    "Akses Air Minum",
    "Pengangguran Terbuka",
    "Partisipasi Angkatan Kerja",
    "PDRB"
]

kolom_numerik = [
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    "Indeks Pembangunan Manusia",
    "Umur Harapan Hidup (Tahun)",
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    "Tingkat Pengangguran Terbuka",
    "Tingkat Partisipasi Angkatan Kerja",
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)"
]

# Default values (median dari dataset asli)
default_values = {
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)": 14.5,
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)": 8.2,
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)": 10324,
    "Indeks Pembangunan Manusia": 69.22,
    "Umur Harapan Hidup (Tahun)": 67.43,
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak": 69.56,
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak": 78.58,
    "Tingkat Pengangguran Terbuka": 6.46,
    "Tingkat Partisipasi Angkatan Kerja": 62.85,
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)": 1780419
}

# Deskripsi cluster
cluster_descriptions = {
    0: "Kabupaten dengan kondisi sosial ekonomi relatif baik",
    1: "Kabupaten dengan tantangan ekonomi signifikan",
    2: "Kabupaten dengan potensi pengembangan lebih lanjut"
}

cluster_colors = {
    0: "#2ecc71",  # Hijau
    1: "#e74c3c",  # Merah
    2: "#f39c12"   # Oranye
}

cluster_icons = {
    0: "✅",
    1: "⚠️",
    2: "🚀"
}

# Form input
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Input Data Indikator Sosial Ekonomi")
    
    inputs = []
    cols = st.columns(2)
    
    for i, col in enumerate(kolom_numerik):
        with cols[i % 2]:
            # Nama pendek untuk tampilan
            short_name = feature_names[i]
            
            # Tentukan tipe input berdasarkan kolom
            if "PDRB" in col or "Pengeluaran" in col:
                # Format besar untuk nilai ekonomi
                val = st.number_input(
                    label=f"{short_name}",
                    min_value=0,
                    max_value=100000000000,
                    value=int(default_values[col]),
                    step=1000,
                    key=col
                )
                # Format tampilan
                st.caption(f"💵 Nilai: {val:,} (ribu)" if "Pengeluaran" in col else f"💵 Nilai: {val:,}")
            else:
                # Format standar untuk persentase dan nilai desimal
                val = st.number_input(
                    label=f"{short_name}",
                    min_value=0.0,
                    max_value=100.0 if "Persen" in col else 100.0,
                    value=float(default_values[col]),
                    step=0.1,
                    format="%.2f",
                    key=col
                )
                # Format tampilan
                unit = "%" if "Persen" in col else ""
                st.caption(f"📈 Nilai: {val:.2f}{unit}")
                
            inputs.append(float(val))
    
    # Tombol submit
    submitted = st.button("🚀 Proses Clustering", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Proses ketika form disubmit
if submitted:
    with st.spinner('Menganalisis data...'):
        time.sleep(1)  # Simulasi proses
        
        df_input = pd.DataFrame([inputs], columns=kolom_numerik)
        
        try:
            # Preprocessing
            scaled = scaler.transform(df_input.values)
            encoded = encoder.predict(scaled, verbose=0)
            label = kmeans.predict(encoded)[0]
            
            # Tampilkan hasil
            st.markdown(f'<div class="card cluster-{label}">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<h2 style='text-align: center; color: {cluster_colors[label]};'>{cluster_icons[label]} Cluster {label}</h2>", 
                           unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 1.2em;'>{cluster_descriptions[label]}</p>", 
                           unsafe_allow_html=True)
                
                # Indikator utama
                st.markdown("<h4 style='text-align: center;'>📋 Indikator Utama</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Kemiskinan: <b>{inputs[0]:.2f}%</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Lama Sekolah: <b>{inputs[1]:.2f} tahun</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>IPM: <b>{inputs[3]:.2f}</b></p>", unsafe_allow_html=True)
                
            with col2:
                # Visualisasi indikator radial
                fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
                
                # Fitur yang ditampilkan di radar chart (6 fitur pertama)
                features_radar = feature_names[:6]
                values = inputs[:6]
                
                # Normalisasi nilai
                max_val = max(values) * 1.2  # Beri sedikit ruang
                values = [v / max_val for v in values] if max_val > 0 else [0 for _ in values]
                
                # Sudut untuk setiap sumbu
                angles = np.linspace(0, 2 * np.pi, len(features_radar), endpoint=False).tolist()
                values += values[:1]  # Tutup loop
                angles += angles[:1]  # Tutup loop
                
                # Plot data
                ax.plot(angles, values, color=cluster_colors[label], linewidth=2, linestyle='solid')
                ax.fill(angles, values, color=cluster_colors[label], alpha=0.25)
                
                # Label sumbu
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(features_radar)
                ax.set_title('Profil Indikator Sosial Ekonomi', size=14, pad=20)
                ax.set_rlabel_position(30)
                ax.set_ylim(0, 1)
                
                st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Rekomendasi
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💡 Rekomendasi Kebijakan")
            
            if label == 1:
                st.markdown("""
                <div class="feature-card">
                    <h4>📉 Program Pengentasan Kemiskinan</h4>
                    <p>Implementasi program bantuan sosial yang lebih terarah dan intensif untuk keluarga miskin.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="feature-card">
                    <h4>🎓 Peningkatan Akses Pendidikan</h4>
                    <p>Perluasan program beasiswa dan peningkatan kualitas sekolah di daerah terpencil.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="feature-card">
                    <h4>👔 Penciptaan Lapangan Kerja</h4>
                    <p>Insentif untuk investasi industri padat karya dan pengembangan UMKM lokal.</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif label == 2:
                st.markdown("""
                <div class="feature-card">
                    <h4>💡 Optimalisasi Potensi Lokal</h4>
                    <p>Identifikasi dan pengembangan sektor unggulan berbasis sumber daya lokal.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="feature-card">
                    <h4>🏗️ Pengembangan Infrastruktur Dasar</h4>
                    <p>Peningkatan akses transportasi, listrik, dan internet untuk mendukung pertumbuhan ekonomi.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="feature-card">
                    <h4>🏥 Peningkatan Layanan Kesehatan</h4>
                    <p>Pembangunan fasilitas kesehatan dan program kesehatan masyarakat yang lebih merata.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="feature-card">
                    <h4>🚀 Inovasi Pembangunan</h4>
                    <p>Pengembangan program inovatif untuk mempertahankan dan meningkatkan capaian pembangunan.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="feature-card">
                    <h4>🤝 Pemberdayaan Masyarakat</h4>
                    <p>Perluasan program pemberdayaan masyarakat untuk meningkatkan partisipasi warga dalam pembangunan.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="feature-card">
                    <h4>🌿 Pembangunan Berkelanjutan</h4>
                    <p>Integrasi prinsip pembangunan berkelanjutan dalam semua aspek kebijakan daerah.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam pemrosesan: {str(e)}")

# Tampilkan penjelasan tentang indikator jika tidak sedang proses
if not submitted:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Tentang Analisis Clustering")
        
        tab1, tab2 = st.tabs(["Metodologi", "Indikator"])
        
        with tab1:
            st.markdown("""
            <div class="feature-card">
                <h4>🔍 Teknik Unsupervised Learning</h4>
                <p>Aplikasi ini menggunakan pendekatan unsupervised learning untuk mengelompokkan kabupaten/kota berdasarkan kesamaan karakteristik sosial ekonomi:</p>
                <ol>
                    <li><b>Autoencoder</b>: Mengurangi dimensi data dari 10 fitur menjadi fitur laten</li>
                    <li><b>K-Means Clustering</b>: Mengelompokkan data menjadi 3 cluster berdasarkan fitur laten</li>
                </ol>
            </div>
            
            <div class="feature-card">
                <h4>🎯 Tujuan Analisis</h4>
                <p>Identifikasi pola dan karakteristik wilayah untuk:</p>
                <ul>
                    <li>Penyusunan kebijakan yang lebih tepat sasaran</li>
                    <li>Alokasi sumber daya yang lebih efisien</li>
                    <li>Pemantauan perkembangan pembangunan daerah</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="feature-card">
                <h4>📋 Indikator Sosial Ekonomi</h4>
                <p>Analisis menggunakan 10 indikator kunci:</p>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                with cols[i % 2]:
                    # Format nilai default
                    if i in [2, 9]:  # Kolom dengan nilai besar
                        display_value = f"{default_values[kolom_numerik[i]]:,.0f}"
                    else:
                        display_value = f"{default_values[kolom_numerik[i]]:.2f}"
                    
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-name">{i+1}. {feature}</div>
                        <div class="feature-value">{display_value}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)