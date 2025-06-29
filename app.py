import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan preprocessing
encoder = tf.keras.models.load_model("encoder_model.keras")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.set_page_config(page_title="Clustering Sosial Ekonomi", layout="wide")
st.title("🧠 Clustering Sosial Ekonomi Kabupaten/Kota di Indonesia")
st.write("Gunakan model Autoencoder + KMeans untuk melihat pengelompokan kabupaten/kota berdasarkan indikator sosial ekonomi.")

uploaded_file = st.file_uploader("📂 Upload data CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Awal")
    st.dataframe(df.head())

    kolom_numerik = [
        'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)',
        'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)',
        'Indeks Pembangunan Manusia',
        'Umur Harapan Hidup (Tahun)',
        'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak',
        'Persentase rumah tangga yang memiliki akses terhadap air minum layak',
        'Tingkat Pengangguran Terbuka',
        'Tingkat Partisipasi Angkatan Kerja'
    ]

    for kolom in kolom_numerik:
        df[kolom] = df[kolom].astype(str).str.replace(",", ".").astype(float)

    # Skala dan encode
    X_scaled = scaler.transform(df[kolom_numerik])
    encoded = encoder.predict(X_scaled)
    cluster_labels = kmeans.predict(encoded)
    df['Cluster'] = cluster_labels

    # Visualisasi PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(encoded)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    st.subheader("📊 Visualisasi Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=80)
    plt.title("Visualisasi Cluster dengan PCA")
    st.pyplot(fig)

    # Unduh hasil
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Hasil Clustering", csv, "hasil_clustering.csv", "text/csv")

else:
    st.info("Silakan upload file CSV yang sesuai format.")
