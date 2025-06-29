import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & scaler
encoder = tf.keras.models.load_model("encoder_model.keras")
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Clustering Sosial Ekonomi Kabupaten/Kota di Indonesia")
st.write("Aplikasi ini menggunakan **autoencoder** dan **KMeans** untuk mengelompokkan data kabupaten/kota berdasarkan indikator sosial ekonomi.")

# Upload data
uploaded_file = st.file_uploader("Upload data CSV dengan format yang sesuai:", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # --- Preprocessing kolom numerik ---
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
        data[kolom] = data[kolom].astype(str).str.replace(",", ".").astype(float)

    X = data[kolom_numerik]
    X_scaled = scaler.transform(X)

    # --- Encode fitur ---
    encoded_features = encoder.predict(X_scaled)

    # --- Cluster ---
    clusters = kmeans.predict(encoded_features)
    data['Cluster'] = clusters

    # --- Visualisasi PCA ---
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(encoded_features)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]

    st.subheader("Hasil Clustering:")
    st.dataframe(data[['Cluster'] + kolom_numerik])

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=80)
    plt.title("Visualisasi Cluster (PCA)")
    st.pyplot(fig)

    # Unduh hasil
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Unduh hasil clustering sebagai CSV", csv, "hasil_clustering.csv", "text/csv")

else:
    st.info("Silakan upload file CSV untuk melihat hasil clustering.")
