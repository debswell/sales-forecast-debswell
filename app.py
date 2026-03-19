import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Konfigurasi halaman ─────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penjualan Shopee",
    page_icon="🛍️",
    layout="wide"
)

# ── Load model & data ───────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model = joblib.load('model_xgboost_tuned.pkl')
    rf_model  = joblib.load('model_rf_tuned.pkl')
    le_produk = joblib.load('label_encoder_produk.pkl')
    le_kat    = joblib.load('label_encoder_kategori.pkl')
    return xgb_model, rf_model, le_produk, le_kat

@st.cache_data
def load_data():
    return pd.read_csv('data_model.csv')

xgb_model, rf_model, le_produk, le_kat = load_models()
df = load_data()

# ── Header ──────────────────────────────────────────────────
st.title("🛍️ Sistem Prediksi Penjualan Produk Shopee")
st.markdown("**Metode:** Random Forest & XGBoost | **Framework:** CRISP-DM")
st.divider()

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Parameter Prediksi")

daftar_produk = sorted(df['Nama Produk'].unique().tolist())
produk_pilihan = st.sidebar.selectbox("Pilih Produk", daftar_produk)

bulan_dict = {
    "Januari":1,"Februari":2,"Maret":3,"April":4,
    "Mei":5,"Juni":6,"Juli":7,"Agustus":8,
    "September":9,"Oktober":10,"November":11,"Desember":12
}
bulan_nama   = st.sidebar.selectbox("Bulan Prediksi", list(bulan_dict.keys()))
bulan_pilihan = bulan_dict[bulan_nama]

model_pilihan = st.sidebar.radio(
    "Pilih Model",
    ["XGBoost (Tuned) 🏆", "Random Forest"]
)

is_event = st.sidebar.checkbox("🎉 Ada Event Bulan Ini?", value=False)
st.sidebar.caption("Contoh: Ramadhan, Lebaran, Nataru, Payday, Double Date")

st.sidebar.divider()
st.sidebar.markdown("**Info Model:**")
st.sidebar.markdown("- XGBoost Tuned: R² = 0.789")
st.sidebar.markdown("- Random Forest: R² = 0.779")

# ── Ambil data historis produk terpilih ─────────────────────
df_produk = df[df['Nama Produk'] == produk_pilihan].sort_values('Bulan')

# ── Hitung fitur untuk prediksi ─────────────────────────────
def buat_fitur(produk, bulan, is_event, df):
    dp = df[df['Nama Produk'] == produk].sort_values('Bulan')

    lag1 = dp[dp['Bulan'] == bulan - 1]['Total_Terjual'].values
    lag2 = dp[dp['Bulan'] == bulan - 2]['Total_Terjual'].values
    lag3 = dp[dp['Bulan'] == bulan - 3]['Total_Terjual'].values

    lag1 = float(lag1[0]) if len(lag1) > 0 else 0.0
    lag2 = float(lag2[0]) if len(lag2) > 0 else 0.0
    lag3 = float(lag3[0]) if len(lag3) > 0 else 0.0

    rolling3 = dp[dp['Bulan'] < bulan]['Total_Terjual'].tail(3).mean()
    rolling6 = dp[dp['Bulan'] < bulan]['Total_Terjual'].tail(6).mean()
    rolling3 = rolling3 if not np.isnan(rolling3) else 0.0
    rolling6 = rolling6 if not np.isnan(rolling6) else 0.0

    tren         = lag1 - lag2
    rata_harga   = dp['Rata_Harga'].mean()
    total_diskon = dp['Total_Diskon'].mean()
    kategori     = produk.split()[0]

    try:
        produk_enc = le_produk.transform([produk])[0]
    except:
        produk_enc = 0
    try:
        kat_enc = le_kat.transform([kategori])[0]
    except:
        kat_enc = 0

    fitur = pd.DataFrame([{
        'Produk_Encoded'  : produk_enc,
        'Kategori_Encoded': kat_enc,
        'Bulan'           : bulan,
        'Rata_Harga'      : rata_harga,
        'Total_Diskon'    : total_diskon,
        'Ada_Event'       : int(is_event),
        'Lag_1'           : lag1,
        'Lag_2'           : lag2,
        'Lag_3'           : lag3,
        'Rolling_Mean_3'  : rolling3,
        'Rolling_Mean_6'  : rolling6,
        'Tren'            : tren,
    }])
    return fitur

# ── Prediksi ────────────────────────────────────────────────
fitur_input = buat_fitur(produk_pilihan, bulan_pilihan, is_event, df)

if "XGBoost" in model_pilihan:
    prediksi = xgb_model.predict(fitur_input)[0]
    nama_model = "XGBoost (Tuned)"
    r2_model   = 0.789
else:
    prediksi = rf_model.predict(fitur_input)[0]
    nama_model = "Random Forest"
    r2_model   = 0.779

prediksi = max(0, round(prediksi))

# Rekomendasi stok
stok_bulan_lalu = df_produk[df_produk['Bulan'] == bulan_pilihan - 1]['Total_Terjual'].values
stok_lalu = int(stok_bulan_lalu[0]) if len(stok_bulan_lalu) > 0 else 0
selisih   = prediksi - stok_lalu

if selisih > 0:
    rekomendasi = f"📈 Tambah stok +{selisih} unit dari bulan lalu"
    warna_rek   = "normal"
elif selisih < 0:
    rekomendasi = f"📉 Kurangi stok {abs(selisih)} unit dari bulan lalu"
    warna_rek   = "inverse"
else:
    rekomendasi = "✅ Pertahankan stok seperti bulan lalu"
    warna_rek   = "off"

# ── Layout utama ────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label=f"🎯 Prediksi Penjualan — {bulan_nama}",
        value=f"{prediksi} unit",
        delta=f"{selisih:+d} vs bulan lalu"
    )

with col2:
    st.metric(
        label="🤖 Model Digunakan",
        value=nama_model,
        delta=f"R² = {r2_model}"
    )

with col3:
    st.metric(
        label="📦 Rekomendasi Stok",
        value=rekomendasi
    )

st.divider()

# ── Grafik tren penjualan ───────────────────────────────────
st.subheader(f"📊 Tren Penjualan — {produk_pilihan}")

bulan_label = ["Jan","Feb","Mar","Apr","Mei","Jun",
               "Jul","Agt","Sep","Okt","Nov","Des"]

fig = go.Figure()

# Garis aktual
fig.add_trace(go.Scatter(
    x=[bulan_label[b-1] for b in df_produk['Bulan']],
    y=df_produk['Total_Terjual'],
    mode='lines+markers',
    name='Aktual',
    line=dict(color='#378ADD', width=2),
    marker=dict(size=7)
))

# Titik prediksi
fig.add_trace(go.Scatter(
    x=[bulan_label[bulan_pilihan-1]],
    y=[prediksi],
    mode='markers',
    name=f'Prediksi {bulan_nama}',
    marker=dict(color='#E24B4A', size=14, symbol='star')
))

fig.update_layout(
    xaxis_title="Bulan",
    yaxis_title="Total Terjual (unit)",
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ── Tabel historis ──────────────────────────────────────────
st.subheader("📋 Data Historis Penjualan")
df_tampil = df_produk[['Bulan','Total_Terjual','Total_Transaksi','Rata_Harga','Ada_Event']].copy()
df_tampil['Bulan'] = df_tampil['Bulan'].apply(lambda x: bulan_label[int(x)-1])
df_tampil.columns = ['Bulan','Total Terjual','Total Transaksi','Rata-rata Harga','Ada Event']
st.dataframe(df_tampil, use_container_width=True, hide_index=True)

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("📌 Sistem Prediksi Penjualan Shopee | Metode CRISP-DM | Model: XGBoost & Random Forest")