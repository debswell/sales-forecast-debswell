import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ── Konfigurasi halaman ─────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penjualan Shopee",
    page_icon="🛍️",
    layout="wide"
)

# ── Load model & data ───────────────────────────────────────
@st.cache_resource
def load_models():
    with open('model_xgboost_tuned.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('model_rf_tuned.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('label_encoder_produk.pkl', 'rb') as f:
        le_produk = pickle.load(f)
    with open('label_encoder_kategori.pkl', 'rb') as f:
        le_kat = pickle.load(f)
    return xgb_model, rf_model, le_produk, le_kat

@st.cache_data
def load_data():
    return pd.read_csv('data_model.csv')

try:
    xgb_model, rf_model, le_produk, le_kat = load_models()
    df = load_data()
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# ── Mapping event ke nilai is_event ─────────────────────────
event_map = {
    "Tidak Ada Event" : 0,
    "Ramadhan"        : 1,
    "Lebaran"         : 1,
    "Nataru"          : 1,
    "Payday"          : 1,
    "Double Date"     : 1,
}

# ── Fungsi buat fitur ────────────────────────────────────────
def buat_fitur(produk, bulan, is_event_val, df, le_produk, le_kat):
    dp = df[df['Nama Produk'] == produk].sort_values('Bulan')

    lag1 = dp[dp['Bulan'] == bulan - 1]['Total_Terjual'].values
    lag2 = dp[dp['Bulan'] == bulan - 2]['Total_Terjual'].values
    lag3 = dp[dp['Bulan'] == bulan - 3]['Total_Terjual'].values

    lag1 = float(lag1[0]) if len(lag1) > 0 else 0.0
    lag2 = float(lag2[0]) if len(lag2) > 0 else 0.0
    lag3 = float(lag3[0]) if len(lag3) > 0 else 0.0

    rolling3 = dp[dp['Bulan'] < bulan]['Total_Terjual'].tail(3).mean()
    rolling6 = dp[dp['Bulan'] < bulan]['Total_Terjual'].tail(6).mean()
    rolling3 = float(rolling3) if not np.isnan(rolling3) else 0.0
    rolling6 = float(rolling6) if not np.isnan(rolling6) else 0.0

    tren         = lag1 - lag2
    rata_harga   = float(dp['Rata_Harga'].mean())
    total_diskon = float(dp['Total_Diskon'].mean())
    kategori     = produk.split()[0]

    try:
        produk_enc = int(le_produk.transform([produk])[0])
    except:
        produk_enc = 0
    try:
        kat_enc = int(le_kat.transform([kategori])[0])
    except:
        kat_enc = 0

    return pd.DataFrame([{
        'Produk_Encoded'  : produk_enc,
        'Kategori_Encoded': kat_enc,
        'Bulan'           : bulan,
        'Rata_Harga'      : rata_harga,
        'Total_Diskon'    : total_diskon,
        'Ada_Event'       : is_event_val,
        'Lag_1'           : lag1,
        'Lag_2'           : lag2,
        'Lag_3'           : lag3,
        'Rolling_Mean_3'  : rolling3,
        'Rolling_Mean_6'  : rolling6,
        'Tren'            : tren,
    }])

# ── Header ──────────────────────────────────────────────────
st.title("🛍️ Sistem Prediksi Penjualan Produk Shopee")
st.markdown("**Metode:** Random Forest & XGBoost | **Framework:** CRISP-DM")
st.divider()

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("⚙️ Parameter Prediksi")

mode = st.sidebar.radio(
    "Mode Prediksi",
    ["🔍 Satu Produk", "📦 Semua Produk (Ranking)"]
)

bulan_dict = {
    "Januari":1,"Februari":2,"Maret":3,"April":4,
    "Mei":5,"Juni":6,"Juli":7,"Agustus":8,
    "September":9,"Oktober":10,"November":11,"Desember":12
}
bulan_nama    = st.sidebar.selectbox("Bulan Prediksi", list(bulan_dict.keys()))
bulan_pilihan = bulan_dict[bulan_nama]

model_pilihan = st.sidebar.radio(
    "Pilih Model",
    ["XGBoost (Tuned) 🏆", "Random Forest"]
)

# Dropdown event spesifik
event_pilihan = st.sidebar.selectbox(
    "🎉 Event Bulan Ini",
    list(event_map.keys())
)
is_event_val = event_map[event_pilihan]

st.sidebar.divider()
st.sidebar.markdown("**Info Model:**")
st.sidebar.markdown("- 🏆 XGBoost Tuned : MAE=1.207 | R²=0.789")
st.sidebar.markdown("- 🌲 Random Forest : MAE=1.174 | R²=0.779")

bulan_label = ["Jan","Feb","Mar","Apr","Mei","Jun",
               "Jul","Agt","Sep","Okt","Nov","Des"]

# ══════════════════════════════════════════════════
# MODE 1: SATU PRODUK
# ══════════════════════════════════════════════════
if mode == "🔍 Satu Produk":

    daftar_produk  = sorted(df['Nama Produk'].unique().tolist())
    produk_pilihan = st.sidebar.selectbox("Pilih Produk", daftar_produk)
    df_produk      = df[df['Nama Produk'] == produk_pilihan].sort_values('Bulan')

    fitur_input = buat_fitur(
        produk_pilihan, bulan_pilihan, is_event_val, df, le_produk, le_kat
    )

    try:
        if "XGBoost" in model_pilihan:
            prediksi   = xgb_model.predict(fitur_input)[0]
            nama_model = "XGBoost (Tuned)"
            r2_model   = 0.789
            mae_model  = 1.207
        else:
            prediksi   = rf_model.predict(fitur_input)[0]
            nama_model = "Random Forest"
            r2_model   = 0.779
            mae_model  = 1.174

        prediksi = max(0, round(float(prediksi)))
    except Exception as e:
        st.error(f"❌ Gagal prediksi: {e}")
        st.stop()

    stok_lalu = df_produk[df_produk['Bulan'] == bulan_pilihan - 1]['Total_Terjual'].values
    stok_lalu = int(stok_lalu[0]) if len(stok_lalu) > 0 else 0
    selisih   = prediksi - stok_lalu

    if selisih > 2:
        rekomendasi = f"📈 Tambah stok +{selisih} unit"
    elif selisih < -2:
        rekomendasi = f"📉 Kurangi stok {abs(selisih)} unit"
    else:
        rekomendasi = "✅ Pertahankan stok"

    # Metrik
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"🎯 Prediksi — {bulan_nama}",
                  f"{prediksi} unit", f"{selisih:+d} vs bulan lalu")
    with col2:
        st.metric("🤖 Model", nama_model, f"R² = {r2_model}")
    with col3:
        st.metric("📏 MAE", f"±{mae_model} unit", "rata-rata error")
    with col4:
        st.metric("📦 Rekomendasi", rekomendasi)

    st.divider()

    # Grafik tren
    st.subheader(f"📊 Tren Penjualan — {produk_pilihan}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[bulan_label[int(b)-1] for b in df_produk['Bulan']],
        y=df_produk['Total_Terjual'],
        mode='lines+markers', name='Aktual 2024',
        line=dict(color='#378ADD', width=2), marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=[bulan_label[bulan_pilihan-1]], y=[prediksi],
        mode='markers', name=f'Prediksi {bulan_nama}',
        marker=dict(color='#E24B4A', size=16, symbol='star')
    ))
    fig.add_trace(go.Scatter(
        x=[bulan_label[bulan_pilihan-1], bulan_label[bulan_pilihan-1]],
        y=[max(0, prediksi - mae_model), prediksi + mae_model],
        mode='lines', name='Rentang ±MAE',
        line=dict(color='#E24B4A', width=1, dash='dot')
    ))
    fig.update_layout(
        xaxis_title="Bulan", yaxis_title="Total Terjual (unit)",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabel historis
    st.subheader("📋 Data Historis Penjualan")
    df_tampil = df_produk[['Bulan','Total_Terjual','Total_Transaksi',
                            'Rata_Harga','Ada_Event']].copy()
    df_tampil['Bulan']     = df_tampil['Bulan'].apply(lambda x: bulan_label[int(x)-1])
    df_tampil['Ada_Event'] = df_tampil['Ada_Event'].apply(
        lambda x: "✅ Ya" if x == 1 else "—"
    )
    df_tampil.columns = ['Bulan','Total Terjual','Total Transaksi',
                         'Rata-rata Harga (Rp)','Ada Event']
    st.dataframe(df_tampil, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════
# MODE 2: SEMUA PRODUK (RANKING)
# ══════════════════════════════════════════════════
else:
    st.subheader(f"📦 Prediksi Semua Produk — {bulan_nama} "
                 f"| Event: {event_pilihan}")

    with st.spinner("⏳ Sedang memprediksi semua produk..."):
        hasil = []
        model = xgb_model if "XGBoost" in model_pilihan else rf_model
        mae_model = 1.207 if "XGBoost" in model_pilihan else 1.174

        for produk in df['Nama Produk'].unique():
            try:
                fitur = buat_fitur(
                    produk, bulan_pilihan, is_event_val,
                    df, le_produk, le_kat
                )
                pred = max(0, round(float(model.predict(fitur)[0])))

                # Stok bulan lalu
                dp = df[df['Nama Produk'] == produk]
                stok_lalu = dp[dp['Bulan'] == bulan_pilihan - 1]['Total_Terjual'].values
                stok_lalu = int(stok_lalu[0]) if len(stok_lalu) > 0 else 0
                selisih   = pred - stok_lalu

                if selisih > 2:
                    rek = f"📈 Tambah +{selisih}"
                elif selisih < -2:
                    rek = f"📉 Kurangi {abs(selisih)}"
                else:
                    rek = "✅ Pertahankan"

                hasil.append({
                    'Nama Produk'         : produk,
                    'Prediksi (unit)'     : pred,
                    'Stok Bulan Lalu'     : stok_lalu,
                    'Selisih'             : selisih,
                    'Rekomendasi Stok'    : rek,
                })
            except:
                pass

    df_hasil = pd.DataFrame(hasil).sort_values(
        'Prediksi (unit)', ascending=False
    ).reset_index(drop=True)
    df_hasil.index += 1

    # Metrik ringkasan
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Total Produk Diprediksi", f"{len(df_hasil)} produk")
    with col2:
        st.metric("🔝 Prediksi Tertinggi",
                  f"{df_hasil.iloc[0]['Prediksi (unit)']} unit",
                  df_hasil.iloc[0]['Nama Produk'][:30])
    with col3:
        total = df_hasil['Prediksi (unit)'].sum()
        st.metric("📊 Total Prediksi Semua Produk", f"{total} unit")

    st.divider()

    # Grafik top 15 produk
    st.subheader("🏆 Top 15 Produk dengan Prediksi Tertinggi")
    top15 = df_hasil.head(15)
    fig = px.bar(
        top15,
        x='Prediksi (unit)',
        y='Nama Produk',
        orientation='h',
        color='Prediksi (unit)',
        color_continuous_scale='Blues',
        text='Prediksi (unit)'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=550,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabel lengkap semua produk
    st.subheader("📋 Tabel Lengkap Prediksi Semua Produk")
    st.dataframe(df_hasil, use_container_width=True)

    # Tombol download
    csv = df_hasil.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Hasil Prediksi (CSV)",
        data=csv,
        file_name=f'prediksi_{bulan_nama.lower()}_2025.csv',
        mime='text/csv'
    )

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("📌 Sistem Prediksi Penjualan Shopee | CRISP-DM | "
           "XGBoost (Tuned) R²=0.789 | Random Forest R²=0.779")