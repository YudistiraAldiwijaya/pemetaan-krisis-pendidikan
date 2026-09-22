import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Deteksi Krisis Pendidikan", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Sistem Deteksi Ketimpangan Infrastruktur & Beban Kerja Guru")
st.markdown("Berdasarkan Analisis Unsupervised Learning (Affinity Propagation + PCA) pada 514 Kabupaten/Kota")
st.markdown("---")

# ==========================================
# 2. FUNGSI MEMUAT DATA
# ==========================================
@st.cache_data
def load_data():
    file_path = "Hasil_Clustering_AP_Final.xlsx"
    df = pd.read_excel(file_path, sheet_name="Dataset_Klaster_Final")
    # Pastikan tipe data Klaster berupa string/kategori agar warnanya diskrit di peta
    df['Klaster_Label'] = "Klaster " + df['Klaster'].astype(str)
    return df

df = load_data()

# ==========================================
# 3. FITUR TAMBAHAN 1: METRIC CARDS (POPULASI KLASTER)
# ==========================================
st.subheader("Ringkasan Distribusi Wilayah")
m1, m2, m3, m4 = st.columns(4)
m1.metric("✅ Klaster 1 (Berkecukupan)", f"{len(df[df['Klaster']==1])} Kab/Kota")
m2.metric("⚠️ Klaster 2 (Kelangkaan Sekolah)", f"{len(df[df['Klaster']==2])} Kab/Kota")
m3.metric("🚨 Klaster 3 (Darurat Ganda)", f"{len(df[df['Klaster']==3])} Kab/Kota")
m4.metric("ℹ️ Klaster 4 (Krisis Ringan)", f"{len(df[df['Klaster']==4])} Kab/Kota")
st.markdown("---")

# ==========================================
# 4. TATA LETAK GRID UTAMA
# ==========================================
col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader("Pencarian & Diagnosis Wilayah")
    daftar_wilayah = sorted(df['Key_Merge'].dropna().unique())
    pilih_kab = st.selectbox("Ketik atau Pilih Nama Kabupaten/Kota:", daftar_wilayah)
    
    data_terpilih = df[df['Key_Merge'] == pilih_kab].iloc[0]
    klaster_aktif = data_terpilih['Klaster']
    
    # Notifikasi Kebijakan
    if klaster_aktif == 3:
        st.error("**⚠️ STATUS: DARURAT GANDA**\n\nWilayah ini menderita krisis ekstrem berupa kelangkaan Rombel sekaligus tingkat kerusakan sekolah yang tinggi. Prioritas utama untuk relaksasi syarat JTM 24 jam.")
    elif klaster_aktif == 1:
        st.success("**✅ STATUS: BERKECUPAN (UNGGUL)**\n\nKapasitas beban kerja guru dan kelayakan infrastruktur di wilayah ini terpelihara di atas rata-rata nasional.")
    elif klaster_aktif == 2:
        st.warning("**⚠️ STATUS: KELANGKAAN SEKOLAH (MENENGAH)**\n\nRombel cenderung padat dan rasio kelas per guru tinggi, namun tingkat ketersediaan sekolah rendah. Fokuskan pada pendirian Unit Sekolah Baru (USB).")
    elif klaster_aktif == 4:
        st.info("**ℹ️ STATUS: KRISIS ROMBEL RINGAN**\n\nTerdapat kelangkaan rombel dan kerusakan infrastruktur namun belum pada tahap ekstrem. Perlu pemantauan kapasitas.")

    # FITUR TAMBAHAN 3: TOMBOL DOWNLOAD DATA
    st.markdown("<br>", unsafe_allow_html=True)
    csv_wilayah = data_terpilih.to_frame().T.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Unduh Data Kuantitatif {pilih_kab} (CSV)",
        data=csv_wilayah,
        file_name=f"Data_{pilih_kab}.csv",
        mime='text/csv'
    )

with col2:
    st.subheader(f"Profil Z-Score: {pilih_kab}")
    st.markdown("*(Garis putus-putus = Rata-rata Nasional. Batang ke bawah = Defisit/Di bawah rata-rata)*")
    
    nilai_fitur = [
        data_terpilih['V1_Kapasitas_Rombel_Scaled'],
        data_terpilih['V2_Rasio_Kelas_Guru_Scaled'],
        data_terpilih['V3_Kerapatan_Infra_Scaled'],
        data_terpilih['V4_Penyerapan_Negeri_Scaled'],
        data_terpilih['V5_Tingkat_Kerusakan_Scaled']
    ]
    label_indikator = ['Kapasitas Rombel', 'Rasio Kelas/Guru', 'Ketersediaan Sekolah', 'Penyerapan Negeri', 'Kerusakan Sekolah']
    
    df_plot = pd.DataFrame({'Indikator': label_indikator, 'Z-Score': nilai_fitur})
    
    fig = px.bar(df_plot, x='Indikator', y='Z-Score', color='Z-Score', color_continuous_scale='RdBu', range_color=[-3, 3], text=df_plot['Z-Score'].round(2))
    fig.update_traces(textposition='outside')
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(xaxis_title="", yaxis_title="Nilai Z-Score", margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. FITUR TAMBAHAN 2: PETA CHOROPLETH INTERAKTIF
# ==========================================
st.markdown("---")
st.subheader("Peta Persebaran Klaster Nasional")

# Menambahkan Bubble Label Notifikasi di atas peta (menggunakan HTML/CSS injeksi di Streamlit)
st.markdown(
    f"""
    <div style='background-color: #9b59b6; color: white; padding: 12px; 
    border-radius: 8px; text-align: center; margin-bottom: 15px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 2px solid #8e44ad;'>
    <b>📍 Sedang Menyorot Wilayah: {pilih_kab} (Warna Ungu)</b>
    </div>
    """, 
    unsafe_allow_html=True
)

df_peta = df.copy()
df_peta['Highlight'] = df_peta['Key_Merge'].apply(lambda x: "Terpilih" if x == pilih_kab else df_peta.loc[df_peta['Key_Merge'] == x, 'Klaster_Label'].values[0])

geojson_path = "indonesia_kabupaten.geojson"

if os.path.exists(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_indo = json.load(f)
    
    fig_map = px.choropleth_map(
        df_peta, 
        geojson=geojson_indo, 
        locations='Key_Merge', 
        featureidkey="properties.WADMKK", 
        color='Highlight',
        color_discrete_map={
            "Terpilih": "#9b59b6", # Warna Ungu Solid & Terang (Amethyst)
            "Klaster 1": "rgba(46, 204, 113, 0.15)", # Transparansi diturunkan drastis menjadi 15%
            "Klaster 2": "rgba(241, 196, 15, 0.15)", 
            "Klaster 3": "rgba(231, 76, 60, 0.15)",  
            "Klaster 4": "rgba(52, 152, 219, 0.15)"  
        },
        map_style="carto-positron",
        zoom=3.5, 
        center={"lat": -0.7893, "lon": 113.9213},
        opacity=1.0, # Opacity global dimaksimalkan agar warna ungu solid
        labels={'Highlight': 'Keterangan'}
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("💡 **Peta Interaktif Dinonaktifkan.** File `indonesia_kabupaten.geojson` tidak ditemukan.")
# ==========================================
# 6. UNDUH SELURUH DATA
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
csv_full = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Unduh Seluruh Hasil Klasterisasi (514 Kab/Kota) format CSV",
    data=csv_full,
    file_name="Hasil_Klasterisasi_Nasional.csv",
    mime='text/csv'
)