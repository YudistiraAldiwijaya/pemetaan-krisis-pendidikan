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
    df['Klaster_Label'] = "Klaster " + df['Klaster'].astype(str)
    return df

df = load_data()
daftar_wilayah = sorted(df['Key_Merge'].dropna().unique())

# Inisialisasi Session State agar interaksi peta dan dropdown selaras
if 'pilih_kab' not in st.session_state:
    st.session_state.pilih_kab = daftar_wilayah[0]

pilih_kab_aktif = st.session_state.pilih_kab
data_terpilih = df[df['Key_Merge'] == pilih_kab_aktif].iloc[0]
klaster_aktif = data_terpilih['Klaster']

# ==========================================
# 3. PETA CHOROPLETH INTERAKTIF (PALING ATAS)
# ==========================================
st.subheader("Peta Persebaran Klaster Nasional")

st.markdown(
    f"""
    <div style='background-color: #9b59b6; color: white; padding: 12px; 
    border-radius: 8px; text-align: center; margin-bottom: 15px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 2px solid #8e44ad;'>
    <b>📍 Sedang Menyorot Wilayah: {pilih_kab_aktif} (Warna Ungu)</b>
    </div>
    """, 
    unsafe_allow_html=True
)

df_peta = df.copy()
df_peta['Highlight'] = df_peta['Key_Merge'].apply(lambda x: "Terpilih" if x == pilih_kab_aktif else df_peta.loc[df_peta['Key_Merge'] == x, 'Klaster_Label'].values[0])

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
            "Terpilih": "#9b59b6",
            "Klaster 1": "rgba(46, 204, 113, 0.15)",
            "Klaster 2": "rgba(241, 196, 15, 0.15)", 
            "Klaster 3": "rgba(231, 76, 60, 0.15)",  
            "Klaster 4": "rgba(52, 152, 219, 0.15)"  
        },
        map_style="carto-positron",
        zoom=3.8, 
        center={"lat": -0.7893, "lon": 113.9213},
        opacity=1.0, 
        labels={'Highlight': 'Keterangan'}
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("💡 **Peta Interaktif Dinonaktifkan.** File `indonesia_kabupaten.geojson` tidak ditemukan.")

st.markdown("---")

# ==========================================
# 4. TATA LETAK BAWAH (SPLIT 50:50)
# ==========================================
col1, col2 = st.columns([1, 1.2])

# KIRI: Pencarian & Interpretasi Detail
with col1:
    st.subheader("Pencarian & Diagnosis Wilayah")
    st.selectbox("Ketik atau Pilih Nama Kabupaten/Kota:", daftar_wilayah, key='pilih_kab')
    
    # 4.1 Notifikasi Kebijakan Klaster
    st.markdown("#### Status & Rekomendasi Klaster")
    if klaster_aktif == 3:
        st.error(f"**⚠️ STATUS: DARURAT GANDA (Klaster 3)**\n\nPrioritas utama untuk kebijakan afirmatif relaksasi syarat JTM 24 jam dan perbaikan/revitalisasi infrastruktur sekolah yang rusak berat.")
    elif klaster_aktif == 1:
        st.success(f"**✅ STATUS: BERKECUPAN / STABIL (Klaster 1)**\n\nKapasitas rombel dan beban guru mendekati rata-rata nasional secara stabil. Syarat JTM 24 jam dapat diterapkan secara normal.")
    elif klaster_aktif == 2:
        st.warning(f"**⚠️ STATUS: ANOMALI KEPADATAN NEGERI (Klaster 2)**\n\nRombel sangat padat dan daya serap sekolah negeri rendah. **Hindari asumsi kelangkaan sekolah** dan jangan bangun sekolah negeri baru tanpa mengkaji daya tampung sekolah swasta di wilayah ini.")
    elif klaster_aktif == 4:
        st.info(f"**ℹ️ STATUS: KRISIS ROMBEL RINGAN (Klaster 4)**\n\nMenghadapi kelangkaan rombel dan kerusakan infrastruktur yang mirip dengan Klaster 3, namun berskala lebih ringan. Perlu pemantauan kapasitas.")

    st.markdown("#### 📝 Interpretasi Detail Kondisi Wilayah")
    
    # 4.2 Logika Interpretasi Ramah Awam (Z-Score)
    v1 = data_terpilih['V1_Kapasitas_Rombel_Scaled']
    v2 = data_terpilih['V2_Rasio_Kelas_Guru_Scaled']
    v3 = data_terpilih['V3_Kerapatan_Infra_Scaled']
    v4 = data_terpilih['V4_Penyerapan_Negeri_Scaled']
    v5 = data_terpilih['V5_Tingkat_Kerusakan_Scaled']
    
    st.markdown("**1. Pembacaan 5 Indikator Kinerja (Grafik Z-Score):**")
    if klaster_aktif == 1:
        teks_z = f"Kondisi di wilayah ini seimbang dan berada di dekat rata-rata nasional. Jumlah murid cukup untuk membentuk kelas/rombel yang ideal (Z={v1:.2f}), dan jumlah guru sebanding dengan beban mengajar yang tersedia (Z={v2:.2f}). Infrastruktur juga relatif memadai.\n\n**Dampak terhadap JTM:** Sangat baik. Guru-guru di wilayah ini dapat dengan mudah dan stabil memenuhi syarat mengajar 24 jam per minggu karena ketersediaan kelas dan murid yang mencukupi."
    elif klaster_aktif == 2:
        teks_z = f"Wilayah ini memiliki kelas yang sangat padat (Z={v1:.2f}) dan guru menanggung beban mengajar yang tinggi di atas rata-rata (Z={v2:.2f}). Namun, ketersediaan dan daya serap sekolah negeri di wilayah ini justru tergolong rendah.\n\n**Dampak terhadap JTM:** Secara matematis, guru sangat mudah memenuhi 24 jam mengajar akibat padatnya kelas. Namun, rendahnya daya serap sekolah negeri (Z={v4:.2f}) mengindikasikan bahwa sebagian besar anak usia sekolah di wilayah ini kemungkinan besar diserap oleh sekolah swasta."
    elif klaster_aktif == 3:
        teks_z = f"Wilayah ini mengalami krisis yang parah. Kapasitas kelas sangat kekurangan murid (Z={v1:.2f}), yang berdampak langsung pada sangat minimnya kelas yang bisa diajar oleh setiap guru (Z={v2:.2f}). Kondisi ini diperburuk oleh tingkat kerusakan bangunan sekolah yang amat parah (Z={v5:.2f}).\n\n**Dampak Fatal terhadap JTM:** Guru di wilayah ini hampir mustahil bisa memenuhi syarat 24 jam mengajar. Sedikitnya murid membuat sekolah tidak bisa membuka kelas/rombel baru, sementara fasilitas yang rusak menghambat proses belajar."
    elif klaster_aktif == 4:
        teks_z = f"Wilayah ini mulai menunjukkan gejala kekurangan murid untuk membentuk kelas yang ideal (Z={v1:.2f}), ditambah dengan kondisi kerusakan bangunan sekolah yang cukup memprihatinkan (Z={v5:.2f}) meski belum separah Klaster 3.\n\n**Dampak terhadap JTM:** Terjadi persaingan ketat antar guru untuk mendapatkan jam mengajar. Pemenuhan 24 jam JTM menjadi sangat rentan dan berisiko gagal terpenuhi jika tren penurunan jumlah siswa terus berlanjut di tahun ajaran berikutnya."
    
    st.info(teks_z)
    
    # 4.3 Logika Interpretasi Ramah Awam (PCA)
    st.markdown("**2. Deteksi Akar Masalah (Grafik Skor PCA):**")
    if 'PC1' in data_terpilih and 'PC2' in data_terpilih:
        pc1 = data_terpilih['PC1']
        pc2 = data_terpilih['PC2']
        
        if klaster_aktif == 1:
            teks_pca = f"Nilai skor gabungan PC1 ({pc1:.2f}) dan PC2 ({pc2:.2f}) yang mendekati/dibawah angka nol (0) menunjukkan bahwa wilayah ini memiliki karakteristik yang sangat wajar tanpa ada masalah atau ketimpangan yang ekstrem.\n\n**Kesimpulan Sistem:** Secara keseluruhan, sistem pendidikan tingkat atas di wilayah ini tergolong aman dan normal. Kebijakan wajib 24 jam JTM dapat diterapkan secara seragam tanpa memerlukan perlakuan khusus dari pemerintah pusat."
        elif klaster_aktif == 2:
            teks_pca = f"Skor PC1 ({pc1:.2f}) dan PC2 ({pc2:.2f}) menunjukkan pergeseran pola yang sangat berbeda dari wilayah lain. Hal ini dipicu oleh ketimpangan antara padatnya siswa di sekolah negeri dengan sedikitnya jumlah bangunan sekolah negeri yang tersedia.\n\n**Kesimpulan Sistem:** Angka pemenuhan JTM yang tinggi di wilayah ini adalah hasil dari daya tampung negeri yang terbatas (memaksa siswa ke swasta). Pemerintah perlu sangat berhati-hati sebelum memutuskan membangun sekolah negeri baru agar tidak mematikan sekolah swasta yang sudah berjalan."
        elif klaster_aktif == 3:
            teks_pca = f"Skor PCA mendeteksi adanya 'tarikan' masalah yang sangat kuat, dibuktikan dengan nilai PC1 ({pc1:.2f}) dan PC2 ({pc2:.2f}) yang ekstrem. Sistem membaca kombinasi rusaknya sekolah dan hilangnya murid sebagai krisis tertinggi.\n\n**Kesimpulan Sistem:** Ini adalah bukti matematis terkuat bahwa ketidakmampuan guru memenuhi JTM murni disebabkan oleh kerusakan sistem dan kondisi geografis wilayah, bukan karena kemalasan personal. Kebijakan keringanan (relaksasi) JTM wajib segera diberikan."
        elif klaster_aktif == 4:
            teks_pca = f"Pola data pada PC1 ({pc1:.2f}) dan PC2 ({pc2:.2f}) menunjukkan pergeseran menjauh dari kondisi ideal rata-rata, menandakan mulai munculnya bibit permasalahan pada fasilitas dan jumlah pendaftar kesiswaan.\n\n**Kesimpulan Sistem:** Sistem mendeteksi wilayah ini sebagai zona 'Peringatan Dini' (*Early Warning*). Meski belum darurat, pemerintah pusat dan daerah perlu memantau wilayah ini agar krisis kelangkaan kelas tidak berujung menjadi darurat JTM di masa depan."
            
        st.info(teks_pca)
    else:
        st.info("Nilai PC1 dan PC2 tidak terdeteksi di dalam dataset.")

    st.markdown("<br>", unsafe_allow_html=True)
    csv_wilayah = data_terpilih.to_frame().T.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Unduh Data Mikro {pilih_kab_aktif} (CSV)",
        data=csv_wilayah,
        file_name=f"Data_{pilih_kab_aktif}.csv",
        mime='text/csv'
    )

# KANAN: Grafik Slide (Tabs)
with col2:
    st.subheader(f"Visualisasi Indikator: {pilih_kab_aktif}")
    
    tab1, tab2 = st.tabs(["📊 Grafik 5 Variabel (Z-Score)", "📉 Grafik Deteksi Masalah (PCA)"])
    
    with tab1:
        st.markdown("*(Garis putus-putus 0 = Rata-rata Nasional. Batang mengarah ke bawah = Defisit/Kekurangan)*")
        nilai_fitur = [v1, v2, v3, v4, v5]
        label_indikator = ['Kapasitas Rombel', 'Rasio Rombel per Guru', 'Ketersediaan Sekolah Negeri', 'Penyerapan Sekolah Negeri', 'Kerusakan Sekolah']
        
        df_plot = pd.DataFrame({'Indikator': label_indikator, 'Z-Score': nilai_fitur})
        
        fig1 = px.bar(df_plot, x='Indikator', y='Z-Score', color='Z-Score', color_continuous_scale='RdBu', range_color=[-3, 3], text=df_plot['Z-Score'].round(2))
        fig1.update_traces(textposition='outside')
        fig1.add_hline(y=0, line_dash="dash", line_color="black")
        fig1.update_layout(xaxis_title="", yaxis_title="Skor Perbandingan (Z-Score)", margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("*(Skor gabungan yang menyederhanakan 5 indikator di atas menjadi 2 pola utama untuk mendeteksi akar masalah/krisis)*")
        if 'PC1' in data_terpilih and 'PC2' in data_terpilih:
            pca_fitur = [data_terpilih['PC1'], data_terpilih['PC2']]
            df_pca = pd.DataFrame({
                'Penyederhanaan Pola (PCA)': ['Faktor Utama (PC1)', 'Faktor Sekunder (PC2)'], 
                'Skor Deteksi Masalah': pca_fitur
            })
            
            fig2 = px.bar(
                df_pca, x='Penyederhanaan Pola (PCA)', y='Skor Deteksi Masalah', 
                color='Skor Deteksi Masalah', color_continuous_scale='Viridis', 
                text=df_pca['Skor Deteksi Masalah'].round(2)
            )
            fig2.update_traces(textposition='outside')
            fig2.add_hline(y=0, line_dash="dash", line_color="black")
            fig2.update_layout(xaxis_title="", yaxis_title="Skor Deteksi Masalah (PCA)", margin=dict(t=10, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Data visualisasi PCA tidak tersedia.")
