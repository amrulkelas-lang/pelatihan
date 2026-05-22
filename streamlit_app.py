import pickle
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

app_dir = Path(__file__).resolve().parent
# Load Data
data = pd.read_csv(app_dir / "data" / "02_realisasi_anggaran_klasifikasi.csv")

model_path = app_dir / "model" / "Best_model.pkcls"

tab_analitik, tab_prediksi = st.tabs(["Analitik", "Prediksi Model"])

with tab_analitik:
    st.subheader("Data Sample")
    st.dataframe(data.head())

    st.subheader("Scatterplot: Skor IKPA vs Deviasi RPD")
    chart = alt.Chart(data).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X("skor_ikpa:Q", title="Skor IKPA"),
        y=alt.Y("deviasi_rpd_persen:Q", title="Deviasi RPD (%)"),
        color=alt.Color("provinsi:N", title="Provinsi"),
        tooltip=[
            "kode_satker:N",
            "nama_kementerian:N",
            "provinsi:N",
            "skor_ikpa:Q",
            "deviasi_rpd_persen:Q",
        ],
    )
    st.altair_chart(chart, use_container_width=True)

with tab_prediksi:
    st.subheader("Prediksi Realisasi 95%")
    st.write("Gunakan model di folder `model` untuk memprediksi apakah realisasi tercapai 95%.")

    tipe_satker_options = [
        "Dekonsentrasi",
        "Kantor Daerah",
        "Kantor Pusat",
        "Tugas Pembantuan",
    ]

    jumlah_spm = st.number_input(
        "Jumlah SPM",
        min_value=0,
        value=0,
        step=1,
        format="%d",
        help="Masukkan jumlah SPM sebagai angka bulat.",
    )
    revisi_dipa = st.number_input(
        "Revisi DIPA",
        min_value=0,
        value=0,
        step=1,
        format="%d",
        help="Masukkan jumlah revisi DIPA sebagai angka bulat.",
    )
    deviasi_rpd = st.number_input(
        "Deviasi RPD (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        format="%.2f",
        help="Masukkan deviasi RPD dalam persen.",
    )
    skor_ikpa = st.number_input(
        "Skor IKPA",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        format="%.2f",
        help="Masukkan skor IKPA.",
    )
    tipe_satker = st.selectbox(
        "Tipe Satker",
        tipe_satker_options,
        help="Pilih tipe satker untuk prediksi.",
    )

    model = None
    if model_path.exists():
        try:
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
        except Exception as exc:
            st.error(f"Gagal memuat model: {exc}")
    else:
        st.error(f"File model tidak ditemukan: {model_path}")

    if st.button("Jalankan Prediksi"):
        if model is None:
            st.warning("Model belum dimuat. Pastikan paket Orange tersedia dan file model ada.")
        else:
            typed_features = {
                "Dekonsentrasi": [1.0, 0.0, 0.0, 0.0],
                "Kantor Daerah": [0.0, 1.0, 0.0, 0.0],
                "Kantor Pusat": [0.0, 0.0, 1.0, 0.0],
                "Tugas Pembantuan": [0.0, 0.0, 0.0, 1.0],
            }
            one_hot = typed_features.get(tipe_satker, [0.0, 0.0, 0.0, 0.0])

            values = [
                float(jumlah_spm),
                float(revisi_dipa),
                float(deviasi_rpd),
                float(skor_ikpa),
                *one_hot,
            ]

            try:
                from Orange.data import Table

                sample = Table.from_list(model.domain, [values])
                prediction_index = model(sample)[0]
                class_labels = list(model.domain.class_var.values)
                predicted_label = class_labels[int(prediction_index)]

                probas = model.predict_proba(sample)[0]
                probability_map = {
                    label: float(proba)
                    for label, proba in zip(class_labels, probas)
                }

                st.success("Prediksi selesai.")
                st.write("**Hasil prediksi:**")
                st.write(f"- Kelas: **{predicted_label}**")
                st.write(f"- Probabilitas Ya: {probability_map.get('Ya', 0.0):.3f}")
                st.write(f"- Probabilitas Tidak: {probability_map.get('Tidak', 0.0):.3f}")

                st.write("**Input yang digunakan:**")
                st.json(
                    {
                        "jumlah_spm": jumlah_spm,
                        "revisi_dipa": revisi_dipa,
                        "deviasi_rpd_persen": deviasi_rpd,
                        "skor_ikpa": skor_ikpa,
                        "tipe_satker": tipe_satker,
                    }
                )
            except Exception as exc:
                st.error(f"Terjadi kesalahan saat prediksi: {exc}")
