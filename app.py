import os
import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="PCD - Ekstraksi Ciri Objek Telur",
    page_icon="🥚",
    layout="wide",
)

st.title("🥚 Implementasi Pengolahan Citra Digital & Ekstraksi Ciri Objek")
st.title("Teguh Maulana Firmansyah (241011401348)")
st.markdown(
    """
    Aplikasi berbasis **Streamlit** ini mendemonstrasikan pipeline pengolahan citra digital untuk ekstraksi ciri warna dan bentuk objek telur.
    Menyesuaikan parameter preprocessing, segmentasi, dan morfologi, 
    serta mengekstraksi ciri warna (Mean R, G, B) dan ciri bentuk (Area, Perimeter, Aspect Ratio, Circularity).
    """
)
st.caption(
    "Pipeline: Akuisisi Citra → Preprocessing → Segmentasi HSV-Threshold → Morfologi → Ekstraksi Ciri Warna & Bentuk"
)
st.divider()


# ==============================================================================
# FUNGSI BANTUAN KONVERSI GAMBAR UNTUK DOWNLOAD
# ==============================================================================
def convert_np_to_png_bytes(img_array: np.ndarray) -> bytes:
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ==============================================================================
# SIDEBAR: INPUT CITRA & KONTROL PARAMETER INTERAKTIF
# ==============================================================================
st.sidebar.header("⚙️ Pengaturan Pipeline")

# 1. Sumber Citra
st.sidebar.subheader("1. Input Citra")
input_mode = st.sidebar.radio(
    "Pilih sumber foto:",
    ["Foto Default (Telur)", "Upload Foto Baru"],
    index=0,
)

img_rgb = None
DEFAULT_PATH = os.path.join("assets/telur.png")

if input_mode == "Foto Default (Telur)":
    if os.path.exists(DEFAULT_PATH):
        img_bgr_raw = cv2.imread(DEFAULT_PATH)
        img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)
    else:
        st.sidebar.warning(
            f"File `{DEFAULT_PATH}` belum ditemukan. Silakan simpan foto di folder `assets/` atau pilih mode Upload."
        )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Unggah gambar objek (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_image)

if img_rgb is None:
    st.info("👈 Silakan siapkan `assets/foto_telur.jpg` atau unggah foto melalui menu di sidebar.")
    st.stop()

# 2. Parameter Preprocessing
st.sidebar.subheader("2. Preprocessing")
median_ksize = st.sidebar.select_slider(
    "Kernel Median Blur (Reduksi Tekstur/Noise)",
    options=[3, 5, 7, 9, 11],
    value=7,
)

# 3. Parameter Segmentasi
st.sidebar.subheader("3. Segmentasi (Kanal Saturation HSV)")
seg_mode = st.sidebar.radio(
    "Metode Thresholding:",
    ["Auto Otsu Thresholding", "Manual Slider"],
    index=0,
)
manual_thresh = st.sidebar.slider(
    "Nilai Ambang Manual (Kanal S)",
    min_value=0,
    max_value=255,
    value=97,
    disabled=(seg_mode == "Auto Otsu Thresholding"),
)

# 4. Parameter Morfologi
st.sidebar.subheader("4. Operasi Morfologi")
close_ksize = st.sidebar.slider(
    "Kernel Closing (Menutup Lubang Kilau)",
    min_value=3,
    max_value=35,
    value=25,
    step=2,
)
open_ksize = st.sidebar.slider(
    "Kernel Opening (Hapus Noise Latar)",
    min_value=3,
    max_value=25,
    value=9,
    step=2,
)


# ==============================================================================
# EKSEKUSI PIPELINE PENGOLAHAN CITRA DIGITAL
# ==============================================================================
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Tahap 1: Preprocessing (Grayscale, Histogram Equalization, & Median Blur)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hist_eq = cv2.equalizeHist(img_gray)
img_blur_bgr = cv2.medianBlur(img_bgr, median_ksize)

# Tahap 2: Segmentasi pada Kanal Saturation (HSV)
img_hsv = cv2.cvtColor(img_blur_bgr, cv2.COLOR_BGR2HSV)
s_channel = img_hsv[:, :, 1]

if seg_mode == "Auto Otsu Thresholding":
    thresh_used, mask_raw = cv2.threshold(
        s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
else:
    thresh_used, mask_raw = cv2.threshold(
        s_channel, manual_thresh, 255, cv2.THRESH_BINARY
    )

# Tahap 3: Operasi Morfologi (Closing -> Opening -> Ekstraksi Kontur Terbesar)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    st.error(
        "Tidak ada kontur objek yang terdeteksi! Coba turunkan nilai threshold manual pada sidebar."
    )
    st.stop()

# Pilih kontur dengan area terbesar sebagai objek utama
egg_contour = max(contours, key=cv2.contourArea)
mask_final = np.zeros_like(mask_clean)
cv2.drawContours(mask_final, [egg_contour], -1, 255, thickness=cv2.FILLED)

# Tahap 4: Ekstraksi Ciri Warna (Mean R, Mean G, Mean B pada citra asli)
mean_r, mean_g, mean_b, _ = cv2.mean(img_rgb, mask=mask_final)

# Tahap 5: Ekstraksi Ciri Bentuk (Area, Perimeter, Aspect Ratio, Circularity)
area = cv2.contourArea(egg_contour)
perimeter = cv2.arcLength(egg_contour, True)
x, y, w, h = cv2.boundingRect(egg_contour)
aspect_ratio = float(w) / h if h > 0 else 0.0
circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

# Titik pusat (Centroid) objek menggunakan Image Moments
M = cv2.moments(egg_contour)
cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

# Citra hasil isolasi objek & visualisasi overlay kontur
segmented_egg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_final)
img_overlay = img_rgb.copy()
cv2.drawContours(img_overlay, [egg_contour], -1, (0, 255, 0), 4)
cv2.rectangle(img_overlay, (x, y), (x + w, y + h), (255, 50, 50), 3)


# ==============================================================================
# TAMPILAN UTAMA BERBASIS TAB
# ==============================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Preprocessing & Segmentasi",
        "2️⃣ Hasil Segmentasi & Kontur",
        "3️⃣ Dashboard Ekstraksi Ciri",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: PREPROCESSING & SEGMENTASI
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Tahapan Transformasi Citra")
    st.info(
        f"**Info Segmentasi:** Menggunakan kanal **Saturation (HSV)** dengan nilai ambang "
        f"`Threshold = {thresh_used:.1f}` ({seg_mode}) agar terbebas dari pantulan cahaya latar dan bayangan."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption="1. Citra Asli (RGB)", use_container_width=True)
        st.image(
            s_channel,
            caption="4. Kanal Saturation (HSV)",
            use_container_width=True,
        )
    with col2:
        st.image(
            img_gray,
            caption="2. Preprocessing: Grayscale",
            use_container_width=True,
        )
        st.image(
            mask_raw,
            caption="5. Mask Biner Awal (Thresholding)",
            use_container_width=True,
        )
    with col3:
        st.image(
            img_hist_eq,
            caption="3. Preprocessing: Histogram Equalization",
            use_container_width=True,
        )
        st.image(
            mask_final,
            caption="6. Mask Final (Setelah Morfologi)",
            use_container_width=True,
        )

    st.divider()
    show_hist = st.checkbox("📊 Tampilkan Grafik Histogram Intensitas (Grayscale vs Equalization)", value=True)
    if show_hist:
        fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 3.5))
        ax_hist[0].hist(img_gray.ravel(), bins=256, range=(0, 256), color="#4B5563")
        ax_hist[0].set_title("Histogram Grayscale Asli")
        ax_hist[0].set_xlabel("Intensitas Piksel (0–255)")
        ax_hist[0].set_ylabel("Jumlah Piksel")

        ax_hist[1].hist(img_hist_eq.ravel(), bins=256, range=(0, 256), color="#2563EB")
        ax_hist[1].set_title("Histogram Setelah Equalization")
        ax_hist[1].set_xlabel("Intensitas Piksel (0–255)")
        ax_hist[1].set_ylabel("Jumlah Piksel")

        plt.tight_layout()
        st.pyplot(fig_hist)

# ------------------------------------------------------------------------------
# TAB 2: HASIL SEGMENTASI & OVERLAY KONTUR
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Isolasi Objek & Deteksi Geometri Kontur")
    c_left, c_right = st.columns(2)

    with c_left:
        st.image(
            segmented_egg,
            caption="Objek Terisolasi (Area Perhitungan Mean RGB)",
            use_container_width=True,
        )
        st.download_button(
            label="⬇️ Unduh Mask Biner Final (.png)",
            data=convert_np_to_png_bytes(mask_final),
            file_name="mask_final_telur.png",
            mime="image/png",
        )

    with c_right:
        st.image(
            img_overlay,
            caption=f"Overlay Kontur (Hijau), Bounding Box (Merah), & Centroid ({cx}, {cy})",
            use_container_width=True,
        )
        st.download_button(
            label="⬇️ Unduh Visualisasi Kontur (.png)",
            data=convert_np_to_png_bytes(img_overlay),
            file_name="overlay_kontur_telur.png",
            mime="image/png",
        )

# ------------------------------------------------------------------------------
# TAB 3: DASHBOARD EKSTRAKSI CIRI WARNA & BENTUK
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Hasil Ekstraksi Ciri Warna (RGB Asli pada Area Objek)")
    w1, w2, w3, w4 = st.columns([1, 1, 1, 1.2])
    w1.metric("Mean R (Red)", f"{mean_r:.2f}")
    w2.metric("Mean G (Green)", f"{mean_g:.2f}")
    w3.metric("Mean B (Blue)", f"{mean_b:.2f}")

    # Pratinjau warna rata-rata objek (Color Swatch)
    hex_color = f"#{int(mean_r):02x}{int(mean_g):02x}{int(mean_b):02x}"
    with w4:
        st.markdown(
            f"""
            <div style="
                background-color: {hex_color};
                border-radius: 10px;
                padding: 14px;
                text-align: center;
                color: white;
                font-weight: bold;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
                border: 1px solid #ccc;">
                Rata-rata Warna Objek<br>{hex_color.upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Hasil Ekstraksi Ciri Bentuk (Geometri Kontur)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Area", f"{area:,.2f} px²")
    b2.metric("Perimeter", f"{perimeter:,.2f} px")
    b3.metric("Aspect Ratio", f"{aspect_ratio:.4f}", help=f"Lebar = {w} px, Tinggi = {h} px")
    b4.metric("Circularity", f"{circularity:.4f}", help="1.0000 = Lingkaran Sempurna")

    st.divider()
    st.subheader("Tabel Ringkasan & Export Laporan")

    df_hasil = pd.DataFrame(
        [
            {"Kategori": "Ciri Warna", "Parameter": "Mean R (Red)", "Nilai": round(mean_r, 2), "Satuan": "Intensitas (0-255)"},
            {"Kategori": "Ciri Warna", "Parameter": "Mean G (Green)", "Nilai": round(mean_g, 2), "Satuan": "Intensitas (0-255)"},
            {"Kategori": "Ciri Warna", "Parameter": "Mean B (Blue)", "Nilai": round(mean_b, 2), "Satuan": "Intensitas (0-255)"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Area", "Nilai": round(area, 2), "Satuan": "piksel²"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Perimeter", "Nilai": round(perimeter, 2), "Satuan": "piksel"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Aspect Ratio", "Nilai": round(aspect_ratio, 4), "Satuan": "Rasio (w/h)"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Circularity", "Nilai": round(circularity, 4), "Satuan": "Indeks (0-1)"},
        ]
    )

    st.dataframe(df_hasil, use_container_width=True, hide_index=True)

    csv_data = df_hasil.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Unduh Tabel Hasil Ekstraksi Ciri (.csv)",
        data=csv_data,
        file_name="hasil_ekstraksi_ciri_telur.csv",
        mime="text/csv",
    )