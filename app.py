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
    page_title="PCD Analytics — Ekstraksi Ciri Objek",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS: MIDNIGHT BLACK (#090D16) & ELECTRIC BLUE (#2563EB)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Latar Belakang Utama & Sidebar */
    .stApp {
        background-color: #090D16;
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E2E8F0;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Hero Banner Presentasi */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #0B1324 65%, #111F3D 100%);
        border: 1px solid #1E293B;
        border-left: 4px solid #2563EB;
        border-radius: 12px;
        padding: 1.4rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(37, 99, 235, 0.16);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 0.35rem 0;
        line-height: 1.3;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin: 0;
    }

    /* Section Header dengan Ikon SVG */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #F1F5F9;
        padding-bottom: 0.5rem;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #1E293B;
    }

    /* Kartu Metrik (st.metric) */
    div[data-testid="stMetric"] {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-top: 3px solid #2563EB;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    /* Styling Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1.1rem !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #2563EB !important;
    }

    /* Tombol Utama & Download */
    .stDownloadButton button, .stButton button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        background-color: #1D4ED8 !important;
        border-color: #60A5FA !important;
    }

    /* Info Box */
    div[data-testid="stAlert"] {
        background-color: #0F172A;
        border: 1px solid #1E3A8A;
        color: #E2E8F0;
        border-radius: 10px;
    }

    /* Frame Gambar */
    div[data-testid="stImage"] {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.5rem;
    }
    div[data-testid="stImage"] figcaption {
        color: #CBD5E1 !important;
        font-size: 0.82rem !important;
        font-weight: 500;
        padding-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# FUNGSI BANTUAN VISUALISASI & KONVERSI
# ==============================================================================
def convert_np_to_png_bytes(img_array: np.ndarray) -> bytes:
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def style_dark_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor("#0F172A")
    ax.set_title(title, color="#F8FAFC", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#94A3B8", fontsize=9)
    ax.set_ylabel(ylabel, color="#94A3B8", fontsize=9)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.18, color="#38BDF8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E293B")


# ==============================================================================
# HERO HEADER (SVG MINIMALIS)
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            PENGOLAHAN CITRA DIGITAL &bull; ANALISIS EKSTRAKSI CIRI
        </div>
        <div class="hero-title">Implementasi Segmentasi Citra &amp; Ekstraksi Ciri Objek</div>
        <p class="hero-subtitle">
            Alur Komputasi: Akuisisi Citra &rarr; Preprocessing (Grayscale &amp; Histogram Equalization) &rarr; Segmentasi HSV-Otsu &rarr; Morfologi &rarr; Ekstraksi Ciri Warna &amp; Bentuk
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SIDEBAR: KONTROL PARAMETER PIPELINE
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem; padding-bottom:0.5rem; border-bottom:1px solid #1E293B;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line>
            <line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line>
            <line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line>
            <line x1="17" y1="16" x2="23" y2="16"></line>
        </svg>
        <span style="font-weight:700; font-size:1rem; color:#F8FAFC;">Kontrol Parameter</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Sumber Citra
st.sidebar.subheader("01. Akuisisi Citra")
input_mode = st.sidebar.radio(
    "Sumber Citra Uji:",
    ["Citra Default (Objek Telur)", "Unggah Citra Baru"],
    index=0,
)

img_rgb = None
DEFAULT_PATH = os.path.join("assets", "foto_telur.jpg")

if input_mode == "Citra Default (Objek Telur)":
    if os.path.exists(DEFAULT_PATH):
        img_bgr_raw = cv2.imread(DEFAULT_PATH)
        img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)
    else:
        st.sidebar.warning(
            f"File `{DEFAULT_PATH}` tidak ditemukan. Pastikan gambar tersimpan di folder `assets/`."
        )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Pilih file citra (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_image)

if img_rgb is None:
    st.info("Silakan sediakan `assets/foto_telur.jpg` atau unggah file gambar melalui panel kiri.")
    st.stop()

st.sidebar.divider()

# 2. Parameter Preprocessing
st.sidebar.subheader("02. Preprocessing")
median_ksize = st.sidebar.select_slider(
    "Kernel Median Blur (Reduksi Noise)",
    options=[3, 5, 7, 9, 11],
    value=7,
)

st.sidebar.divider()

# 3. Parameter Segmentasi
st.sidebar.subheader("03. Segmentasi (Kanal S - HSV)")
seg_mode = st.sidebar.radio(
    "Metode Thresholding:",
    ["Auto Otsu Thresholding", "Manual Threshold"],
    index=0,
)
manual_thresh = st.sidebar.slider(
    "Nilai Ambang Manual (0–255)",
    min_value=0,
    max_value=255,
    value=97,
    disabled=(seg_mode == "Auto Otsu Thresholding"),
)

st.sidebar.divider()

# 4. Parameter Morfologi
st.sidebar.subheader("04. Operasi Morfologi")
close_ksize = st.sidebar.slider(
    "Kernel Closing (Tutup Lubang Kilau)",
    min_value=3,
    max_value=35,
    value=25,
    step=2,
)
open_ksize = st.sidebar.slider(
    "Kernel Opening (Bersihkan Latar)",
    min_value=3,
    max_value=25,
    value=9,
    step=2,
)


# ==============================================================================
# EKSEKUSI PIPELINE PENGOLAHAN CITRA DIGITAL
# ==============================================================================
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Tahap 1: Preprocessing
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hist_eq = cv2.equalizeHist(img_gray)
img_blur_bgr = cv2.medianBlur(img_bgr, median_ksize)

# Tahap 2: Segmentasi via HSV Saturation + Thresholding
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

# Tahap 3: Operasi Morfologi
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    st.error("Objek tidak terdeteksi. Turunkan nilai threshold manual pada panel parameter.")
    st.stop()

egg_contour = max(contours, key=cv2.contourArea)
mask_final = np.zeros_like(mask_clean)
cv2.drawContours(mask_final, [egg_contour], -1, 255, thickness=cv2.FILLED)

# Tahap 4: Ekstraksi Ciri Warna (Mean R, Mean G, Mean B pada citra asli)
mean_r, mean_g, mean_b, _ = cv2.mean(img_rgb, mask=mask_final)

# Tahap 5: Ekstraksi Ciri Bentuk
area = cv2.contourArea(egg_contour)
perimeter = cv2.arcLength(egg_contour, True)
x, y, w, h = cv2.boundingRect(egg_contour)
aspect_ratio = float(w) / h if h > 0 else 0.0
circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

# Visualisasi Hasil Segmentasi & Overlay Kontur (Tanpa Titik Biru Centroid)
segmented_egg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_final)
img_overlay = img_rgb.copy()
cv2.drawContours(img_overlay, [egg_contour], -1, (0, 255, 128), 4)
cv2.rectangle(img_overlay, (x, y), (x + w, y + h), (56, 189, 248), 3)


# ==============================================================================
# NAVIGASI UTAMA BERBASIS TAB
# ==============================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "01. Preprocessing & Segmentasi",
        "02. Isolasi Objek & Kontur",
        "03. Ekstraksi Ciri & Laporan",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: PREPROCESSING & SEGMENTASI
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(
        """
        <div class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
            Matriks Transformasi Tahapan Pengolahan Citra
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        f"Segmentasi aktif menggunakan kanal Saturation (HSV) dengan nilai ambang Threshold = {thresh_used:.1f} ({seg_mode}). "
        f"Resolusi citra masukan: {img_rgb.shape[1]} x {img_rgb.shape[0]} piksel."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption="Tahap 1 — Citra Asli (RGB)", use_container_width=True)
        st.image(s_channel, caption="Tahap 4 — Ekstraksi Kanal Saturation (HSV)", use_container_width=True)
    with col2:
        st.image(img_gray, caption="Tahap 2 — Preprocessing: Grayscale", use_container_width=True)
        st.image(mask_raw, caption="Tahap 5 — Mask Biner Awal (Thresholding)", use_container_width=True)
    with col3:
        st.image(img_hist_eq, caption="Tahap 3 — Preprocessing: Histogram Equalization", use_container_width=True)
        st.image(mask_final, caption="Tahap 6 — Mask Final (Morfologi Closing & Opening)", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            Analisis Distribusi Intensitas Piksel (Histogram)
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 3.6))
    fig_hist.patch.set_facecolor("#090D16")

    ax_hist[0].hist(img_gray.ravel(), bins=256, range=(0, 256), color="#38BDF8", alpha=0.85)
    style_dark_axis(ax_hist[0], "Distribusi Intensitas Grayscale Asli", "Intensitas Piksel (0–255)", "Frekuensi Piksel")

    ax_hist[1].hist(img_hist_eq.ravel(), bins=256, range=(0, 256), color="#2563EB", alpha=0.9)
    style_dark_axis(ax_hist[1], "Distribusi Setelah Histogram Equalization", "Intensitas Piksel (0–255)", "Frekuensi Piksel")

    plt.tight_layout()
    st.pyplot(fig_hist)

# ------------------------------------------------------------------------------
# TAB 2: HASIL SEGMENTASI & OVERLAY KONTUR
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(
        """
        <div class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
            </svg>
            Validasi Area Objek &amp; Geometri Kontur
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.image(
            segmented_egg,
            caption="Citra Objek Terisolasi (Region of Interest untuk Ekstraksi Mean RGB)",
            use_container_width=True,
        )
        st.download_button(
            label="Unduh Mask Biner Final (.PNG)",
            data=convert_np_to_png_bytes(mask_final),
            file_name="mask_final_objek.png",
            mime="image/png",
            use_container_width=True,
        )

    with c_right:
        st.image(
            img_overlay,
            caption="Overlay Kontur Eksternal (Hijau) & Bounding Box (Biru Cyan)",
            use_container_width=True,
        )
        st.download_button(
            label="Unduh Visualisasi Kontur (.PNG)",
            data=convert_np_to_png_bytes(img_overlay),
            file_name="overlay_kontur_objek.png",
            mime="image/png",
            use_container_width=True,
        )

# ------------------------------------------------------------------------------
# TAB 3: DASHBOARD EKSTRAKSI CIRI WARNA & BENTUK
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(
        """
        <div class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="13.5" cy="6.5" r=".5"></circle><circle cx="17.5" cy="10.5" r=".5"></circle>
                <circle cx="8.5" cy="7.5" r=".5"></circle><circle cx="6.5" cy="12.5" r=".5"></circle>
                <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"></path>
            </svg>
            Ekstraksi Ciri Warna (Kanal RGB Citra Asli pada Area Objek)
        </div>
        """,
        unsafe_allow_html=True,
    )

    w1, w2, w3, w4 = st.columns([1, 1, 1, 1.3])
    w1.metric("Mean R (Red)", f"{mean_r:.2f}")
    w2.metric("Mean G (Green)", f"{mean_g:.2f}")
    w3.metric("Mean B (Blue)", f"{mean_b:.2f}")

    hex_color = f"#{int(mean_r):02x}{int(mean_g):02x}{int(mean_b):02x}"
    with w4:
        st.markdown(
            f"""
            <div style="
                background-color: #0F172A;
                border: 1px solid #1E293B;
                border-top: 3px solid {hex_color};
                border-radius: 10px;
                padding: 0.72rem 1rem;
                display: flex;
                align-items: center;
                gap: 12px;
                height: 100%;">
                <div style="
                    width: 42px;
                    height: 42px;
                    border-radius: 8px;
                    background-color: {hex_color};
                    border: 1px solid rgba(255,255,255,0.25);
                    flex-shrink: 0;">
                </div>
                <div>
                    <div style="font-size:0.78rem; color:#94A3B8; font-weight:500;">Representasi Warna Objek</div>
                    <div style="font-size:1.05rem; color:#F8FAFC; font-weight:700; letter-spacing:0.04em;">{hex_color.upper()}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"></polygon>
                <line x1="12" y1="22" x2="12" y2="15.5"></line>
                <polyline points="22 8.5 12 15.5 2 8.5"></polyline>
            </svg>
            Ekstraksi Ciri Bentuk (Parameter Geometri Kontur)
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Area", f"{area:,.2f} px²")
    b2.metric("Perimeter", f"{perimeter:,.2f} px")
    b3.metric("Aspect Ratio", f"{aspect_ratio:.4f}", help=f"Dimensi Bounding Box: {w} x {h} piksel")
    b4.metric("Circularity", f"{circularity:.4f}", help="Indeks kebulatan (1.0000 = lingkaran sempurna)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
            </svg>
            Ringkasan Kuantitatif Hasil Ekstraksi Ciri
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_hasil = pd.DataFrame(
        [
            {"Kategori": "Ciri Warna", "Parameter": "Mean R (Red)", "Nilai": round(mean_r, 2), "Satuan": "Intensitas (0–255)"},
            {"Kategori": "Ciri Warna", "Parameter": "Mean G (Green)", "Nilai": round(mean_g, 2), "Satuan": "Intensitas (0–255)"},
            {"Kategori": "Ciri Warna", "Parameter": "Mean B (Blue)", "Nilai": round(mean_b, 2), "Satuan": "Intensitas (0–255)"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Area", "Nilai": round(area, 2), "Satuan": "piksel²"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Perimeter", "Nilai": round(perimeter, 2), "Satuan": "piksel"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Aspect Ratio", "Nilai": round(aspect_ratio, 4), "Satuan": "Rasio (w / h)"},
            {"Kategori": "Ciri Bentuk", "Parameter": "Circularity", "Nilai": round(circularity, 4), "Satuan": "Indeks (0–1)"},
        ]
    )

    st.dataframe(df_hasil, use_container_width=True, hide_index=True)

    csv_data = df_hasil.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Ekspor Hasil Ekstraksi Ciri (.CSV)",
        data=csv_data,
        file_name="hasil_ekstraksi_ciri_objek.csv",
        mime="text/csv",
    )