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
    page_title="PCD Vision Lab — Ekstraksi Ciri Objek",
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
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }

    /* Styling Sidebar Radio sebagai Vertical Navbar */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #0B1120;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.35rem;
        transition: all 0.18s ease;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #3B82F6;
        background-color: #131E36;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid #2563EB;
        border-left: 4px solid #38BDF8;
    }

    /* Top Header Banner */
    .top-header {
        background: linear-gradient(135deg, #0F172A 0%, #0B1324 60%, #112042 100%);
        border: 1px solid #1E293B;
        border-left: 4px solid #2563EB;
        border-radius: 12px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(37, 99, 235, 0.18);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.45rem;
    }
    .header-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 0.25rem 0;
    }
    .header-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        margin: 0;
    }
    .status-pill {
        background-color: #090D16;
        border: 1px solid #1E3A8A;
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        font-size: 0.8rem;
        color: #CBD5E1;
    }

    /* Section Title dengan Ikon SVG */
    .section-title {
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F1F5F9;
        padding-bottom: 0.45rem;
        margin-top: 0.2rem;
        margin-bottom: 0.95rem;
        border-bottom: 1px solid #1E293B;
    }

    /* Kartu Analisis & Interpretasi */
    .insight-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .insight-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .insight-body {
        font-size: 0.9rem;
        color: #E2E8F0;
        line-height: 1.6;
        margin: 0;
    }

    /* Kartu Metrik (st.metric) */
    div[data-testid="stMetric"] {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-top: 3px solid #2563EB;
        border-radius: 10px;
        padding: 0.85rem 1.05rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.38rem !important;
        font-weight: 700 !important;
    }

    /* Tombol Download */
    .stDownloadButton button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton button:hover {
        background-color: #1D4ED8 !important;
        border-color: #60A5FA !important;
    }

    /* Frame Gambar */
    div[data-testid="stImage"] {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.45rem;
    }
    div[data-testid="stImage"] figcaption {
        color: #CBD5E1 !important;
        font-size: 0.8rem !important;
        font-weight: 500;
        padding-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# FUNGSI BANTUAN & AKUISISI CITRA LANGSUNG
# ==============================================================================
@st.cache_data
def load_default_image() -> np.ndarray:
    candidates = [
        os.path.join("assets/telur.png"),
        os.path.join("assets/telur.png"),
        "assets/telur.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            bgr = cv2.imread(path)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


def convert_np_to_png_bytes(img_array: np.ndarray) -> bytes:
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def style_dark_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor("#0F172A")
    ax.set_title(title, color="#F8FAFC", fontsize=10, fontweight="bold", pad=9)
    ax.set_xlabel(xlabel, color="#94A3B8", fontsize=8.5)
    ax.set_ylabel(ylabel, color="#94A3B8", fontsize=8.5)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.18, color="#38BDF8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E293B")


img_rgb = load_default_image()
if img_rgb is None:
    st.error("File citra `assets/foto_telur.jpg` tidak ditemukan. Pastikan file tersebut berada di dalam folder `assets/`.")
    st.stop()


# ==============================================================================
# SIDEBAR: NAVBAR MODUL & KONTROL PARAMETER PIPELINE
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px; padding:0.25rem 0 0.75rem 0; border-bottom:1px solid #1E293B; margin-bottom:0.75rem;">
        <div style="background:#2563EB; width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M3 7V5a2 2 0 0 1 2-2h2"></path>
                <path d="M17 3h2a2 2 0 0 1 2 2v2"></path>
                <path d="M21 17v2a2 2 0 0 1-2 2h-2"></path>
                <path d="M7 21H5a2 2 0 0 1-2-2v-2"></path>
            </svg>
        </div>
        <div>
            <div style="font-weight:700; font-size:0.98rem; color:#F8FAFC; line-height:1.1;">PCD Vision Lab</div>
            <div style="font-size:0.72rem; color:#60A5FA;">Ekstraksi Ciri Objek</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader("Navigasi Modul")
active_page = st.sidebar.radio(
    "Pilih halaman analisis:",
    [
        "01. Ringkasan & Akuisisi Citra",
        "02. Preprocessing & Histogram",
        "03. Segmentasi & Morfologi",
        "04. Ekstraksi Ciri & Laporan",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.subheader("Parameter Preprocessing")
median_ksize = st.sidebar.select_slider(
    "Kernel Median Blur (Reduksi Noise)",
    options=[1, 3, 5, 7, 9, 11],
    value=7,
    help="Nilai 1 = tanpa filter. Nilai lebih besar meredam tekstur alas & pori cangkang.",
)

eq_method = st.sidebar.selectbox(
    "Metode Histogram Equalization",
    ["Global Histogram Equalization", "CLAHE (Adaptive Equalization)"],
    index=0,
)

clahe_clip = st.sidebar.slider(
    "CLAHE Clip Limit",
    min_value=1.0,
    max_value=8.0,
    value=2.5,
    step=0.5,
    disabled=(eq_method == "Global Histogram Equalization"),
)

st.sidebar.subheader("Parameter Segmentasi (HSV)")
seg_mode = st.sidebar.selectbox(
    "Metode Thresholding (Kanal S)",
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

st.sidebar.subheader("Parameter Morfologi")
close_ksize = st.sidebar.slider(
    "Kernel Closing (Tutup Lubang Kilau)",
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
# EKSEKUSI PIPELINE PENGOLAHAN CITRA (TERHUBUNG DINAMIS KE SEMUA KONTROL)
# ==============================================================================
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# 1. Filtering terlebih dahulu agar perubahan kernel langsung memengaruhi histogram
if median_ksize > 1:
    img_filtered_bgr = cv2.medianBlur(img_bgr, median_ksize)
else:
    img_filtered_bgr = img_bgr.copy()

# 2. Konversi Grayscale & Histogram Equalization / CLAHE
img_gray = cv2.cvtColor(img_filtered_bgr, cv2.COLOR_BGR2GRAY)

if eq_method == "Global Histogram Equalization":
    img_hist_eq = cv2.equalizeHist(img_gray)
else:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    img_hist_eq = clahe.apply(img_gray)

# 3. Segmentasi via Kanal Saturation (HSV)
img_hsv = cv2.cvtColor(img_filtered_bgr, cv2.COLOR_BGR2HSV)
s_channel = img_hsv[:, :, 1]

if seg_mode == "Auto Otsu Thresholding":
    thresh_used, mask_raw = cv2.threshold(
        s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
else:
    thresh_used, mask_raw = cv2.threshold(
        s_channel, manual_thresh, 255, cv2.THRESH_BINARY
    )

# 4. Operasi Morfologi (Closing -> Opening -> Kontur Terbesar)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_close)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    st.error("Objek tidak terdeteksi pada nilai ambang ini. Turunkan nilai Manual Threshold di sidebar.")
    st.stop()

egg_contour = max(contours, key=cv2.contourArea)
mask_final = np.zeros_like(mask_clean)
cv2.drawContours(mask_final, [egg_contour], -1, 255, thickness=cv2.FILLED)

# Komparasi Kuantitatif Sebelum vs Sesudah Morfologi
raw_contours, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(raw_contours) > 0:
    raw_main_cnt = max(raw_contours, key=cv2.contourArea)
    raw_area = cv2.contourArea(raw_main_cnt)
    raw_perim = cv2.arcLength(raw_main_cnt, True)
    raw_circularity = (4 * np.pi * raw_area) / (raw_perim ** 2) if raw_perim > 0 else 0.0
else:
    raw_circularity = 0.0

holes_filled_px = int(np.sum((mask_final == 255) & (mask_raw == 0)))
noise_removed_px = int(np.sum((mask_raw == 255) & (mask_final == 0)))

# 5. Ekstraksi Ciri Warna (Mean R, Mean G, Mean B pada citra asli)
mean_r, mean_g, mean_b, _ = cv2.mean(img_rgb, mask=mask_final)

# 6. Ekstraksi Ciri Bentuk
area = cv2.contourArea(egg_contour)
perimeter = cv2.arcLength(egg_contour, True)
x, y, w, h = cv2.boundingRect(egg_contour)
aspect_ratio = float(w) / h if h > 0 else 0.0
circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
circ_delta = circularity - raw_circularity

# Visualisasi Hasil Segmentasi, Alpha Blending, & Overlay Kontur
segmented_egg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_final)

# Alpha Blending Overlay: latar belakang diredupkan, area objek diberi tint biru halus
alpha_blend = (img_rgb * 0.35).astype(np.uint8)
tint_layer = img_rgb.copy()
tint_layer[mask_final == 255] = (
    0.72 * img_rgb[mask_final == 255] + 0.28 * np.array([37, 99, 235])
).astype(np.uint8)
alpha_blend[mask_final == 255] = tint_layer[mask_final == 255]
cv2.drawContours(alpha_blend, [egg_contour], -1, (56, 189, 248), 3)

img_overlay = img_rgb.copy()
cv2.drawContours(img_overlay, [egg_contour], -1, (0, 255, 128), 4)
cv2.rectangle(img_overlay, (x, y), (x + w, y + h), (56, 189, 248), 3)


# ==============================================================================
# TOP HEADER BANNER
# ==============================================================================
st.markdown(
    f"""
    <div class="top-header">
        <div>
            <div class="header-tag">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2.2">
                    <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                TUGAS INDIVIDU PENGOLAHAN CITRA DIGITAL
            </div>
            <div class="header-title">Implementasi Pengolahan Citra Digital &amp; Ekstraksi Ciri Objek</div>
            <p class="header-desc">Objek Uji: Telur Ayam Negeri &bull; Dimensi Citra: {img_rgb.shape[1]} &times; {img_rgb.shape[0]} px &bull; Ruang Warna: RGB &amp; HSV</p>
        </div>
        <div class="status-pill">
            <div style="color:#60A5FA; font-weight:600;">Status Pipeline Aktif</div>
            <div>Threshold S: <b>{thresh_used:.1f}</b> | Closing: <b>{close_ksize}px</b> | Opening: <b>{open_ksize}px</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# HALAMAN 01: RINGKASAN & AKUISISI CITRA
# ==============================================================================
if active_page == "01. Ringkasan & Akuisisi Citra":
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            Akuisisi Citra Mandiri &amp; Ringkasan Ekstraksi Ciri
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Resolusi Citra", f"{img_rgb.shape[1]} x {img_rgb.shape[0]} px")
    k2.metric("Mean RGB Objek", f"{mean_r:.0f}, {mean_g:.0f}, {mean_b:.0f}")
    k3.metric("Luas Objek (Area)", f"{area:,.0f} px²")
    k4.metric("Circularity", f"{circularity:.4f}", delta=f"{circ_delta:+.4f} pasca morfologi")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(img_rgb, caption="1. Citra Asli (Akuisisi Mandiri)", use_container_width=True)
    with col_b:
        st.image(alpha_blend, caption="2. Validasi Presisi Mask (Alpha Blending)", use_container_width=True)
    with col_c:
        st.image(img_overlay, caption="3. Deteksi Kontur & Bounding Box", use_container_width=True)


# ==============================================================================
# HALAMAN 02: PREPROCESSING & HISTOGRAM
# ==============================================================================
elif active_page == "02. Preprocessing & Histogram":
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            Tahap Preprocessing: Grayscale, Histogram Equalization &amp; Analisis Distribusi
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(
            img_gray,
            caption=f"Citra Grayscale (Median Blur {median_ksize}x{median_ksize})",
            use_container_width=True,
        )
    with c2:
        st.image(
            img_hist_eq,
            caption=f"Hasil {eq_method}",
            use_container_width=True,
        )
    with c3:
        st.image(
            s_channel,
            caption="Kanal Saturation (HSV) untuk Segmentasi",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            Grafik Histogram Dinamis (Responsif Terhadap Kontrol Parameter)
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_hist, ax_hist = plt.subplots(1, 3, figsize=(15, 3.8))
    fig_hist.patch.set_facecolor("#090D16")

    ax_hist[0].hist(img_gray.ravel(), bins=256, range=(0, 256), color="#38BDF8", alpha=0.85)
    style_dark_axis(
        ax_hist[0],
        f"1. Histogram Grayscale (Blur={median_ksize})",
        "Intensitas (0–255)",
        "Jumlah Piksel",
    )

    ax_hist[1].hist(img_hist_eq.ravel(), bins=256, range=(0, 256), color="#2563EB", alpha=0.9)
    style_dark_axis(
        ax_hist[1],
        f"2. Histogram {eq_method.split()[0]}",
        "Intensitas (0–255)",
        "Jumlah Piksel",
    )

    ax_hist[2].hist(s_channel.ravel(), bins=256, range=(0, 256), color="#10B981", alpha=0.8)
    ax_hist[2].axvline(
        thresh_used,
        color="#F59E0B",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {thresh_used:.1f}",
    )
    ax_hist[2].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
    style_dark_axis(
        ax_hist[2],
        "3. Histogram Kanal Saturation & Threshold",
        "Nilai Saturasi (0–255)",
        "Jumlah Piksel",
    )

    plt.tight_layout()
    st.pyplot(fig_hist)


# ==============================================================================
# HALAMAN 03: SEGMENTASI & OPERASI MORFOLOGI
# ==============================================================================
elif active_page == "03. Segmentasi & Morfologi":
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
            Komparasi Kuantitatif &amp; Transformasi Morfologi
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Lubang Kilau Tertutup (Closing)",
        f"{holes_filled_px:,} px",
        help="Jumlah piksel di dalam objek akibat pantulan flash yang berhasil ditambal operasi Closing.",
    )
    d2.metric(
        "Noise Latar Dihapus (Opening)",
        f"{noise_removed_px:,} px",
        help="Jumlah piksel noise di luar objek utama yang dibersihkan operasi Opening.",
    )
    d3.metric(
        "Circularity (Sebelum -> Sesudah)",
        f"{circularity:.4f}",
        delta=f"{circ_delta:+.4f} (Awal: {raw_circularity:.4f})",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.image(
            mask_raw,
            caption=f"1. Mask Awal (Threshold S = {thresh_used:.1f})",
            use_container_width=True,
        )
    with m2:
        st.image(
            mask_closed,
            caption=f"2. Setelah Closing ({close_ksize}x{close_ksize}) — Lubang Tertutup",
            use_container_width=True,
        )
    with m3:
        st.image(
            mask_final,
            caption=f"3. Mask Final Setelah Opening ({open_ksize}x{open_ksize})",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    with o1:
        st.image(segmented_egg, caption="Objek Terisolasi (Area Perhitungan RGB)", use_container_width=True)
        st.download_button(
            label="Unduh Mask Biner Final (.PNG)",
            data=convert_np_to_png_bytes(mask_final),
            file_name="mask_final_telur.png",
            mime="image/png",
            use_container_width=True,
        )
    with o2:
        st.image(alpha_blend, caption="Validasi Tepi Segmentasi (Alpha Blending)", use_container_width=True)
        st.download_button(
            label="Unduh Alpha Blending (.PNG)",
            data=convert_np_to_png_bytes(alpha_blend),
            file_name="validasi_tepi_telur.png",
            mime="image/png",
            use_container_width=True,
        )
    with o3:
        st.image(img_overlay, caption="Overlay Kontur (Hijau) & Bounding Box (Biru)", use_container_width=True)
        st.download_button(
            label="Unduh Visualisasi Kontur (.PNG)",
            data=convert_np_to_png_bytes(img_overlay),
            file_name="overlay_kontur_telur.png",
            mime="image/png",
            use_container_width=True,
        )


# ==============================================================================
# HALAMAN 04: EKSTRAKSI CIRI & EKSPOR LAPORAN
# ==============================================================================
elif active_page == "04. Ekstraksi Ciri & Laporan":
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <circle cx="13.5" cy="6.5" r=".5"></circle><circle cx="17.5" cy="10.5" r=".5"></circle>
                <circle cx="8.5" cy="7.5" r=".5"></circle><circle cx="6.5" cy="12.5" r=".5"></circle>
                <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"></path>
            </svg>
            Ekstraksi Ciri Warna (Mean R, Mean G, Mean B pada Area Objek)
        </div>
        """,
        unsafe_allow_html=True,
    )

    w1, w2, w3, w4 = st.columns([1, 1, 1, 1.35])
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
                    <div style="font-size:0.78rem; color:#94A3B8; font-weight:500;">Warna Rata-rata Objek</div>
                    <div style="font-size:1.05rem; color:#F8FAFC; font-weight:700; letter-spacing:0.04em;">{hex_color.upper()}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Histogram Distribusi Kanal RGB Khusus pada Area Mask Objek
    st.markdown("<br>", unsafe_allow_html=True)
    fig_rgb, ax_rgb = plt.subplots(figsize=(12, 3.4))
    fig_rgb.patch.set_facecolor("#090D16")

    obj_pixels_r = img_rgb[:, :, 0][mask_final == 255]
    obj_pixels_g = img_rgb[:, :, 1][mask_final == 255]
    obj_pixels_b = img_rgb[:, :, 2][mask_final == 255]

    bins = np.arange(257)
    hist_r, _ = np.histogram(obj_pixels_r, bins=bins)
    hist_g, _ = np.histogram(obj_pixels_g, bins=bins)
    hist_b, _ = np.histogram(obj_pixels_b, bins=bins)

    ax_rgb.plot(hist_r, color="#EF4444", linewidth=2, label=f"Red (Mean = {mean_r:.2f})")
    ax_rgb.fill_between(range(256), hist_r, color="#EF4444", alpha=0.14)
    ax_rgb.axvline(mean_r, color="#EF4444", linestyle="--", linewidth=1.3)

    ax_rgb.plot(hist_g, color="#22C55E", linewidth=2, label=f"Green (Mean = {mean_g:.2f})")
    ax_rgb.fill_between(range(256), hist_g, color="#22C55E", alpha=0.14)
    ax_rgb.axvline(mean_g, color="#22C55E", linestyle="--", linewidth=1.3)

    ax_rgb.plot(hist_b, color="#3B82F6", linewidth=2, label=f"Blue (Mean = {mean_b:.2f})")
    ax_rgb.fill_between(range(256), hist_b, color="#3B82F6", alpha=0.14)
    ax_rgb.axvline(mean_b, color="#3B82F6", linestyle="--", linewidth=1.3)

    ax_rgb.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)
    style_dark_axis(
        ax_rgb,
        "Distribusi Intensitas Kanal R, G, B Khusus pada Area Objek Tersegmentasi",
        "Intensitas Warna (0–255)",
        "Jumlah Piksel Objek",
    )
    plt.tight_layout()
    st.pyplot(fig_rgb)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"></polygon>
                <line x1="12" y1="22" x2="12" y2="15.5"></line>
                <polyline points="22 8.5 12 15.5 2 8.5"></polyline>
            </svg>
            Ekstraksi Ciri Bentuk &amp; Formulasi Matematis
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Area", f"{area:,.2f} px²")
    b2.metric("Perimeter", f"{perimeter:,.2f} px")
    b3.metric("Aspect Ratio", f"{aspect_ratio:.4f}", help=f"Lebar = {w} px, Tinggi = {h} px")
    b4.metric("Circularity", f"{circularity:.4f}", help="1.0000 = Lingkaran Sempurna")

    # Formulasi Matematis & Interpretasi Otomatis
    f_col1, f_col2 = st.columns([1.15, 1])
    with f_col1:
        st.markdown("**Formulasi Perhitungan Ciri:**")
        st.latex(
            r"\text{Mean } C = \frac{1}{N} \sum_{(x,y) \in \text{Objek}} I_C(x,y), \quad C \in \{R, G, B\}"
        )
        st.latex(
            r"\text{Aspect Ratio} = \frac{W_{\text{bbox}}}{H_{\text{bbox}}} = \frac{"
            + f"{w}"
            + r"}{"
            + f"{h}"
            + r"} = "
            + f"{aspect_ratio:.4f}"
        )
        st.latex(
            r"\text{Circularity} = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2} = \frac{4\pi \times "
            + f"{area:.2f}"
            + r"}{("
            + f"{perimeter:.2f}"
            + r")^2} = "
            + f"{circularity:.4f}"
        )

    with f_col2:
        orientasi = "lonjong vertikal (tinggi > lebar)" if aspect_ratio < 0.95 else (
            "lonjong horizontal (lebar > tinggi)" if aspect_ratio > 1.05 else "simetris (lebar ≈ tinggi)"
        )
        bentuk_desc = "mendekati elips/oval mulus" if circularity >= 0.80 else "cenderung memanjang atau tepi tidak beraturan"
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Interpretasi Otomatis Karakteristik Objek</div>
                <p class="insight-body">
                    <b>1. Karakteristik Warna:</b> Dominasi kanal merah (<b>R = {mean_r:.2f}</b>) diikuti hijau (<b>G = {mean_g:.2f}</b>) dan biru rendah (<b>B = {mean_b:.2f}</b>) merepresentasikan warna cokelat-oranye hangat khas cangkang telur.<br><br>
                    <b>2. Karakteristik Geometri:</b> Nilai <b>Aspect Ratio = {aspect_ratio:.4f}</b> menunjukkan proporsi objek {orientasi}, sedangkan <b>Circularity = {circularity:.4f}</b> mengonfirmasi kontur objek {bentuk_desc}.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
            </svg>
            Tabel Hasil Ekstraksi Ciri untuk Laporan
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
        label="Ekspor Tabel Ekstraksi Ciri (.CSV)",
        data=csv_data,
        file_name="hasil_ekstraksi_ciri_telur.csv",
        mime="text/csv",
    )