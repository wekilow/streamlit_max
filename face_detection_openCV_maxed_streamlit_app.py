# face_detection_openCV_maxed_streamlit_app.py

from __future__ import annotations

import os
from pathlib import Path
import urllib.request
from io import BytesIO
import base64

import streamlit as st
import cv2
import numpy as np
from PIL import Image


# ============================================================
# Model: OpenCV DNN Caffe SSD face detector (ResNet-10 backbone)
# Files:
#   deploy.prototxt
#   res10_300x300_ssd_iter_140000_fp16.caffemodel
# ============================================================

PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"


# -----------------------------
# Download helpers
# -----------------------------
def download_if_missing(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    with st.spinner(f"Downloading {dst.name} ..."):
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {dst.name}.\n"
                f"URL: {url}\n"
                f"Destination: {dst}\n"
                f"Error: {e}\n\n"
                f"If you're behind a network filter, try:\n"
                f"1) connect via another network (hotspot)\n"
                f"2) download the files manually into {dst.parent}\n"
            ) from e

    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is missing/empty: {dst}")


@st.cache_resource()
def load_model() -> cv2.dnn.Net:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    proto_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    download_if_missing(PROTO_URL, proto_path)
    download_if_missing(MODEL_URL, model_path)

    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    return net


# -----------------------------
# UI download link helper
# -----------------------------
def get_image_download_link(img_pil: Image.Image, filename: str, text: str) -> str:
    buf = BytesIO()
    img_pil.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'


# -----------------------------
# Preprocessing (optional)
# -----------------------------
def apply_clahe_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    # CLAHE on luminance only (Y channel) → tends to help in shadows/backlight
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    y2 = clahe.apply(y)
    ycrcb2 = cv2.merge((y2, cr, cb))
    return cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)


# -----------------------------
# One-scale DNN detection
# Returns list of [x1,y1,x2,y2,conf] in that image's coords
# -----------------------------
def detect_one_scale(net: cv2.dnn.Net, img_bgr: np.ndarray, conf_threshold: float) -> list[list[float]]:
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 117, 123),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()

    boxes: list[list[float]] = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        boxes.append([x1, y1, x2, y2, conf])
    return boxes


# -----------------------------
# Multi-scale detection
# Upscaling helps small faces; mapping returns original coords
# -----------------------------
def detect_multiscale(
    net: cv2.dnn.Net,
    img_bgr: np.ndarray,
    conf_threshold: float,
    scales: list[float],
) -> list[list[float]]:
    h0, w0 = img_bgr.shape[:2]
    all_boxes: list[list[float]] = []

    for s in scales:
        if s <= 0:
            continue
        if s == 1.0:
            resized = img_bgr
        else:
            resized = cv2.resize(
                img_bgr,
                (int(w0 * s), int(h0 * s)),
                interpolation=cv2.INTER_LINEAR,
            )

        boxes_s = detect_one_scale(net, resized, conf_threshold=conf_threshold)
        hs, ws = resized.shape[:2]

        # Map to original coords (divide by scale)
        for x1, y1, x2, y2, conf in boxes_s:
            # Clip in resized first, then map back
            x1 = max(0, min(int(x1), ws - 1))
            y1 = max(0, min(int(y1), hs - 1))
            x2 = max(0, min(int(x2), ws - 1))
            y2 = max(0, min(int(y2), hs - 1))

            x1o = int(x1 / s)
            y1o = int(y1 / s)
            x2o = int(x2 / s)
            y2o = int(y2 / s)
            all_boxes.append([x1o, y1o, x2o, y2o, conf])

    return all_boxes


# -----------------------------
# Tiling detection (crowds/tiny faces)
# Runs multi-scale inside each tile
# -----------------------------
def detect_tiled(
    net: cv2.dnn.Net,
    img_bgr: np.ndarray,
    conf_threshold: float,
    scales: list[float],
    tile_size: int,
    overlap: float,
) -> list[list[float]]:
    h, w = img_bgr.shape[:2]
    overlap = float(np.clip(overlap, 0.0, 0.6))
    step = max(1, int(tile_size * (1.0 - overlap)))

    all_boxes: list[list[float]] = []

    y = 0
    while y < h:
        x = 0
        y2 = min(y + tile_size, h)

        while x < w:
            x2 = min(x + tile_size, w)
            tile = img_bgr[y:y2, x:x2]

            tile_boxes = detect_multiscale(net, tile, conf_threshold, scales)

            # Offset to global coords
            for bx1, by1, bx2, by2, conf in tile_boxes:
                all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y, conf])

            if x2 == w:
                break
            x += step

        if y2 == h:
            break
        y += step

    return all_boxes


# -----------------------------
# Filters to reduce junk when threshold lowered
# -----------------------------
def filter_boxes(
    boxes: list[list[float]],
    img_shape: tuple[int, int, int],
    min_size: int,
    max_aspect_ratio: float,
) -> list[list[float]]:
    h, w = img_shape[:2]
    out: list[list[float]] = []

    for x1, y1, x2, y2, conf in boxes:
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w - 1)))
        y2 = int(max(0, min(y2, h - 1)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        if min(bw, bh) < min_size:
            continue

        ar = max(bw / (bh + 1e-6), bh / (bw + 1e-6))  # always >= 1
        if ar > max_aspect_ratio:
            continue

        out.append([x1, y1, x2, y2, float(conf)])

    return out


# -----------------------------
# NMS using OpenCV
# -----------------------------
def nms_boxes(
    boxes: list[list[float]],
    iou_threshold: float,
    top_k: int,
) -> list[list[float]]:
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)

    rects = []
    scores = []
    for x1, y1, x2, y2, conf in boxes_sorted:
        rects.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(float(conf))

    idxs = cv2.dnn.NMSBoxes(
        bboxes=rects,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )

    if len(idxs) == 0:
        return []

    idxs = np.array(idxs).reshape(-1).tolist()
    kept = [boxes_sorted[i] for i in idxs]
    return kept[: int(top_k)]


# -----------------------------
# Draw
# -----------------------------
def draw_boxes(img_bgr: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    out = img_bgr.copy()
    h = out.shape[0]
    thickness = max(1, int(round(h / 200)))
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
        cv2.putText(
            out,
            f"{conf:.2f}",
            (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


# ============================================================
# Streamlit App
# ============================================================
st.set_page_config(page_title="OpenCV Face Detection (Maxed)", layout="wide")
st.title("OpenCV Face Detection — Maxed Out (Same Caffe SSD ResNet-10 Model)")

# Load model once
try:
    net = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Controls
c1, c2, c3 = st.columns(3)
with c1:
    max_mode = st.checkbox("MAX Recall Mode (multi-scale)", value=True)
    use_tiling = st.checkbox("Enable Tiling (best for crowds/tiny faces)", value=False)
    use_clahe = st.checkbox("Enable CLAHE (low/uneven light)", value=False)

with c2:
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25 if max_mode else 0.5, 0.01)
    nms_iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.40, 0.01)
    top_k = st.slider("Keep top-K after NMS", 10, 500, 200, 10)

with c3:
    min_size = st.slider("Min box size (px)", 5, 200, 20, 1)
    max_ar = st.slider("Max aspect ratio filter", 1.2, 6.0, 2.5, 0.1)
    tile_size = st.slider("Tile size (px)", 400, 2000, 900, 50)
    overlap = st.slider("Tile overlap", 0.0, 0.6, 0.25, 0.05)

default_scales = [1.0, 1.5, 2.0, 2.5] if max_mode else [1.0]
scale_str = st.text_input("Scales (comma-separated)", ",".join(str(s) for s in default_scales))

try:
    scales = [float(s.strip()) for s in scale_str.split(",") if s.strip()]
    scales = [s for s in scales if s > 0]
    if not scales:
        scales = default_scales
except Exception:
    scales = default_scales

if img_file is None:
    st.info("Upload an image to run detection.")
    st.stop()

raw = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not decode image. Try another file.")
    st.stop()

proc = apply_clahe_bgr(img) if use_clahe else img

# Detect
if use_tiling:
    boxes = detect_tiled(net, proc, conf_threshold, scales, tile_size=tile_size, overlap=overlap)
else:
    boxes = detect_multiscale(net, proc, conf_threshold, scales)

# Filter + NMS
boxes = filter_boxes(boxes, img.shape, min_size=min_size, max_aspect_ratio=max_ar)
boxes = nms_boxes(boxes, iou_threshold=nms_iou, top_k=top_k)

# Draw
out = draw_boxes(img, boxes)

# Show
l, r = st.columns(2)
l.image(img, channels="BGR", caption="Input")
r.image(out, channels="BGR", caption=f"Output (faces kept: {len(boxes)})")

# Download
out_pil = Image.fromarray(out[:, :, ::-1])
st.markdown(get_image_download_link(out_pil, "face_output.jpg", "Download Output Image"), unsafe_allow_html=True)

# Debug
with st.expander("Debug details"):
    st.write("Model name:", "OpenCV DNN Caffe SSD face detector (ResNet-10 backbone)")
    st.write("Scales:", scales)
    st.write("Mode:", "Tiling" if use_tiling else "Multi-scale only")
    st.write("conf_threshold:", conf_threshold, "NMS IoU:", nms_iou, "top_k:", top_k)
    st.write("min_size:", min_size, "max_aspect_ratio:", max_ar)
    st.write("CLAHE:", use_clahe)
    st.write("Tip:", "If small faces are still missed, increase scales or enable tiling (slower).")
