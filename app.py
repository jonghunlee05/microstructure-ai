
import os, io, json, math, cv2
import numpy as np
import streamlit as st
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="UHCS Microstructure Classifier", layout="centered")

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "hybrid_svm_pipeline.joblib")
META_PATH  = os.path.join(ART_DIR, "hybrid_svm_meta.json")

# ---------- Utilities ----------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    classes = list(pipe.named_steps["svc"].classes_)
    return pipe, meta, classes

def load_image_to_gray(bytes_or_path):
    if isinstance(bytes_or_path, (str, os.PathLike)):
        img = cv2.imread(str(bytes_or_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(bytes_or_path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        pil = Image.open(io.BytesIO(bytes_or_path)).convert("L")
        return np.array(pil)

def ensure_u8(img):
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---- Classical features (GLCM 72 + LBP 28 = 100 dims) ----
# GLCM: distances = [1,2,4]; angles = [0,45,90,135]; props = 6 -> 72 dims
GLCM_DIST = [1,2,4]
GLCM_ANGS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]

def glcm_feats(img_u8):
    # Downsample intensity to 64 levels to stabilize textures
    img_q = (img_u8 // 4).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=GLCM_DIST, angles=GLCM_ANGS, levels=64, symmetric=True, normed=True)
    feats = []
    for p in GLCM_PROPS:
        v = graycoprops(glcm, p).ravel()  # shape: len(dist)*len(ang)
        feats.extend(v.tolist())
    return np.array(feats, dtype=np.float32)  # 6 * 3 * 4 = 72

# LBP: concatenate histograms for P=8,R=1 (10 bins) and P=16,R=2 (18 bins) with 'uniform' -> 28 dims
def lbp_feats(img_u8):
    arr = []
    for (P,R) in [(8,1),(16,2)]:
        lbp = local_binary_pattern(img_u8, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins), density=True)
        arr.append(hist.astype(np.float32))
    return np.concatenate(arr, axis=0)  # 10 + 18 = 28

def featurize_classical(img_u8):
    return np.concatenate([glcm_feats(img_u8), lbp_feats(img_u8)], axis=0)  # (100,)

# ---- Deep features (ResNet18 penultimate, 512-D) ----
@st.cache_resource
def build_backbone():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # output 512-D
    resnet.eval().to(device)
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return resnet, device, tfm

def to_rgb_for_backbone(img_u8):
    if img_u8.ndim == 2:
        rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
    return rgb

def deep_embed_single(img_u8, backbone_pack):
    resnet, device, tfm = backbone_pack
    rgb = to_rgb_for_backbone(img_u8)
    with torch.no_grad():
        x = tfm(rgb).unsqueeze(0).to(device)
        z = resnet(x).cpu().numpy().ravel().astype(np.float32)  # (512,)
    return z

# ---------- UI ----------
st.title("UHCS Microstructure Classifier (Hybrid: GLCM+LBP + ResNet18 + SVM)")

# Load model
try:
    pipe, meta, classes = load_artifacts()
except Exception as e:
    st.error(f"Could not load artifacts from {ART_DIR}. Make sure you ran the 'finalize model' cell.\n{e}")
    st.stop()

st.caption(f"Model: {meta.get('method','?')} • PCA: {meta.get('pca_n_components')} • SVM C={meta.get('svc_C')} γ={meta.get('svc_gamma')} • n={meta.get('n_samples')}")

# Optional: user-provided µm/px (not strictly needed here, kept for future)
um_per_px = st.number_input("Assumed scale (µm/px) for classical features", value=1.0, min_value=0.01, step=0.01, help="Used only if you later adjust classical features to be scale-aware.")

uploaded = st.file_uploader("Upload a micrograph (PNG/JPG/TIF). You can upload multiple.", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True)

if uploaded:
    bb = build_backbone()
    rows = []
    for f in uploaded:
        try:
            raw = f.read()
            img = load_image_to_gray(raw)
            img = ensure_u8(img)

            # Build HYBRID feature vector: classical(100) + deep(512)
            Xc = featurize_classical(img)                      # (100,)
            Xd = deep_embed_single(img, bb)                    # (512,)
            x  = np.hstack([Xc, Xd]).reshape(1, -1)

            pred = pipe.predict(x)[0]
            prob = None
            if hasattr(pipe.named_steps["svc"], "predict_proba"):
                try:
                    prob = float(pipe.named_steps["svc"].predict_proba(x).max())
                except Exception:
                    prob = None

            rows.append((f.name, pred, prob))

            # Display
            col1, col2 = st.columns([3,2])
            with col1:
                st.image(img, caption=f.name, clamp=True, use_column_width=True)
            with col2:
                st.markdown(f"**Prediction:** `{pred}`")
                if prob is not None:
                    st.markdown(f"**Confidence:** {prob:.2%}")
        except Exception as e:
            st.warning(f"{f.name}: failed — {e}")

    if rows:
        st.subheader("Batch results")
        import pandas as pd
        df = pd.DataFrame(rows, columns=["filename","pred","confidence"])
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
else:
    st.info("Upload one or more SEM micrographs to get predictions.")
