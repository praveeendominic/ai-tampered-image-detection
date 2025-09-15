
import io
import os
import glob
import time
import tempfile
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

import streamlit as st
import torch
import torchvision.transforms as T

try:
    import cv2  # opencv-python
except Exception:
    cv2 = None



st.set_page_config(page_title="AI-Edited Image Detector (IML-ViT)", layout="wide")
st.title("AI-Edited Image Detector")


with st.sidebar:
    st.header("Runtime")
    st.write("Torch:", torch.__version__)
    st.write("Device:", "`cpu`")

    st.markdown("---")
    st.subheader("Model Checkpoint")

    def auto_find_ckpt() -> str:
        candidates = []
        for root in ["weights", "pretrained-weights", ".", "ckpts", "checkpoints"]:
            candidates.extend(glob.glob(os.path.join(root, "*.pth")))
            candidates.extend(glob.glob(os.path.join(root, "*.pt")))
        return candidates[0] if candidates else ""

    default_ckpt = auto_find_ckpt()
    ckpt_path_input = st.text_input(
        "Path to checkpoint (.pth / .pt)",
        value=default_ckpt,
        help="If empty, you can upload a file below.",
    )
    uploaded_ckpt = st.file_uploader("…or upload a .pth/.pt", type=["pth", "pt"], key="ckpt_up")

    if uploaded_ckpt is not None:
        tmp_ckpt_dir = Path(tempfile.gettempdir()) / "imlvit_ckpts"
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tmp_path = tmp_ckpt_dir / f"uploaded_{int(time.time())}.pth"
        ckpt_tmp_path.write_bytes(uploaded_ckpt.read())
        ckpt_path = str(ckpt_tmp_path)
    else:
        ckpt_path = ckpt_path_input.strip()

    if not ckpt_path:
        st.info("No checkpoint selected yet. Enter a path or upload one.")

    st.markdown("---")
    st.subheader("Preprocessing")
    input_size = st.number_input("Model input size (square)", 256, 2048, 1024, 64)
    keep_aspect = st.checkbox("Letterbox-pad to keep aspect ratio", value=True)
    normalize_imagenet = st.checkbox("Normalize (ImageNet mean/std)", value=True)

    st.subheader("Edge Mask (for IML-ViT forward)")
    st.caption("We auto-compute with Canny + dilation.")
    canny_low = st.slider("Canny low", 0, 255, 50)
    canny_high = st.slider("Canny high", 0, 255, 150)
    dilate_iter = st.slider("Dilate iterations", 0, 5, 1)
    kernel_sz = st.slider("Dilation kernel (odd)", 1, 15, 3, step=2)

    st.markdown("---")
    st.subheader("Postprocessing")
    threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01)
    show_overlay = st.checkbox("Show overlay", value=True)
    overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)



# Image utils
def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def numpy_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def letterbox_pad(img: Image.Image, target: int):
    """Resize with unchanged aspect ratio & pad to square (target x target).
    Returns: padded_img, scale, pad (left, top) and original size.
    """
    ow, oh = img.size
    scale = min(target / ow, target / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new("RGB", (target, target), (0, 0, 0))
    left = (target - nw) // 2
    top = (target - nh) // 2
    new_img.paste(resized, (left, top))
    return new_img, scale, (left, top), (ow, oh)

def remove_letterbox(mask: np.ndarray, scale: float, pad: tuple[int, int], orig_size: tuple[int, int]) -> np.ndarray:
    """Undo letterbox on a single-channel mask in [H, W]."""
    left, top = pad
    ow, oh = orig_size
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    crop = mask[top : top + nh, left : left + nw]
    crop_img = Image.fromarray((crop * 255.0).astype(np.uint8))
    restored = crop_img.resize((ow, oh), Image.BILINEAR)
    return np.array(restored, dtype=np.float32) / 255.0

def build_edge_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Compute edge mask [H,W] in float32 from RGB image (uint8)."""
    if cv2 is None:
        # Fallback: Sobel magnitude if OpenCV isn't available
        gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        mag = np.sqrt(gx * gx + gy * gy)
        mag /= (mag.max() + 1e-6)
        return mag.astype(np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate_iter > 0 and kernel_sz > 1:
        kernel = np.ones((kernel_sz, kernel_sz), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return (edges.astype(np.float32) / 255.0)



# Model loading
@st.cache_resource(show_spinner=True)
def load_model(ckpt_path: str, input_size: int):
    from IMDLBenCo.registry import MODELS

    model_args = dict(
        input_size=input_size,
        patch_size=16,
        embed_dim=768,
        vit_pretrain_path=None,
        fpn_channels=256,
        fpn_scale_factors=(4.0, 2.0, 1.0, 0.5),
        mlp_embeding_dim=256,
        predict_head_norm="BN",
        edge_lambda=20,
    )

    model_cls = MODELS.get("IML_ViT")
    model = model_cls(**model_args)
    model.eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = None
        for key in ["model", "state_dict", "ema_state_dict", "model_state", "net"]:
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            state = ckpt if isinstance(ckpt, dict) else None

        def _strip(sd, prefix):
            return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

        if state is not None:
            for pfx in ["module.", "model.", "net.", "student.", "encoder."]:
                state = _strip(state, pfx)
            missing, unexpected = model.load_state_dict(state, strict=False)
            st.write("Loaded checkpoint:", os.path.basename(ckpt_path))
            if missing:
                st.info(f"Missing keys: {len(missing)}")
            if unexpected:
                st.info(f"Unexpected keys: {len(unexpected)}")
        else:
            st.warning("Checkpoint loaded but no compatible state dict found. Using random weights.")
    else:
        st.warning("No checkpoint file found; using random weights (not recommended).")

    return model

# Inference
def preprocess_for_model(pil_img: Image.Image, input_size: int, keep_aspect: bool, normalize: bool):
    if keep_aspect:
        padded, scale, pad, orig_size = letterbox_pad(pil_img, input_size)
        work_img = padded
    else:
        orig_size = pil_img.size
        scale = min(input_size / orig_size[0], input_size / orig_size[1])
        pad = (0, 0)
        work_img = pil_img.resize((input_size, input_size), Image.BICUBIC)

    np_img = pil_to_numpy_rgb(work_img)  # [H,W,3] uint8
    edge = build_edge_mask(np_img)       # [H,W] float32 in [0,1]

    tfms = [T.ToTensor()]  # -> [0,1], CxHxW
    if normalize:
        tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))
    tfm = T.Compose(tfms)
    tp_tensor = tfm(work_img)  # [3,H,W]

    edge_tensor = torch.from_numpy(edge).unsqueeze(0)  # [1,H,W]
    # Dummy mask for inference (some BenCo versions require it even if unused)
    mask_tensor = torch.zeros_like(edge_tensor)        # [1,H,W]
    return tp_tensor, edge_tensor, mask_tensor, (scale, pad, orig_size, np_img)


def _extract_pred_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        # common keys
        for k in ["pred", "logits", "out", "output", "pred_mask", "mask"]:
            v = out.get(k)
            if isinstance(v, torch.Tensor):
                return v
        # fallback: first tensor value
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    raise RuntimeError(f"Unexpected model output type/structure: {type(out)}")


def run_model(model, tp_tensor: torch.Tensor, edge_tensor: torch.Tensor, mask_tensor: torch.Tensor):
    """Robustly call model across different BenCo IML_ViT signatures."""
    b_img = tp_tensor.unsqueeze(0)                        # [1,3,H,W]
    b_edge = edge_tensor.unsqueeze(0)                     # [1,1,H,W]
    b_mask = mask_tensor.unsqueeze(0)                     # [1,1,H,W]

    last_err = None
    with torch.no_grad():
        # 1) named args 
        try:
            out = model(image=b_img, mask=b_mask, edge_mask=b_edge)
            pred = _extract_pred_tensor(out)
            return torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()
        except TypeError as e:
            last_err = e

        # 2) Alternate BenCo-style names
        for kwargs in [
            dict(tp_img=b_img, edge_mask=b_edge),
            dict(tp_img=b_img, mask=b_mask, edge_mask=b_edge),
        ]:
            try:
                out = model(**kwargs)
                pred = _extract_pred_tensor(out)
                return torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()
            except TypeError as e:
                last_err = e

        # 3) Positional: (image, mask, edge_mask)
        try:
            out = model(b_img, b_mask, b_edge)
            pred = _extract_pred_tensor(out)
            return torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()
        except TypeError as e:
            last_err = e

        # 4) Last resort: image only
        try:
            out = model(b_img)
            pred = _extract_pred_tensor(out)
            return torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()
        except Exception as e:
            last_err = e
            raise last_err


def overlay_mask(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.45):
    h, w = mask01.shape
    base = rgb.astype(np.float32).copy()
    color = np.zeros_like(base)
    color[..., 0] = 255.0  # red
    heat = (mask01[..., None] * color).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


uploaded_image = st.file_uploader("Upload an image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"], key="img_up")

col1, col2 = st.columns(2)

if uploaded_image is not None:
    raw = uploaded_image.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

    with col1:
        st.subheader("Input")
        st.image(pil_img, caption=f"{uploaded_image.name} ({pil_img.size[0]}×{pil_img.size[1]})")

    if 'ckpt_path' in locals() and (ckpt_path or uploaded_ckpt is not None):
        with st.spinner("Loading model…"):
            model = load_model(ckpt_path, input_size)

        with st.spinner("Preprocessing & running inference on CPU…"):
            tp_tensor, edge_tensor, mask_tensor, meta = preprocess_for_model(
                pil_img, input_size=input_size, keep_aspect=keep_aspect, normalize=normalize_imagenet
            )
            prob_map = run_model(model, tp_tensor, edge_tensor, mask_tensor)  # [H,W] in [0,1]

            scale, pad, orig_size, np_padded_rgb = meta
            if keep_aspect:
                restored = remove_letterbox(prob_map, scale, pad, orig_size)  # [H0,W0]
            else:
                restored = np.array(
                    Image.fromarray((prob_map * 255).astype(np.uint8)).resize(orig_size, Image.BILINEAR)
                ) / 255.0

            bin_mask = (restored >= threshold).astype(np.float32)
            tamper_ratio = float(bin_mask.mean())
            st.success(f"Tamper ratio (>= {threshold:.2f}): **{tamper_ratio*100:.2f}%**")

        with col2:
            st.subheader("Prediction")
            st.image((restored * 255).astype(np.uint8), caption="Raw probability map")

            if show_overlay:
                over = overlay_mask(pil_to_numpy_rgb(pil_img), restored, alpha=overlay_alpha)
                st.image(over, caption="Overlay")

        
        job_dir = Path(tempfile.mkdtemp(prefix="benco_ws_"))
        (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
        in_path = job_dir / "input.png"
        pil_img.save(in_path)

        out_prob_path = job_dir / "outputs" / "prob_map.png"
        out_bin_path = job_dir / "outputs" / "binary_mask.png"
        Image.fromarray((restored * 255).astype(np.uint8)).save(out_prob_path)
        Image.fromarray((bin_mask * 255).astype(np.uint8)).save(out_bin_path)
        if show_overlay:
            out_overlay_path = job_dir / "outputs" / "overlay.png"
            Image.fromarray(overlay_mask(pil_to_numpy_rgb(pil_img), restored, overlay_alpha)).save(out_overlay_path)

        st.info(f"📁 Job dir: `{job_dir}`")
        st.caption("Intermediate files and outputs were saved here for this run.")
    else:
        st.warning("Select or upload a checkpoint in the sidebar to run inference.")
else:
    st.info("Upload an image to begin.")
