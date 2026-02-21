#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3-base: resize(h=480) -> pad(518x518) -> extract patch tokens
1) 抽样 token 建码本：GPU KMeans (FAISS)
2) 全量推理：token -> 最近中心 -> 离散标签图
3) 标签图：Hp×Wp -> 518×518 -> 去 padding -> new_w×480 -> 回原图尺寸 -> PNG(uint8/uint16)

依赖：
  pip install torch torchvision transformers pillow opencv-python tqdm numpy
  # GPU KMeans:
  conda install -c pytorch -c nvidia faiss-gpu
"""

import os
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, AutoModel
import cv2


# =========================
# 配置区（按需修改）
# =========================
@dataclass
class Config:
    image_dir: str = "datasets/Cambridge_OldHospital/train/rgb"
    out_dir: str = "semantics"

    model_name: str = "facebook/dinov3-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True

    # 预处理：按你要求
    resize_target_h: int = 480
    pad_final_size: int = 518

    # Codebook / 聚类（抽样建码本）
    K: int = 128
    sample_imgs: int = 300
    sample_tokens_per_img: int = 512

    # FAISS GPU KMeans 参数
    faiss_niter: int = 25
    faiss_nredo: int = 1
    faiss_seed: int = 0

    # 推理批处理
    infer_batch_size: int = 8
    num_workers: int = 8

    # 输出
    png_compress: int = 3  # 0~9
    save_uint16_if_needed: bool = True  # K>255 自动 uint16


CFG = Config()


# =========================
# 预处理：resize + padding
# =========================
def resize_and_pad_pil(
    img: Image.Image,
    target_h: int,
    final_size: int
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    1) 等比例 resize 到高度=target_h
    2) padding 到 final_size×final_size
    返回：(处理后图, meta)
      meta 用于把标签映射回原图：orig_w/h, new_w/h, pad_left/top
    """
    orig_w, orig_h = img.size
    scale = target_h / orig_h
    new_w = int(round(orig_w * scale))
    new_h = target_h

    if new_w > final_size or new_h > final_size:
        raise ValueError(
            f"Resize 后尺寸 {new_w}x{new_h} 超过 pad 目标 {final_size}x{final_size}。"
            f"请减小 resize_target_h 或增大 pad_final_size。"
        )

    img_rs = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = final_size - new_w
    pad_h = final_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    arr = np.array(img_rs)  # H,W,3
    ten = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
    ten = F.pad(ten, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    out = ten.permute(1, 2, 0).numpy().astype(np.uint8)
    out_pil = Image.fromarray(out)

    meta = dict(
        orig_w=orig_w, orig_h=orig_h,
        new_w=new_w, new_h=new_h,
        pad_left=pad_left, pad_top=pad_top,
        final_size=final_size,
        target_h=target_h
    )
    return out_pil, meta


# =========================
# Dataset & Collate
# =========================
class ImgDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        proc, meta = resize_and_pad_pil(img, CFG.resize_target_h, CFG.pad_final_size)
        return path, img.size, proc, meta  # img.size = (orig_w, orig_h)


def collate_fn(batch):
    paths = [b[0] for b in batch]
    orig_sizes = [b[1] for b in batch]
    proc_imgs = [b[2] for b in batch]  # PIL list
    metas = [b[3] for b in batch]
    return paths, orig_sizes, proc_imgs, metas


# =========================
# DINOv3 patch token 提取
# =========================
@torch.no_grad()
def extract_patch_tokens(
    model,
    processor,
    proc_imgs: List[Image.Image],
    device: str,
    use_fp16: bool
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    返回：
      patch_tokens: [B,N,C] (on device)
      (Hp,Wp): patch grid
    """
    inputs = processor(images=proc_imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if use_fp16 and device.startswith("cuda"):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(**inputs)
    else:
        out = model(**inputs)

    x = out.last_hidden_state  # [B,1+N,C]
    patch = x[:, 1:, :]        # [B,N,C]
    B, N, C = patch.shape

    Hp = Wp = int(math.sqrt(N))
    if Hp * Wp != N:
        raise RuntimeError(f"Token grid 非正方形：N={N}, sqrt={math.sqrt(N)}")
    return patch, (Hp, Wp)


# =========================
# GPU KMeans：FAISS
# =========================
def faiss_gpu_kmeans(X: np.ndarray, K: int, niter: int, nredo: int, seed: int) -> np.ndarray:
    """
    X: [M,C] float32 numpy
    返回 centers: [K,C] float32 numpy (L2 normalized)
    """
    try:
        import faiss
    except Exception as e:
        raise ImportError(
            "未检测到 faiss。请用 conda 安装：\n"
            "  conda install -c pytorch -c nvidia faiss-gpu\n"
            f"原始错误：{e}"
        )

    assert X.dtype == np.float32
    M, C = X.shape
    if M < K:
        raise ValueError(f"样本数 M={M} 小于 K={K}，请增大抽样 token 或减小 K。")

    # faiss.Kmeans 支持 gpu=True（使用可见 GPU）
    km = faiss.Kmeans(
        d=C,
        k=K,
        niter=niter,
        nredo=nredo,
        verbose=True,
        seed=seed,
        gpu=True
    )
    km.train(X)
    centers = km.centroids.astype(np.float32)
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-6)
    return centers


# =========================
# 抽样建码本
# =========================
def build_codebook_centers(
    model,
    processor,
    image_paths: List[str]
) -> np.ndarray:
    """
    1) 抽样图片
    2) 提取 patch tokens
    3) 每图随机采样 token
    4) concat -> L2 normalize
    5) FAISS GPU KMeans -> centers
    """
    if len(image_paths) <= CFG.sample_imgs:
        sampled_paths = image_paths
    else:
        sampled_paths = random.sample(image_paths, CFG.sample_imgs)

    ds = ImgDataset(sampled_paths)
    dl = DataLoader(
        ds,
        batch_size=CFG.infer_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=(CFG.num_workers > 0),
        collate_fn=collate_fn
    )

    sampled_tokens = []
    for _, _, proc_imgs, _ in tqdm(dl, desc="Sampling tokens for codebook"):
        patch, _ = extract_patch_tokens(model, processor, proc_imgs, CFG.device, CFG.use_fp16)
        B, N, C = patch.shape
        s = min(CFG.sample_tokens_per_img, N)

        idx = torch.randint(low=0, high=N, size=(B, s), device=patch.device)
        picked = patch.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))  # [B,s,C]
        picked = picked.reshape(-1, C).float().cpu().numpy().astype(np.float32)  # [B*s,C]
        sampled_tokens.append(picked)

    X = np.concatenate(sampled_tokens, axis=0).astype(np.float32)
    # L2 normalize for cosine assignment later
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)

    centers = faiss_gpu_kmeans(
        X=X,
        K=CFG.K,
        niter=CFG.faiss_niter,
        nredo=CFG.faiss_nredo,
        seed=CFG.faiss_seed
    )
    return centers


# =========================
# token -> label（cosine）
# =========================
def assign_labels_cosine(patch_tokens: torch.Tensor, centers_t: torch.Tensor) -> torch.Tensor:
    """
    patch_tokens: [B,N,C] on GPU
    centers_t: [K,C] on GPU (normalized)
    return labels: [B,N] int64
    """
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    sim = patch_tokens @ centers_t.T  # [B,N,K]
    return sim.argmax(dim=-1)


# =========================
# label 回到原图并保存 PNG
# =========================
def save_label_png(save_path: str, label_patch_hw: np.ndarray, meta: dict, orig_size: Tuple[int, int]):
    orig_w, orig_h = orig_size
    new_w = meta["new_w"]
    new_h = meta["new_h"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    final_size = meta["final_size"]

    # 1) Hp×Wp -> 518×518 (nearest)
    lab_518 = cv2.resize(label_patch_hw.astype(np.int32), (final_size, final_size), interpolation=cv2.INTER_NEAREST)

    # 2) 去 padding：得到 new_w×new_h
    lab_crop = lab_518[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    # 3) resize 回原图 orig_w×orig_h (nearest)
    lab_orig = cv2.resize(lab_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 4) 保存类型
    if CFG.save_uint16_if_needed and CFG.K > 255:
        out = lab_orig.astype(np.uint16)
    else:
        out = lab_orig.astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, out, [cv2.IMWRITE_PNG_COMPRESSION, CFG.png_compress])


# =========================
# 主流程
# =========================
@torch.no_grad()
def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    # 收集图片
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(CFG.image_dir, e)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {CFG.image_dir}")

    print(f"[Info] Found {len(image_paths)} images.")
    print(f"[Info] Device={CFG.device}, FP16={CFG.use_fp16}")
    print(f"[Info] Preprocess: resize_h={CFG.resize_target_h}, pad={CFG.pad_final_size}")
    print(f"[Info] K={CFG.K}, sample_imgs={CFG.sample_imgs}, sample_tokens_per_img={CFG.sample_tokens_per_img}")

    # 加载模型
    processor = AutoImageProcessor.from_pretrained(CFG.model_name)
    model = AutoModel.from_pretrained(CFG.model_name).to(CFG.device).eval()

    # 1) 抽样建码本（GPU KMeans）
    centers = build_codebook_centers(model, processor, image_paths)
    np.save(os.path.join(CFG.out_dir, f"codebook_K{CFG.K}.npy"), centers)
    centers_t = torch.from_numpy(centers).to(CFG.device).float()  # [K,C]
    print(f"[Info] Codebook saved: {os.path.join(CFG.out_dir, f'codebook_K{CFG.K}.npy')}")

    # 2) 全量推理打标签
    ds_all = ImgDataset(image_paths)
    dl_all = DataLoader(
        ds_all,
        batch_size=CFG.infer_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=(CFG.num_workers > 0),
        collate_fn=collate_fn
    )

    for paths, orig_sizes, proc_imgs, metas in tqdm(dl_all, desc="Labeling all images"):
        patch, (Hp, Wp) = extract_patch_tokens(model, processor, proc_imgs, CFG.device, CFG.use_fp16)
        labels = assign_labels_cosine(patch, centers_t)  # [B,N]
        labels = labels.view(-1, Hp, Wp).cpu().numpy()

        for p, orig_sz, meta, lab_hw in zip(paths, orig_sizes, metas, labels):
            base = os.path.splitext(os.path.basename(p))[0]
            save_path = os.path.join(CFG.out_dir, base + ".png")
            save_label_png(save_path, lab_hw, meta, orig_sz)

    print(f"[Done] Labels saved to: {CFG.out_dir}")


if __name__ == "__main__":
    # 让抽样可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    main()