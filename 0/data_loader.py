"""
MNIST 数据加载 —— 只依赖 numpy，供本目录下的练习脚本使用。

数据不用重新下载：仓库里 nn_3b1b/data/ 下已经有一份验证过的 MNIST
（4 个 IDX 原始文件 + 导出好的 .npy）。本文件只是把路径指过去。

用法：
    from data_loader import load_mnist, one_hot
    x_train, y_train, x_test, y_test = load_mnist()   # (60000,784) float32 0~1

数据位置：
    <仓库根>\\nn_3b1b\\data\\mnist_tv\\MNIST\\raw\\   ← 原始 IDX（权威）
    <仓库根>\\nn_3b1b\\data\\mnist_np\\              ← 导出的 .npy（快）

本文件用"向上找 nn_3b1b"的方式定位数据，**不依赖自己在第几层目录**。
所以 0/ 这一层目录以后再改名、再套一层，这里都不用跟着改。

如果哪天这份数据被删了，重建命令（在仓库根目录）：
    pymanager exec -V:3.14 nn_3b1b\\00_prepare_data.py
"""

from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

import numpy as np


def _find_data_root() -> Path:
    """从本文件所在目录逐级向上，找那个含有 nn_3b1b/data 的祖先目录。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "nn_3b1b" / "data").is_dir():
            return parent
    raise FileNotFoundError(
        f"从 {here} 向上找不到 nn_3b1b/data —— 请确认本文件还在 ai_learning 仓库里，"
        f"或者手动设置 RAW_DIR / NP_DIR。"
    )


_REPO = _find_data_root()
RAW_DIR = _REPO / "nn_3b1b" / "data" / "mnist_tv" / "MNIST" / "raw"
NP_DIR = _REPO / "nn_3b1b" / "data" / "mnist_np"


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def _read_idx(path: Path) -> np.ndarray:
    """解析 IDX 大端格式。图像返回 float32 0~1，标签返回 uint8。"""
    with _open_maybe_gz(path) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF  # 0x03 = 图像(3维)，0x01 = 标签(1维)
        dims = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        buf = f.read(int(np.prod(dims)))
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(dims)
    return arr.astype(np.float32) / 255.0 if ndim == 3 else arr.astype(np.uint8)


def _find_raw(stem: str) -> Path:
    for name in (f"{stem}.gz", stem):
        p = RAW_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"找不到 {stem}（目录：{RAW_DIR}）。\n"
        f"数据应该还在的；如果没有了，在仓库根目录跑："
        f"pymanager exec -V:3.14 nn_3b1b\\00_prepare_data.py"
    )


def load_mnist(flatten: bool = True, limit: int | None = None):
    """
    返回 (x_train, y_train, x_test, y_test)。

    flatten=True  → x 形状 (N, 784)，float32，0~1
    flatten=False → x 形状 (N, 1, 28, 28)，方便画图
    limit         → 只取前 limit 个训练样本（调参时用，快很多）
    """
    if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台默认 GBK
        sys.stdout.reconfigure(encoding="utf-8")

    if (NP_DIR / "x_train.npy").exists():
        xt = np.load(NP_DIR / "x_train.npy")
        yt = np.load(NP_DIR / "y_train.npy")
        xe = np.load(NP_DIR / "x_test.npy")
        ye = np.load(NP_DIR / "y_test.npy")
    else:
        xt = _read_idx(_find_raw("train-images-idx3-ubyte")).reshape(-1, 784)
        yt = _read_idx(_find_raw("train-labels-idx1-ubyte"))
        xe = _read_idx(_find_raw("t10k-images-idx3-ubyte")).reshape(-1, 784)
        ye = _read_idx(_find_raw("t10k-labels-idx1-ubyte"))

    if limit is not None:
        xt, yt = xt[:limit], yt[:limit]
    if not flatten:
        xt = xt.reshape(-1, 1, 28, 28)
        xe = xe.reshape(-1, 1, 28, 28)
    return xt, yt.astype(np.int64), xe, ye.astype(np.int64)


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


if __name__ == "__main__":
    xt, yt, xe, ye = load_mnist()
    print("train", xt.shape, xt.dtype, "test", xe.shape)
    print("像素范围", float(xt.min()), "~", float(xt.max()))
    print("每类样本数", np.bincount(yt).tolist())
