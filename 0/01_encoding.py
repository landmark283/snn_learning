"""
练习 1：编码 —— 把 MNIST 的灰度图变成脉冲序列（rate coding）。

核心公式（Bernoulli / 泊松率编码）：

    p_i = x_i            # 像素值当作"每个时间步发放的概率"
    s_i(t) ~ Bernoulli(p_i),  t = 0 .. T-1

于是：像素越亮 → 发得越频繁；T 越大 → 估计越准（噪声 ~ 1/sqrt(T)）。

运行：
    cd D:\\lg\\else\\ai_learning\\snn
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" 01_encoding.py

**注意**：这个文件是练习，`TODO` 没填之前跑它会抛 `NotImplementedError` —— 这是**预期的**。
"""

from __future__ import annotations

import sys

import numpy as np

from data_loader import load_mnist

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))


def rate_encode(x: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    x:   (B, N) float32，取值 0~1
    T:   时间步数
    返回: (T, B, N) float32，取值 {0., 1.}
    """
    # TODO 1: 用 rng.random((T, B, N)) < x[None] 生成脉冲，注意转成 float32
    raise NotImplementedError("TODO 1")


def rate_encode_deterministic(x: np.ndarray, T: int) -> np.ndarray:
    """
    不随机的版本：用"相位累加"让 T 步内的脉冲总数严格等于 round(x*T)。
    好处是同一个像素的编码可复现，调试时更好用。
    """
    # TODO 2（可选）：thr = (np.arange(1, T + 1) / T)[:, None, None]
    #                return (thr <= x[None]).astype(np.float32)
    raise NotImplementedError("TODO 2")


def main() -> None:
    T = 25
    rng = np.random.default_rng(0)
    x_train, y_train, _, _ = load_mnist()

    x = x_train[:64]                       # 一个小批量就够看
    spikes = rate_encode(x, T, rng)        # (T, 64, 784)

    print("spikes.shape =", spikes.shape, spikes.dtype)
    print("取值集合 =", np.unique(spikes))

    # 检查 1：T 步总脉冲数 / T 应该 ≈ 原像素值（无偏估计）
    est = spikes.mean(axis=0)              # (64, 784)
    err = np.abs(est - x)
    print(f"平均绝对误差 = {err.mean():.4f}  （T=25 实测约 0.008，T 越大越小）")
    print(f"估计值范围   = {est.min():.3f} ~ {est.max():.3f}")

    # 检查 2：单个样本的总脉冲数分布
    per_sample = spikes.sum(axis=(0, 2))   # (64,) 每个样本 T 步总共发了多少脉冲
    print(f"每个样本总脉冲数 = {per_sample.min()} ~ {per_sample.max()}"
          f"（均值 {per_sample.mean():.0f} ≈ T × 平均像素值）")

    # 检查 3：不同 T 的估计误差
    for t in (5, 10, 25, 50, 100):
        s = rate_encode(x, t, np.random.default_rng(0))
        print(f"  T={t:3d}  MAE={np.abs(s.mean(axis=0) - x).mean():.4f}")

    # 附：把第一个样本的脉冲栅格图画出来（没装 matplotlib 就跳过）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = __import__("pathlib").Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        axes[0].imshow(x[0].reshape(28, 28), cmap="gray")
        axes[0].set_title("original pixel")
        axes[1].imshow(spikes[:, 0, :].sum(axis=0).reshape(28, 28), cmap="gray")
        axes[1].set_title(f"spike count over T={T}")
        axes[2].imshow(spikes[:, 0, :].T, aspect="auto", cmap="gray_r")
        axes[2].set_title("spike raster (784 neurons x T)")
        axes[2].set_xlabel("time step")
        for ax in axes[:2]:
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        p = out_dir / "fig_encoding.png"
        fig.savefig(p, dpi=130)
        print("已保存", p)
    except ImportError:
        print("（没装 matplotlib，跳过画图）")


if __name__ == "__main__":
    main()
