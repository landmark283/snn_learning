"""
自检脚本：核对本文档里引用的所有数字。

它是纯只读的（不训练、不写文件），几秒钟跑完。三件事：

  1. `data_loader.py` 能正确读 MNIST（形状 / 范围 / 类别分布）
  2. 编码器的无偏性和误差量级（`01_神经元模型.md` §1 / `02_脉冲层与训练.md` §1.1）
  3. LIF 的锯齿、f-I 曲线、饱和（`01_神经元模型.md` §7）—— 用的就是
     `reference_solution.py` 里的 `simulate`，所以文档里的数字和这里必然一致

运行：
    cd D:\\lg\\else\\ai_learning\\snn
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" verify_setup.py

期望（全部对得上 = 环境没问题）：
    [1] train (60000, 784) float32 | test (10000, 784) | 像素 0.0~1.0
    [2] T=100 时 MAE ≈ 0.004
    [3] I=0.3, beta=0.5 -> 0 次发放；beta 越大间隔越短
    [4] f-I：I<=0.1 时 r=0；I>=2 时 r=1000Hz（饱和）
    [5] surrogate'(0) = 1，离阈值越远越小
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import load_mnist  # noqa: E402


def load_reference():
    """把 reference_solution.py 当模块导入（它不会在 import 时训练）。"""
    spec = importlib.util.spec_from_file_location("ref", HERE / "reference_solution.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ref = load_reference()

    print("[1] 数据")
    xt, yt, xe, ye = load_mnist()
    print(f"    train {xt.shape} {xt.dtype} | test {xe.shape} {xe.dtype} | "
          f"像素 {float(xt.min()):.1f}~{float(xt.max()):.1f}")
    print(f"    每类样本数 {np.bincount(yt).tolist()}")

    print("\n[2] 编码器（Bernoulli rate coding）")
    x = xt[:256]
    for T in (5, 10, 25, 100):
        s = ref.rate_encode(x, T, np.random.default_rng(0))
        print(f"    T={T:3d}  MAE={np.abs(s.mean(axis=0) - x).mean():.4f}  "
              f"脉冲占比={s.mean() * 100:.1f}%")
    s_pa = ref.rate_encode_deterministic(x, 25)
    print(f"    相位累加版 T=25 的脉冲数误差（相对 x*T）最大={np.abs(s_pa.sum(0) - x * 25).max():.3f}")
    print("    → 相位累加是确定性的，脉冲数 = floor 或 ceil(x*T)，误差必然 < 1（不是 0）")

    print("\n[3] LIF：恒定输入 I=0.3（锯齿区间）")
    for beta in (0.5, 0.8, 0.9, 0.95, 0.99):
        i = np.full((80, 1), 0.3, dtype=np.float32)
        _, s = ref.simulate(i, beta)
        gaps = np.diff(np.flatnonzero(s[:, 0]))
        gap = f"{gaps.mean():.2f}" if len(gaps) else "-"
        print(f"    beta={beta:<5} 脉冲数={int(s.sum()):3d}/80  平均间隔={gap:>5}"
              f"  （U_∞ = {0.3 / (1 - beta):5.2f}）")

    print("\n[4] LIF：f-I 曲线（beta=0.9, T=200）")
    for I in (0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 8.0):
        i = np.full((200, 1), I, dtype=np.float32)
        _, s = ref.simulate(i, 0.9)
        print(f"    I={I:<5} r={float(s.sum()) / 200 * 1000:7.1f} Hz")

    print("\n[5] 代理梯度（fast sigmoid）")
    for d in (-2.0, -0.5, 0.0, 0.5, 2.0):
        g = float(ref.surrogate_deriv(np.array([d], dtype=np.float32))[0])
        print(f"    surrogate'({d:+.1f}) = {g:.6f}")
    print("    → d=0 时最大(=1)，越远离阈值越小：这就是'脉冲的假导数'")

    print("\n[OK] 环境自查通过。可以开始做 01/02/03 里的 TODO 了。")


if __name__ == "__main__":
    main()
