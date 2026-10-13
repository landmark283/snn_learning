"""
练习 2：手写 LIF 神经元（numpy）。

要实现的动力学（snntorch / SpikingJelly 的"显式欧拉 + 硬重置"版本）：

    U[t] = beta * U[t-1] + I[t] - S[t-1] * U_th
    S[t] = 1 if U[t] >= U_th else 0        # 注意：是"下一时刻"用 S[t-1] 做重置

其中 beta = 1 - dt/tau 是膜电位衰减系数（0 < beta < 1），
U_th = 1 是阈值。beta 越大，膜电位"记得"越久（时间常数越长）。

运行：
    cd D:\\lg\\else\\ai_learning\\snn\\0
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" 02_lif_neuron.py

**注意**：这个文件是练习，没填完 `TODO` 之前跑它会直接抛
`NotImplementedError: TODO 1 / TODO 2` —— 这是**预期的**，不是 bug。
先填 `lif_step` 的两行，再跑。
（想看填好之后的样子，直接跑 `reference_solution.py --n 3000 --epochs 1`，
或者看 `reference_solution.py` 里的 `lif_step` / `simulate`。）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

U_TH = 1.0


def lif_step(u_prev: np.ndarray, s_prev: np.ndarray, i_t: np.ndarray,
             beta: float, u_th: float = U_TH):
    """
    单步更新。三个数组同形状 (N,)。

    返回 (u_t, s_t)。
    """
    # TODO 1: 先"充电 + 泄漏 + 扣掉上一步刚发放的那一份"
    u_t = None
    # TODO 2: 和阈值比较，得到 0/1 的脉冲（float32）
    s_t = None
    raise NotImplementedError("TODO 1 / TODO 2")


def simulate(i_seq: np.ndarray, beta: float, u_th: float = U_TH,
             u0: float = 0.0):
    """
    跑完整段时间序列。

    i_seq: (T, N) 输入电流
    返回: (u_seq (T, N), s_seq (T, N))
    """
    T, N = i_seq.shape
    u = np.full(N, u0, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    u_seq = np.empty((T, N), dtype=np.float32)
    s_seq = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        u, s = lif_step(u, s, i_seq[t].astype(np.float32), beta, u_th)
        u_seq[t] = u
        s_seq[t] = s
    return u_seq, s_seq


def constant_input_sweep() -> None:
    """
    恒定输入下扫描 beta。

    I 取 0.3（比阈值电流 1-beta 稍大），这样能看到"锯齿"——累积几拍才过阈值。
    如果 I 取 1.5 这种大电流，神经元会**每步都发放**，看不到锯齿（那也是一种正确的行为）。
    """
    T = 60
    i = np.full((T, 1), 0.3, dtype=np.float32)
    print(f"{'beta':>6} {'脉冲数/60步':>12} {'平均间隔':>8} {'后半段 U 峰谷':>18}")
    for beta in (0.5, 0.8, 0.9, 0.95, 0.99):
        u, s = simulate(i, beta)
        n = int(s.sum())
        gaps = np.diff(np.flatnonzero(s[:, 0]))
        gap = f"{gaps.mean():.2f}" if len(gaps) else "-"
        uu = u[30:, 0]
        # 离散版的稳态：U_∞ = I/(1-beta)；它 > 1 才会发放
        print(f"{beta:>6.2f} {n:>12d} {gap:>8} "
              f"{uu.min():>8.3f}~{uu.max():<8.3f}  （U_∞={0.3 / (1 - beta):.2f}）")
    print("  → beta 越小泄漏越快，反而更难攒到阈值；beta=0.5 时 U_∞=0.6<1，一次都不发")


def f_i_curve() -> None:
    """f-I 曲线：发放率是输入的（近似）线性函数，这就是 rate coding 的基础。"""
    T = 200
    beta = 0.9
    print("\nI → 发放率 (Hz, 设 dt=1ms):")
    for i_val in (0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        i = np.full((T, 1), i_val, dtype=np.float32)
        _, s = simulate(i, beta)
        rate = float(s.sum()) / T * 1000  # 每步 = 1ms
        print(f"  I={i_val:5.2f}  r={rate:7.1f}")
    print("  → I 小于阈值电流(1-beta=0.1)时 r=0；之后近似线性；I>=2 时饱和在 1000Hz")
    print("    饱和是因为硬重置每步最多发一次（离散时间下的'不应期'）")


def refractory_note() -> None:
    """
    输入远大于阈值 -> 每步都发（这就是"离散时间下的不应期"）。

    注意区分两种情况：
      I < U_th（这里 U_th=1）：重置归零后，至少要充一步才能再发 → 隔步发放
      I > U_th：重置归零后下一步又立刻过阈值 → 每步发放
    """
    T = 12
    _, s_small = simulate(np.full((T, 1), 0.6, dtype=np.float32), 0.9)
    _, s_big = simulate(np.full((T, 1), 10.0, dtype=np.float32), 0.9)
    print("\nI=0.6 (<U_th) 的脉冲序列:", s_small[:, 0].astype(int).tolist())
    print("I=10  (>U_th) 的脉冲序列:", s_big[:, 0].astype(int).tolist())
    print("  → 输入超过阈值后，重置归零也压不住，每步发一个（频率上限 = 1/dt）")


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 60)
    print("实验 A：单神经元，随机输入")
    i_seq = rng.random((30, 1)).astype(np.float32)
    u, s = simulate(i_seq, beta=0.9)
    print("U 范围", float(u.min()), "~", float(u.max()), " 脉冲数", int(s.sum()))

    print("\n" + "=" * 60)
    constant_input_sweep()
    f_i_curve()
    refractory_note()

    # 实验 B：和"纯 IF 神经元"对比（beta=1，无泄漏）
    print("\n" + "=" * 60)
    print("实验 B：IF vs LIF（I=0.3 恒定）")
    for beta in (1.0, 0.9):
        i = np.full((100, 1), 0.3, dtype=np.float32)
        _, s = simulate(i, beta)
        print(f"  beta={beta:.1f} → 脉冲数 {int(s.sum())}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(__file__).resolve().parent / "outputs"
        out.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for beta in (0.8, 0.9, 0.95):
            i = np.full((60, 1), 0.3, dtype=np.float32)   # 小电流才看得到锯齿
            u, s = simulate(i, beta)
            axes[0].plot(u[:, 0], label=f"beta={beta}")
            axes[1].plot(np.flatnonzero(s[:, 0]), np.full(int(s.sum()), beta),
                         "|", ms=12)
        axes[0].axhline(U_TH, ls="--", c="gray", lw=1)
        axes[0].set_title("membrane potential U[t]  (I=0.3)")
        axes[0].legend(); axes[0].set_xlabel("time step"); axes[0].set_ylabel("U")
        axes[1].set_title("spike raster"); axes[1].set_xlabel("time step")
        axes[1].set_yticks([0.8, 0.9, 0.95]); axes[1].set_ylim(0.7, 1.05)
        fig.tight_layout()
        p = out / "fig_lif.png"
        fig.savefig(p, dpi=130)
        print("已保存", p)
    except ImportError:
        print("（没装 matplotlib，跳过画图）")


if __name__ == "__main__":
    main()
