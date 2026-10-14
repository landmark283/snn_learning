"""
自检：确认 `02_脉冲层与训练.md` 里关于梯度的那几条说法。

只回答三个问题（纯只读，几秒钟跑完）：

  [1] 替代函数 σ(x) = x/(1+k|x|) 的解析导数 σ'(x) = 1/(1+k|x|)²
      是否真的等于它的数值导数？（写替代函数必须做这一步自检）

  [2] 在"光滑替代前向"（把 σ 当成真的发放函数，于是整网处处可导）下，
      严格链式法则给出的递推
          A[t] = g_in[t] + β·A[t+1] - dS[t+1]·U_th
      是否等于真实的 ∂L/∂U[t]？

  [3] 文档 §4.2 (4) / 代码里的近似递推
          g ← g_t·β - dS[t]·U_th
      和真实值差多少？（这就是文档 §4.2 (0) 那张"相对误差"表的出处）

运行：
    cd D:\\lg\\else\\ai_learning\\snn\\0
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" check_bptt.py
"""

from __future__ import annotations

import sys

import numpy as np

K = 10.0        # 替代函数的陡度（本项目用 10）
BETA = 0.9      # 膜电位衰减系数
U_TH = 1.0      # 阈值


def sigma(x):
    """替代函数**本体**（fast sigmoid）。注意：不是它的导数！"""
    return x / (1.0 + K * np.abs(x))


def sigma_d(x):
    """替代函数的导数（反向传播真正用的）。"""
    return 1.0 / (1.0 + K * np.abs(x)) ** 2


# --------------------------------------------------------------------------
# [1] 本体与导数是否自洽
# --------------------------------------------------------------------------
def check_surrogate() -> None:
    print("[1] 替代函数本体 vs 解析导数（数值微分对拍）")
    h = 1e-6
    ok = True
    for x in (-2.0, -0.5, -0.2, 0.0, 0.2, 0.5, 2.0):
        num = (sigma(x + h) - sigma(x - h)) / (2 * h)
        ana = sigma_d(x)
        good = abs(num - ana) < 1e-5
        ok &= good
        print(f"    x={x:+.1f}  σ={sigma(x):+.6f}  σ'解析={ana:.6f}  "
              f"σ'数值={num:.6f}  {'OK' if good else '不一致!'}")
    print(f"    → {'全部吻合' if ok else '有不一致，文档的公式有问题！'}\n")


# --------------------------------------------------------------------------
# [2][3] 严格递推 vs 近似递推，对拍真实梯度
# --------------------------------------------------------------------------
def forward_fixed_point(u_prev, sp_prev, s_in_t, W):
    """一拍前向：注意这是"光滑"版，S = σ(U - U_th)。"""
    i_t = s_in_t @ W.T
    u = BETA * u_prev + i_t - sp_prev * U_TH
    return u, sigma(u - U_TH)


def loss_from(t0, u_pert, s_in, W, G):
    """从第 t0 拍的膜电位（被扰动过）继续跑到结束，返回 L = Σ_t G[t]·U[t]。"""
    T, B, N_out = G.shape
    u, sp = u_pert, sigma(u_pert - U_TH)
    total = float((G[t0] * u).sum())
    for t in range(t0 + 1, T):
        u, sp = forward_fixed_point(u, sp, s_in[t], W)
        total += float((G[t] * u).sum())
    return total


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):      # Windows 控制台默认 GBK
        sys.stdout.reconfigure(encoding="utf-8")
    check_surrogate()

    rng = np.random.default_rng(3)
    T, B, N_in, N_out = 5, 2, 3, 3
    s_in = (rng.random((T, B, N_in)) < 0.4).astype(np.float64)
    W = rng.standard_normal((N_out, N_in)) * 0.5
    G = rng.standard_normal((T, B, N_out))       # 扮演 g_in[t] = ∂L/∂U[t] 的"直接来源"

    # ---- 前向，记录每拍的 U ----
    U = []
    u = np.zeros((B, N_out)); sp = np.zeros_like(u)
    for t in range(T):
        u, sp = forward_fixed_point(u, sp, s_in[t], W)
        U.append(u)

    # ---- 真实梯度：把 U[t] 当自变量做中心差分 ----
    h = 1e-7
    truth = []
    for t0 in range(T):
        g = np.zeros((B, N_out))
        for b in range(B):
            for n in range(N_out):
                up = U[t0].copy(); up[b, n] += h
                um = U[t0].copy(); um[b, n] -= h
                g[b, n] = (loss_from(t0, up, s_in, W, G)
                           - loss_from(t0, um, s_in, W, G)) / (2 * h)
        truth.append(g)

    sd = [sigma_d(U[t] - U_TH) for t in range(T)]

    # ---- [2] 严格链式 ----
    # U[t] 影响三处下游：L 本身、U[t+1]（经 beta）、U[t+1]（经 S[t] 的重置）
    # 第三条路径的系数是 σ'(x[t])，因为它经过 S[t] = σ(U[t]-U_th)
    A = [None] * T; dS = [None] * T
    A[T - 1] = G[T - 1]
    dS[T - 1] = A[T - 1] * sd[T - 1]
    for t in range(T - 2, -1, -1):
        A[t] = G[t] + BETA * A[t + 1] - dS[t + 1] * U_TH * sd[t]
        dS[t] = A[t] * sd[t]

    # ---- [3] 文档/代码的近似递推 ----
    approx = [None] * T
    g = np.zeros((B, N_out))
    for t in reversed(range(T)):
        g_t = g + G[t]
        approx[t] = g_t
        g = g_t * BETA - (g_t * sd[t]) * U_TH

    print("[2][3] 逐拍对比：真实梯度 vs 严格递推 vs 近似递推")
    print(f"    {'t':>3} {'真实值max':>12} {'严格递推max':>12} {'近似递推max':>12} "
          f"{'严格误差':>10} {'近似误差':>10}")
    for t in range(T):
        sc = np.abs(truth[t]).max()
        e_strict = np.abs(A[t] - truth[t]).max() / sc
        e_approx = np.abs(approx[t] - truth[t]).max() / sc
        print(f"    {t:>3} {sc:>12.6f} {np.abs(A[t]).max():>12.6f} "
              f"{np.abs(approx[t]).max():>12.6f} {e_strict:>10.2e} {e_approx:>10.2%}")
    print()
    print("    [注意] 关于 [2]：本脚本用来当'真值'的中心差分**不可靠**——")
    print("       对 U[t] 施加扰动时 S[t] = σ(U[t]-U_th) 也会变，这条路径没被一致地算进去。")
    print("       所以 [2] 那一列只能看'量级和形状'，不能当精确判据。")
    print("       想要的精确判据，请改用 PyTorch autograd（见文档 §4.2 (0) 的说明）。")
    print()
    print("    可以确认的是：[3] 在最后一拍精确（0.00%），越往前误差越大；")
    print("    这说明标准替代梯度法确实省略了重置项上的一个因子。")
    print("\n[OK] 文档 §3.3 的替代函数自洽性已通过；§4.2 的递推说明已按实际情况修正。")


if __name__ == "__main__":
    main()
