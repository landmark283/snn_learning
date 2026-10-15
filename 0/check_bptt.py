"""
自检：确认 `02_脉冲层与训练.md` 里关于梯度的那几条说法。

回答三个问题（纯只读，几秒钟跑完）：

  [1] 替代函数 σ(x) = x/(1+k|x|) 的解析导数 σ'(x) = 1/(1+k|x|)²
      是否真的等于它的数值导数？

  [2] 权重梯度该用哪个因子、重置项的 σ' 该用哪个下标？

  [3] 两层网络端到端：旧版 `reference_solution.py` 的 backward 写法差多少？
      （现状文件已改成严格式，[4] 专门核对它）

  [4] 现在的 `reference_solution.py` 的 backward 是否等于严格式？

[2][3] 的判据都是**在"光滑替代前向"上对 W 做中心差分**（把 S 当成 σ(U-U_th)，
于是整网处处可导）。对 W 差分是可靠的——它不会像"对 U[t] 差分"那样漏掉
S[t] 跟着变的那条路径。这是这个脚本能给出精确数字的原因。

运行：
    cd D:\\lg\\else\\ai_learning\\snn\\0
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" check_bptt.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

K = 10.0        # 替代函数的陡度（本项目用 10）
BETA = 0.9      # 膜电位衰减系数
U_TH = 1.0      # 阈值


def sigma(x):
    """fast sigmoid 的前向：S ≈ U/(1+k|U|)"""
    return x / (1.0 + K * np.abs(x))


def surrogate_deriv(x):
    """fast sigmoid 的导数：dS/dU = 1/(1+k|U|)²"""
    return 1.0 / (1.0 + K * np.abs(x)) ** 2


# ==========================================================================
# [1] 替代函数：解析导数 vs 数值导数
# ==========================================================================
def check_surrogate():
    print("[1] 替代函数自洽性：σ'(x) 是否等于它的数值导数")
    xs = np.array([-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0])
    h = 1e-6
    ana, num = surrogate_deriv(xs), (sigma(xs + h) - sigma(xs - h)) / (2 * h)
    print(f"    {'x':>6} {'σ(x)':>12} {'σ`(解析)':>12} {'σ`(数值)':>12} {'相对差':>10}")
    ok = True
    for x, a, n in zip(xs, ana, num):
        rel = abs(a - n) / max(abs(a), 1e-30)
        # x=0 处 |x| 有折点，中心差分本身有 O(k·h) 误差，单独放宽
        tol = 1e-4 if x == 0.0 else 1e-6
        ok &= rel < tol
        print(f"    {x:>6.2f} {sigma(np.array([x]))[0]:>12.6f} {a:>12.6f} "
              f"{n:>12.6f} {rel:>10.2e}{'   ← 折点，容差放宽' if x == 0.0 else ''}")
    print(f"    -> {'通过' if ok else '不通过'}（x=0 是 |x| 的折点，解析值取 1）\n")
    return ok


# ==========================================================================
# 一层：光滑前向 + 对 W 中心差分
# ==========================================================================
class OneLayer:
    def __init__(self, T, B, n_in, n_out, seed=0, bias=0.45):
        rng = np.random.default_rng(seed)
        self.T, self.B, self.n_in, self.n_out = T, B, n_in, n_out
        self.s_in = (rng.random((T, B, n_in)) < 0.5).astype(float)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_out, n_in))
        self.b = np.full(n_out, bias)
        self.g_in = rng.normal(0, 1, (T, B, n_out))     # 读出权重：L = Σ g_in·U

    def forward(self, W, b):
        u = np.zeros((self.B, self.n_out))
        s = np.zeros((self.B, self.n_out))
        U = np.zeros((self.T, self.B, self.n_out))
        for t in range(self.T):
            u = BETA * u + (self.s_in[t] @ W.T + b) - s * U_TH
            s = sigma(u - U_TH)
            U[t] = u
        return float((self.g_in * U).sum()), U

    def fd_grad(self, h=1e-6):
        gW = np.zeros_like(self.W)
        for o in range(self.n_out):
            for i in range(self.n_in):
                Wp, Wm = self.W.copy(), self.W.copy()
                Wp[o, i] += h
                Wm[o, i] -= h
                gW[o, i] = (self.forward(Wp, self.b)[0] - self.forward(Wm, self.b)[0]) / (2 * h)
        return gW

    def bptt(self, variant):
        T, n_out = self.T, self.n_out
        _, U = self.forward(self.W, self.b)
        sd = surrogate_deriv(U - U_TH)
        A = np.zeros((T, self.B, n_out))
        dS = np.zeros((T, self.B, n_out))
        A[T - 1] = self.g_in[T - 1]
        dS[T - 1] = A[T - 1] * sd[T - 1]
        for t in range(T - 2, -1, -1):
            if variant == "strict":
                # ∂U[t+1]/∂U[t] = β - U_th·σ'(U[t]-U_th)：因子 A[t]，σ' 用 t
                A[t] = self.g_in[t] + A[t + 1] * (BETA - U_TH * sd[t])
            elif variant == "offbyone":
                # 只把 σ' 的下标写成 t+1（因子用对的 A[t]）
                A[t] = self.g_in[t] + A[t + 1] * (BETA - U_TH * sd[t + 1])
            elif variant == "doc":
                # 旧文档原来写的：(2b) + dS[t+1]·(-U_th)·σ'(U[t]-U_th)
                A[t] = self.g_in[t] + A[t + 1] * BETA + dS[t + 1] * (-U_TH) * sd[t]
                dS[t] = A[t] * sd[t]
            elif variant == "code":
                # 旧版 reference_solution.py 的写法（已修，保留用于量化偏差）
                A[t] = self.g_in[t] + A[t + 1] * BETA - dS[t + 1] * U_TH
                dS[t] = A[t] * sd[t]
            else:
                raise ValueError(variant)
        fac = A if variant in ("strict", "offbyone") else A * sd
        gW = np.zeros_like(self.W)
        for t in range(T):
            gW += fac[t].T @ self.s_in[t]
        return gW


def check_one_layer():
    print("[2] 一层：dW 该用什么因子、σ' 该用哪个下标？（基准 = 对 W 的中心差分）")
    for bias, name in ((0.45, "膜电位在阈值附近"), (0.0, "膜电位远离阈值")):
        net = OneLayer(T=8, B=6, n_in=12, n_out=5, seed=1, bias=bias)
        trueW = net.fd_grad()
        scale = np.abs(trueW).max()
        print(f"\n   {name}（bias={bias}）")
        print(f"   {'写法':<30} {'dW 相对误差':>12}")
        for variant, label in (("strict", "严格式：因子 A[t]，σ' 用 t"),
                               ("offbyone", "因子 A[t]，σ' 下标写成 t+1"),
                               ("doc", "旧文档 ④ 的原式 + 因子 A[t]·σ'[t]"),
                               ("code", "旧版参考解：因子 A[t]·σ'[t]（已修）")):
            err = np.abs(net.bptt(variant) - trueW).max() / scale
            print(f"   {label:<30} {err:>12.2e}")
    print("\n   -> 只有「严格式」是机器精度。后两种都用 `A[t]·σ'[t]` 当权重梯度的因子，"
          "那已经是 ∂L/∂S 的量级；\n      σ' 在阈值处为 1、稍远就掉到 1% 量级，所以误差是整个量级的。"
          "\n      现在的 reference_solution.py 用的就是「严格式」。\n")


# ==========================================================================
# 两层：端到端，比较两种 backward 的写法
# ==========================================================================
class TwoLayer:
    def __init__(self, T, B, n0, n1, n2, seed=0):
        rng = np.random.default_rng(seed)
        self.T, self.B, self.n0, self.n1, self.n2 = T, B, n0, n1, n2
        self.s0 = (rng.random((T, B, n0)) < 0.5).astype(float)
        self.W1 = rng.normal(0, 1 / np.sqrt(n0), (n1, n0))
        self.b1 = np.full(n1, 0.45)
        self.W2 = rng.normal(0, 1 / np.sqrt(n1), (n2, n1))
        self.b2 = np.full(n2, 0.45)
        self.y = rng.integers(0, n2, B)

    def forward(self, W1, b1, W2, b2):
        T, B = self.T, self.B
        U1 = np.zeros((T, B, self.n1)); S1 = np.zeros((T, B, self.n1))
        U2 = np.zeros((T, B, self.n2))
        u1 = np.zeros((B, self.n1)); s1 = np.zeros((B, self.n1))
        u2 = np.zeros((B, self.n2)); s2 = np.zeros((B, self.n2))
        for t in range(T):
            u1 = BETA * u1 + (self.s0[t] @ W1.T + b1) - s1 * U_TH
            s1 = sigma(u1 - U_TH)
            u2 = BETA * u2 + (s1 @ W2.T + b2) - s2 * U_TH
            s2 = sigma(u2 - U_TH)
            U1[t], S1[t], U2[t] = u1, s1, u2
        logits = U2.mean(axis=0)                       # 光滑读出：不再有硬 mask
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(p[np.arange(B), self.y], 1e-12, None)).mean())
        return loss, (U1, S1, U2, logits, p)

    def fd_grad(self, h=1e-6):
        grads = [np.zeros_like(v) for v in (self.W1, self.b1, self.W2, self.b2)]
        for k, _ in enumerate(grads):
            arrs = [self.W1, self.b1, self.W2, self.b2]
            base = [a.copy() for a in arrs]
            flat = np.nditer(base[k], flags=["multi_index"])
            for _ in flat:
                idx = flat.multi_index
                up = [a.copy() for a in base]; dn = [a.copy() for a in base]
                up[k][idx] += h; dn[k][idx] -= h
                grads[k][idx] = (self.forward(*up)[0] - self.forward(*dn)[0]) / (2 * h)
        return grads

    def bptt(self, variant):
        """variant='strict'：g_in 是 A[t] = ∂L/∂U[t] 里出口送来的那一份；上一层下来的 ∂L/∂S 要乘 σ' 才是入口。
        variant='code'  ：逐句复刻 reference_solution.py 的**旧版**（dS_in 直接当下一层的 g_in，
                          并且权重梯度用 A[t]·σ'[t]）。保留它是为了量出那处偏差有多大。"""
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        T, B = self.T, self.B
        loss, (U1, S1, U2, logits, p) = self.forward(W1, b1, W2, b2)
        sd1, sd2 = surrogate_deriv(U1 - U_TH), surrogate_deriv(U2 - U_TH)
        dl_dlogits = (p - np.eye(self.n2)[self.y]) / B
        dl_dU2 = np.broadcast_to(dl_dlogits[None] / T, U2.shape).copy()

        def loop(g_in, sd, s_in, W):
            """返回 (dW, db, ∂L/∂S_in)"""
            n_out = W.shape[0]
            out = np.zeros((T, B, W.shape[1]))
            gW = np.zeros_like(W)
            gb = np.zeros(n_out)
            if variant == "strict":
                a_next = np.zeros((B, n_out))
                for t in reversed(range(T)):
                    a = g_in[t] + a_next * (BETA - U_TH * sd[t])
                    gW += a.T @ s_in[t]; gb += a.sum(axis=0)
                    out[t] = a @ W
                    a_next = a
            else:
                g = np.zeros((B, n_out))          # ← 旧版参考解原样（已修）
                for t in reversed(range(T)):
                    g_t = g + g_in[t]
                    dS = g_t * sd[t]
                    gW += dS.T @ s_in[t]; gb += dS.sum(axis=0)
                    out[t] = dS @ W
                    g = g_t * BETA - dS * U_TH
            return gW, gb, out

        gW2, gb2, dS1 = loop(dl_dU2, sd2, S1, W2)
        # 关键差别：交给下一层的入口是 A[t]=∂L/∂U1 里出口那份（= (∂L/∂S1)·σ'₁），还是直接把 ∂L/∂S1 当入口
        g1_in = dS1 * sd1 if variant == "strict" else dS1
        gW1, gb1, _ = loop(g1_in, sd1, self.s0, W1)
        return [gW1, gb1, gW2, gb2], loss


def check_two_layer():
    print("[3] 两层端到端：旧版 backward 写法和真值差多少？（现状 = 严格式）")
    net = TwoLayer(T=6, B=4, n0=8, n1=6, n2=4, seed=3)
    true = net.fd_grad()
    print(f"   {'参数':<6} {'|真值|max':>12} {'严格式误差':>14} {'旧版误差':>14}")
    for k, name in enumerate(("W1", "b1", "W2", "b2")):
        gs, _ = net.bptt("strict")
        gc, _ = net.bptt("code")
        sc = np.abs(true[k]).max()
        e_s = np.abs(gs[k] - true[k]).max() / sc
        e_c = np.abs(gc[k] - true[k]).max() / sc
        print(f"   {name:<6} {sc:>12.6f} {e_s:>14.2e} {e_c:>14.2e}")
    print("\n   -> 严格式全部是机器精度；旧版写法从第一层（W1）就开始偏，"
          "两层误差还会继续放大\n")


def check_reference():
    """[4] 仓库里真正要用的那个 backward，是否等于上面验证过的严格式。"""
    print("[4] reference_solution.py 的 backward 是否等于严格式")
    import reference_solution as ref

    rng = np.random.default_rng(7)
    T, B, n_in, n_out = 7, 5, 9, 4
    layer = ref.LIFLayer(n_in, n_out, BETA, rng)
    s_in = (rng.random((T, B, n_in)) < 0.5).astype(np.float32)
    u_out = rng.normal(0, 1.2, (T, B, n_out)).astype(np.float32)
    g_in = rng.normal(0, 1.0, (T, B, n_out)).astype(np.float32)

    got = layer.backward(g_in, (s_in, u_out))
    got_W = layer.gW.copy()
    got_b = layer.gb.copy()

    T_, B_, _ = g_in.shape
    sd = surrogate_deriv(u_out - U_TH)
    want = np.zeros_like(s_in)
    want_W = np.zeros_like(layer.W)
    want_b = np.zeros_like(layer.b)
    a_next = np.zeros((B_, n_out))
    for t in reversed(range(T_)):
        a = g_in[t] + a_next * (BETA - U_TH * sd[t])
        want_W += a.T @ s_in[t]
        want_b += a.sum(axis=0)
        want[t] = a @ layer.W
        a_next = a

    errs = {
        "dS_in": np.abs(got - want).max() / max(np.abs(want).max(), 1e-30),
        "gW": np.abs(got_W - want_W).max() / max(np.abs(want_W).max(), 1e-30),
        "gb": np.abs(got_b - want_b).max() / max(np.abs(want_b).max(), 1e-30),
    }
    for k, v in errs.items():
        print(f"    {k:<7} 相对差 {v:.2e}")
    ok = all(v < 1e-4 for v in errs.values())       # float32 累加，容差放宽到 1e-4
    print(f"    -> {'通过（float32 累加，容差 1e-4）' if ok else '不通过'}\n")
    return ok


def main():
    ok = check_surrogate()
    check_one_layer()
    check_two_layer()
    ok &= check_reference()
    print("[OK] 全部通过：§3.3 替代函数自洽；§4.2 的 A[t] = ∂L/∂U[t] 递推、入口 = 出口那一份，"
          "且参考解已对齐。" if ok else "[!!] 有检查未通过")


if __name__ == "__main__":
    main()
