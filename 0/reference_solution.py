"""
参考实现（完整可跑版）：01/02/03 三个练习的答案都在这一个文件里。
自己写完 02/03 之后再对照，不要一上来就看。

运行：
    cd D:\\lg\\else\\ai_learning\\snn\\0
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" reference_solution.py
    & ... reference_solution.py --n 10000 --epochs 12 --T 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_mnist, one_hot  # noqa: E402

U_TH = 1.0
K_SURROGATE = 10.0


# ==========================================================================
# 1) 编码
# ==========================================================================
def rate_encode(x, T, rng):
    """Bernoulli rate coding：(T, B, N) 的 0/1 脉冲。"""
    return (rng.random((T,) + x.shape) < x[None]).astype(np.float32)


def rate_encode_deterministic(x, T):
    """相位累加：T 步内脉冲数严格 = round(x*T)，无随机性。"""
    thr = (np.arange(1, T + 1) / T)[:, None, None].astype(np.float32)
    return (thr <= x[None]).astype(np.float32)


# ==========================================================================
# 2) LIF 神经元（单步）
# ==========================================================================
def lif_step(u_prev, s_prev, i_t, beta, u_th=U_TH):
    u_t = beta * u_prev + i_t - s_prev * u_th
    s_t = (u_t >= u_th).astype(np.float32)
    return u_t, s_t


def simulate(i_seq, beta, u_th=U_TH, u0=0.0):
    T, N = i_seq.shape
    u = np.full(N, u0, dtype=np.float32)
    s = np.zeros(N, dtype=np.float32)
    u_seq = np.empty((T, N), dtype=np.float32)
    s_seq = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        u, s = lif_step(u, s, i_seq[t].astype(np.float32), beta, u_th)
        u_seq[t], s_seq[t] = u, s
    return u_seq, s_seq


# ==========================================================================
# 3) 代理梯度
# ==========================================================================
def surrogate_deriv(u_centered, k=K_SURROGATE):
    """fast sigmoid: S ≈ U/(1+k|U|)，dS/dU = 1/(1+k|U|)^2。"""
    return 1.0 / (1.0 + k * np.abs(u_centered)) ** 2


# ==========================================================================
# 4) 一层 Linear + LIF
# ==========================================================================
class LIFLayer:
    def __init__(self, n_in, n_out, beta, rng):
        self.W = (rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.beta = beta
        self.n_in, self.n_out = n_in, n_out
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def forward(self, s_in, record_u=True):
        """
        返回 (s_out, cache)。
        record_u=True（默认）时把每一步的膜电位也存进 cache
        —— 反向算代理梯度要用它，输出层做"膜电位读出"也要用它。
        """
        T, B, _ = s_in.shape
        u = np.zeros((B, self.n_out), dtype=np.float32)
        s_prev = np.zeros_like(u)
        s_out = np.empty((T, B, self.n_out), dtype=np.float32)
        u_out = np.empty_like(s_out) if record_u else None
        for t in range(T):
            i_t = s_in[t] @ self.W.T + self.b
            u = self.beta * u + i_t - s_prev * U_TH
            s = (u >= U_TH).astype(np.float32)
            s_out[t] = s
            if record_u:
                u_out[t] = u
            s_prev = s
        return s_out, (s_in, u_out)

    def backward(self, g_in, cache):
        """
        g_in: 入口梯度，形状 (T, B, n_out)。
              **入口一律是 ∂L/∂U**——如果上一层给的是 ∂L/∂S，调用方要自己乘 σ'。见 loss_and_grad。
              注意 g_in[t] 只是 ∂L/∂U[t] 里"出口送来的那一份"，
              "沿时间传回来的那一份"由下面的循环补上，两者相加才是 ∂L/∂U[t]。
        返回 dL/dS_in (T, B, n_in)，即 ∂L/∂(本层输入的脉冲)；
        下一层要用它，得先乘它自己的 σ' 才是它要的 ∂L/∂U。

        严格式（`check_bptt.py` 用中心差分验证过，机器精度）：
            A[t] = g_in[t] + A[t+1] · ( β − U_th · σ'(U[t] − U_th) )
            gW  += A[t].T @ S_in[t]      gb += A[t].sum(0)
        注意权重梯度用的是 A[t] = ∂L/∂U[t]，**不是** A[t]·σ'(U[t]-U_th)——
        后者是 ∂L/∂S 的量级，会让梯度差一个数量级。
        """
        s_in, u_out = cache
        T, B, _ = g_in.shape
        dS_in = np.zeros_like(s_in)
        self.gW[...] = 0.0
        self.gb[...] = 0.0
        a_next = np.zeros((B, self.n_out), dtype=np.float32)
        for t in reversed(range(T)):
            sd = surrogate_deriv(u_out[t] - U_TH)
            a_t = g_in[t] + a_next * (self.beta - U_TH * sd)
            self.gW += a_t.T @ s_in[t]
            self.gb += a_t.sum(axis=0)
            dS_in[t] = a_t @ self.W
            a_next = a_t
        return dS_in


class SNN2:
    def __init__(self, n_in, n_hidden, n_out, beta, seed=0):
        rng = np.random.default_rng(seed)
        self.l1 = LIFLayer(n_in, n_hidden, beta, rng)
        self.l2 = LIFLayer(n_hidden, n_out, beta, rng)
        self.adam = [dict(m=np.zeros_like(p), v=np.zeros_like(p)) for p in self.params()]
        self.step_i = 0

    def params(self):
        return [self.l1.W, self.l1.b, self.l2.W, self.l2.b]

    def grads(self):
        return [self.l1.gW, self.l1.gb, self.l2.gW, self.l2.gb]

    def forward(self, s0):
        """隐藏层脉冲；输出层不发放，只读出膜电位（membrane readout）。"""
        s1, c1 = self.l1.forward(s0)
        s2, c2 = self.l2.forward(s1)
        return s1, s2, (c1, c2)

    def loss_and_grad(self, caches, y_onehot):
        """
        输出层"膜电位读出"：
            u2[t]  = beta*u2[t-1] + W2@s1[t] + b2                  (LIF 的充放电)
            latent[t] = u2[t] * (1 - s2[t-1])                       ← 刚发放那一步被重置，
                                                                       膜电位不作数
            logits = mean_t latent[t]
            loss = softmax 交叉熵

        为什么不用"脉冲计数"当 logits：计数是 0~T 的小整数，太粗，
        而且没发放对应 logits=0，梯度会大面积归零。膜电位是连续的，
        从 3Blue1Brown 式 MLP 过渡过来更自然。
        """
        c1, c2 = caches
        s1_in, u2_out = c2
        T, B, C = u2_out.shape
        s2_prev = np.zeros_like(u2_out)
        s2_prev[1:] = (u2_out[:-1] > U_TH).astype(np.float32)  # 前一步是否发放
        mask = 1.0 - s2_prev
        latents = u2_out * mask

        logits = latents.mean(axis=0)                      # (B, C)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(np.clip((p * y_onehot).sum(axis=1), 1e-9, None)).mean()

        dl_dlogits = (p - y_onehot) / B                    # (B, C)
        # logits = mean_t(latents) → 每个时间步都分到 1/T
        dl_dlatent = np.broadcast_to(dl_dlogits[None] / T, latents.shape).copy()
        dl_du2 = dl_dlatent * mask

        # 输出层的入口：读出头直接给出 ∂L/∂U2[t]
        dl_ds1 = self.l2.backward(dl_du2, c2)
        # 隐藏层的入口必须是 ∂L/∂U1[t]，而 l2 吐出来的是 ∂L/∂S1[t]（这才是它消费的东西）。
        # 两者的关系就是 S1 = σ(U1-U_th)：∂L/∂U1 = ∂L/∂S1 · σ'(U1-U_th)。
        # 这一步漏掉的话，隐藏层会把 σ' 当成 1，梯度偏离一个数量级。
        s0_in, u1_out = c1
        dl_du1 = dl_ds1 * surrogate_deriv(u1_out - U_TH)
        self.l1.backward(dl_du1, c1)
        return loss

    def adam_step(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.step_i += 1
        for i, (p, g) in enumerate(zip(self.params(), self.grads())):
            st = self.adam[i]
            st["m"] = b1 * st["m"] + (1 - b1) * g
            st["v"] = b2 * st["v"] + (1 - b2) * g * g
            mh = st["m"] / (1 - b1 ** self.step_i)
            vh = st["v"] / (1 - b2 ** self.step_i)
            p -= lr * mh / (np.sqrt(vh) + eps)


# ==========================================================================
# 训练
# ==========================================================================
def encode(x, T, rng):
    if rng is None:
        return np.broadcast_to(x, (T,) + x.shape).copy()
    return rate_encode(x, T, rng)


def evaluate(model, x, y, T, batch=256, bernoulli=True, seed=12345):
    """用"输出层膜电位峰值/均值"或"脉冲计数"投票都行；这里用膜电位平均值。"""
    rng = np.random.default_rng(seed)
    correct, fires = 0, []
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        s0 = encode(xb, T, rng if bernoulli else None)
        s1, s2, (c1, c2) = model.forward(s0)
        _, u2 = c2
        s_prev = np.zeros_like(u2)
        s_prev[1:] = (u2[:-1] > U_TH).astype(np.float32)
        logits = (u2 * (1.0 - s_prev)).mean(axis=0)
        correct += int((logits.argmax(axis=1) == yb).sum())
        fires.append(float(s1.mean()))
    return correct / len(x), float(np.mean(fires))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--test-n", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- 先做个 sanity check：编码器的无偏性 ----
    x_train, y_train, x_test, y_test = load_mnist(limit=args.n)
    x_test, y_test = x_test[:args.test_n], y_test[:args.test_n]
    rng0 = np.random.default_rng(0)
    s = rate_encode(x_train[:256], 100, rng0)
    print(f"[编码检查] MAE={np.abs(s.mean(0) - x_train[:256]).mean():.4f} "
          f"(T=100 应 < 0.03)")
    print(f"[数据] train {x_train.shape}  test {x_test.shape}")

    model = SNN2(784, args.hidden, 10, args.beta, seed=args.seed)
    ytr_oh = one_hot(y_train)
    n_batches = len(x_train) // args.batch
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        idx = np.random.default_rng(args.seed + ep).permutation(len(x_train))
        coding_rng = np.random.default_rng(1000 + ep)
        use_bernoulli = ep > max(1, args.epochs // 4)
        losses = []
        for b in range(n_batches):
            sel = idx[b * args.batch:(b + 1) * args.batch]
            s0 = encode(x_train[sel], args.T, coding_rng if use_bernoulli else None)
            s1, s2, caches = model.forward(s0)
            loss = model.loss_and_grad(caches, ytr_oh[sel])
            for g in model.grads():
                np.clip(g, -args.clip, args.clip, out=g)
            model.adam_step(args.lr)
            losses.append(loss)
        if ep % args.eval_every == 0 or ep == args.epochs:
            acc, fire = evaluate(model, x_test, y_test, args.T)
            print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  test acc {acc * 100:6.2f}%  "
                  f"hidden fire {fire * 100:5.1f}%  {time.time() - t0:5.0f}s", flush=True)
        else:
            print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  {time.time() - t0:5.0f}s",
                  flush=True)

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    np.savez(out / "snn2_weights.npz", W1=model.l1.W, b1=model.l1.b,
             W2=model.l2.W, b2=model.l2.b)
    print("权重 →", out / "snn2_weights.npz")


if __name__ == "__main__":
    main()
