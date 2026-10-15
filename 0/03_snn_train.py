"""
练习 3：完整的两层 SNN + BPTT + 代理梯度（numpy 手写反向传播）。

网络结构（跟 snntorch 教程 3 的结构一致，只是手写）：

    x (B, 784) --rate encode--> S0 (T, B, 784)
    S0 --Linear(784->H)--> I1 --LIF--> S1 (T, B, H)
    S1 --Linear(H->10) --> I2 --LIF--> 读出膜电位 U2
    logits = mean_t( U2[t] * (1 - S2[t-1]) )        ← 输出层不发放，只读膜电位
    loss   = softmax 交叉熵
    预测   = argmax(logits)

替代梯度（surrogate gradient）：脉冲 S = H(U - U_th) 是阶跃函数，
导数几乎处处为 0，无法直接反向传播。做法是前向用阶跃、反向用一条"形状相近
的光滑曲线"的导数：

    fast sigmoid (snntorch 默认，也是 SpikingJelly 的 ATan 近亲)：
        S ≈ U / (1 + k|U|)          dS/dU ≈ 1 / (1 + k|U|)^2 ,  k = 10

完整反向传播（对一层，B 个样本，T 个时间步）：

    前向：  I[t] = S_in[t] @ W.T + b
            U[t] = beta * U[t-1] + I[t] - S[t-1] * U_th
            S[t] = (U[t] >= U_th)

    反向（记 A[t] = dL/dU[t]）：
            A[t]     = g_in[t] + A[t+1] * ( beta - U_th * surrogate'(U[t] - U_th) )
            dW      += A[t].T @ S_in[t]
            db      += A[t].sum(0)
            dS_in[t] = A[t] @ W                        ← 这是 dL/dS_in，不是 dL/dU

    参数梯度要的因子是 A[t] = dL/dU[t] 本身，**不是** A[t]*surrogate'——
    后者是 dL/dS 的量级，会让梯度差一个数量级（check_bptt.py 有实测）。

    每一层的入口 g_in[t] 也必须是 dL/dU[t]。输出层由读出头给出（读的就是 U2）；
    隐藏层拿到的是上一层吐出的 dL/dS1[t]，**要先乘 surrogate'(U1-U_th)** 才是入口。

    关键：LIF 是有状态的，反向必须一路把 A 沿时间轴传回去，这就是"时序反向传播 (BPTT)"。
    (beta - U_th*surrogate') 里的第二项就是"重置"的贡献，最容易漏。
    surrogate' 该取哪一拍的膜电位，是标准做法唯一的近似之处（见 02 §4.2.3）。

运行：
    cd D:\\lg\\else\\ai_learning\\snn\\0
    & "C:\\Users\\moon side\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" 03_snn_train.py
    & ... 03_snn_train.py --n 10000 --epochs 12 --T 20     # 快速跑
    & ... 03_snn_train.py --model ann                     # 对照组：同样的层，去掉脉冲

**注意**：这个文件是练习，`TODO` 没填之前训练会抛 `NotImplementedError` —— 这是**预期的**。
`--model ann` 那条路不需要任何 TODO，可以先跑它确认数据和环境没问题。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 让脚本无论从哪里被调用，都能 import 到同目录的 data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_mnist, one_hot  # noqa: E402

U_TH = 1.0
K_SURROGATE = 10.0


# ==========================================================================
# 代理梯度
# ==========================================================================
def surrogate_deriv(u_centered: np.ndarray, k: float = K_SURROGATE) -> np.ndarray:
    """fast sigmoid 的导数：1 / (1 + k|U - U_th|)^2 。"""
    # TODO 1
    raise NotImplementedError("TODO 1")


# ==========================================================================
# 一层：Linear + LIF
# ==========================================================================
class LIFLayer:
    def __init__(self, n_in: int, n_out: int, beta: float, rng: np.random.Generator):
        # He 初始化（和 ReLU 网络同理：保持前向方差稳定）
        self.W = (rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.beta = beta
        self.n_in, self.n_out = n_in, n_out
        # 反向累积用的缓存
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def forward(self, s_in, record_u=False):
        """
        s_in: (T, B, n_in)。
        返回 (s_out (T,B,n_out), cache)。
        record_u=True 时额外把每一步的膜电位 u_out 存进 cache。
        **本项目的两层都传 record_u=True**：
          - 输出层要给"膜电位读出"用；
          - 隐藏层要在反向算代理梯度 surrogate'(u - U_th) 时用。
        """
        T, B, _ = s_in.shape
        u = np.zeros((B, self.n_out), dtype=np.float32)
        s_prev = np.zeros_like(u)
        s_out = np.empty((T, B, self.n_out), dtype=np.float32)
        u_out = np.empty_like(s_out) if record_u else None
        for t in range(T):
            # TODO 2: 充电 + 泄漏 + 重置。注意顺序：先用"上一步的 s"扣掉阈值，
            #         再和阈值比较求这一步的 s（snntorch 的写法就是 U[t] 先减 S[t-1]*U_th）
            i_t = s_in[t] @ self.W.T + self.b
            u = None
            # TODO 3: 阈值比较，得到 s（float32 的 0/1）
            s = None
            s_out[t] = s
            if record_u:
                u_out[t] = u
            s_prev = s
        return s_out, (s_in, u_out)

    def backward(self, g_in: np.ndarray, cache) -> np.ndarray:
        """
        g_in: dL/dU[t] 的形状 (T, B, n_out)。**入口一律是 ∂L/∂U**——
              上一层吐出来的是 ∂L/∂S，调用方要先乘 σ' 才是这里要的东西。
        返回: dL/dS_in (T, B, n_in)，即 ∂L/∂(本层输入的脉冲)。
              下一层要用它，同样得先乘它自己的 σ'。

        严格式（`check_bptt.py` 用中心差分验证过，机器精度）：
            A[t] = g_in[t] + A[t+1] * (beta - U_th * surrogate'(U[t] - U_th))
            gW += A[t].T @ s_in[t]        gb += A[t].sum(0)
        权重梯度的因子是 A[t] = ∂L/∂U[t]，**不是** A[t]*σ'。
        """
        s_in, u_out = cache
        T, B, _ = g_in.shape
        dS_in = np.zeros_like(s_in)
        self.gW[...] = 0.0
        self.gb[...] = 0.0
        a_next = np.zeros((B, self.n_out), dtype=np.float32)  # A[t+1]，从 t=T-1 往 t=0 传
        for t in reversed(range(T)):
            # TODO 4: 本拍的 A[t] = ∂L/∂U[t]。两路相加：
            #         ① 外面进来的 g_in[t]
            #         ② A[t+1]（就是 a_next）乘上系数 (β − U_th·σ'(U[t] - U_th))
            #         系数用**本拍**的 U[t]（它代表 ∂U[t+1]/∂U[t]），写成 U[t+1] 就错了一拍
            a_t = None
            # TODO 5: 参数梯度。注意因子是 a_t 本身，**不要再乘 σ'**
            self.gW += None  # a_t.T @ s_in[t]
            self.gb += None  # a_t.sum(0)
            # TODO 6: 往输入传。这是 ∂L/∂S_in，不是 ∂L/∂U（下一层要自己乘 σ'）
            dS_in[t] = None  # a_t @ W
            # TODO 7: 滚动变量。本拍的 a_t 要**原样**留给下一轮（t-1）当 A[t+1] 用；
            #         系数是下一轮才乘的，这里不要再乘 σ'
            a_next = None
        return dS_in

    def step(self, lr: float):
        self.W -= lr * self.gW
        self.b -= lr * self.gb


# ==========================================================================
# 两层 SNN
# ==========================================================================
class SNN2:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, beta: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.l1 = LIFLayer(n_in, n_hidden, beta, rng)
        self.l2 = LIFLayer(n_hidden, n_out, beta, rng)
        self.m = [dict(m=np.zeros_like(p), v=np.zeros_like(p)) for p in self.params()]
        self.step_i = 0

    def params(self):
        return [self.l1.W, self.l1.b, self.l2.W, self.l2.b]

    def grads(self):
        return [self.l1.gW, self.l1.gb, self.l2.gW, self.l2.gb]

    def forward(self, s0: np.ndarray):
        s1, c1 = self.l1.forward(s0, record_u=True)
        s2, c2 = self.l2.forward(s1, record_u=True)
        return s1, s2, (c1, c2)

    def loss_and_grad(self, caches, y_onehot: np.ndarray):
        """
        输出层用"膜电位读出"（membrane readout），这是 SNN 里的标准做法：

            logits = mean_t( u2[t] * (1 - s2[t-1]) )
            loss   = softmax 交叉熵

        括号里那一项是把"刚发放、膜电位被重置"的那些时刻排除掉（重置时刻的
        u 接近 0，不代表任何类别）。

        为什么不用"s2 的脉冲计数"当 logits：计数是 0~T 的整数，分辨率太粗，
        且无脉冲时 logits=0 会导致大片梯度归零。膜电位是连续的。
        """
        c1, c2 = caches
        _, u2 = c2
        T, B, C = u2.shape
        s2_prev = np.zeros_like(u2)
        s2_prev[1:] = (u2[:-1] > U_TH).astype(np.float32)
        mask = 1.0 - s2_prev
        latents = u2 * mask

        logits = latents.mean(axis=0)                    # (B, C)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(np.clip((p * y_onehot).sum(axis=1), 1e-9, None)).mean()

        dl_dlogits = (p - y_onehot) / B                  # (B, C)
        # TODO 8: logits = mean_t(latents) → 每个时间步都分到 1/T
        dl_dlatent = None
        dl_du2 = dl_dlatent * mask

        # 一路回传：输出层 → 隐藏层
        dl_ds1 = self.l2.backward(dl_du2, c2)
        # TODO 9: l2 吐出来的是 ∂L/∂S1（脉冲那一侧），而 l1 要的入口是 ∂L/∂U1，
        #         两者差一个 σ'(U1 - U_th)。漏掉它，隐藏层的梯度会偏一个数量级。
        dl_du1 = None
        self.l1.backward(dl_du1, c1)
        return loss

    def adam_step(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.step_i += 1
        for i, (p, g) in enumerate(zip(self.params(), self.grads())):
            self.m[i]["m"] = b1 * self.m[i]["m"] + (1 - b1) * g
            self.m[i]["v"] = b2 * self.m[i]["v"] + (1 - b2) * g * g
            mh = self.m[i]["m"] / (1 - b1 ** self.step_i)
            vh = self.m[i]["v"] / (1 - b2 ** self.step_i)
            p -= lr * mh / (np.sqrt(vh) + eps)


# ==========================================================================
# 对照组：同样的两层，但把脉冲换成连续激活（ReLU）→ 就是普通 ANN
# ==========================================================================
class ANN2:
    """用连续的 ReLU 替代脉冲，层结构和初始化完全一样，用来对照 loss 曲线。"""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((n_hidden, n_in)) * np.sqrt(2 / n_in)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((n_out, n_hidden)) * np.sqrt(2 / n_hidden)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self.cache = None

    def forward(self, x):
        h = np.maximum(x @ self.W1.T + self.b1, 0)
        z = h @ self.W2.T + self.b2
        self.cache = (x, h)
        return z

    def loss_and_grad(self, z, y_onehot):
        B = len(z)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(np.clip((p * y_onehot).sum(axis=1), 1e-9, None)).mean()
        x, h = self.cache
        dz = (p - y_onehot) / B
        gW2 = dz.T @ h; gb2 = dz.sum(0)
        dh = dz @ self.W2
        dh[h <= 0] = 0
        gW1 = dh.T @ x; gb1 = dh.sum(0)
        return loss, (gW1, gb1, gW2, gb2)

    def step(self, lr, grads):
        for p, g in zip((self.W1, self.b1, self.W2, self.b2), grads):
            p -= lr * g


# ==========================================================================
# 编码
# ==========================================================================
def encode(x: np.ndarray, T: int, rng: np.random.Generator | None) -> np.ndarray:
    """
    rng=None → direct coding（把 0~1 的像素直接当电流，前几个 epoch 收敛更快）
    否则    → Bernoulli rate coding
    """
    if rng is None:
        return np.broadcast_to(x, (T,) + x.shape).copy().astype(np.float32)
    return (rng.random((T,) + x.shape) < x[None]).astype(np.float32)


# ==========================================================================
# 训练 / 评估
# ==========================================================================
def evaluate(model: SNN2, x: np.ndarray, y: np.ndarray, T: int, batch: int = 256,
             coding: str = "bernoulli", seed: int = 12345) -> tuple[float, float]:
    """返回 (准确率, 平均发放率)。发放率是诊断 SNN 是否"死掉/饱和"的关键指标。"""
    rng = np.random.default_rng(seed)
    correct = 0
    fire = []
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        s0 = encode(xb, T, rng if coding == "bernoulli" else None)
        s1, s2, (_, c2) = model.forward(s0)
        _, u2 = c2
        s_prev = np.zeros_like(u2)
        s_prev[1:] = (u2[:-1] > U_TH).astype(np.float32)
        logits = (u2 * (1.0 - s_prev)).mean(axis=0)
        correct += int((logits.argmax(axis=1) == yb).sum())
        fire.append(float(s1.mean()))
    return correct / len(x), float(np.mean(fire))


def train(args) -> None:
    x_train, y_train, x_test, y_test = load_mnist(limit=args.n)
    x_test, y_test = x_test[:args.test_n], y_test[:args.test_n]
    print(f"训练集 {x_train.shape}  测试集 {x_test.shape}  T={args.T}  "
          f"beta={args.beta}  hidden={args.hidden}")

    rng = np.random.default_rng(args.seed)
    model = SNN2(784, args.hidden, 10, args.beta, seed=args.seed)
    ytr_oh = one_hot(y_train)

    n_batches = len(x_train) // args.batch
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(x_train))
        # 前 1/4 的 epoch 用 direct coding：梯度更"密"，起步稳
        coding = "direct" if ep <= max(1, args.epochs // 4) else "bernoulli"
        losses = []
        for b in range(n_batches):
            sel = idx[b * args.batch:(b + 1) * args.batch]
            xb, yb = x_train[sel], ytr_oh[sel]
            s0 = encode(xb, args.T, rng if coding == "bernoulli" else None)
            s1, s2, caches = model.forward(s0)
            loss = model.loss_and_grad(caches, yb)
            # 梯度裁剪：SNN 的 BPTT 容易爆
            for g in model.grads():
                np.clip(g, -args.clip, args.clip, out=g)
            model.adam_step(args.lr)
            losses.append(loss)
        if ep % args.eval_every == 0 or ep == args.epochs:
            acc, fire = evaluate(model, x_test, y_test, args.T)
            print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  "
                  f"test acc {acc * 100:.2f}%  隐藏层发放率 {fire * 100:.1f}%  "
                  f"({time.time() - t0:.0f}s)")
        else:
            print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  "
                  f"({time.time() - t0:.0f}s)")

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    np.savez(out / "snn2_weights.npz", W1=model.l1.W, b1=model.l1.b,
             W2=model.l2.W, b2=model.l2.b)
    print("权重已保存", out / "snn2_weights.npz")


def train_ann(args) -> None:
    """对照组：普通两层 ANN，同一个数据、同一个 lr，看能到多少。"""
    x_train, y_train, x_test, y_test = load_mnist(limit=args.n)
    x_test, y_test = x_test[:args.test_n], y_test[:args.test_n]
    rng = np.random.default_rng(args.seed)
    model = ANN2(784, args.hidden, 10, seed=args.seed)
    ytr_oh = one_hot(y_train)
    yte_oh = one_hot(y_test)
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(x_train))
        for b in range(len(x_train) // args.batch):
            sel = idx[b * args.batch:(b + 1) * args.batch]
            z = model.forward(x_train[sel])
            loss, grads = model.loss_and_grad(z, ytr_oh[sel])
            model.step(args.lr, grads)
        z = model.forward(x_test)
        acc = (z.argmax(axis=1) == y_test).mean()
        print(f"epoch {ep:3d}  loss {loss:.4f}  test acc {acc * 100:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["snn", "ann"], default="snn")
    ap.add_argument("--n", type=int, default=60000, help="训练样本数（先用 10000 试）")
    ap.add_argument("--test-n", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--T", type=int, default=25, help="时间步数")
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.model == "ann":
        train_ann(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
