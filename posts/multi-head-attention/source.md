<title>Multi-Head Attention：从多组注意力到高效推理</title>

在 Transformer 中，Attention 负责让一个位置从其他位置读取信息。**Multi-Head Attention（MHA）则让同一个位置拥有多套独立的读取权重，再把读取结果融合起来。** 理解它，既要看到“多个头”，也要看清每个头的维度、Softmax 的归一化方向，以及数据在 GPU 上怎样存储和移动。

这篇文章从单头计算出发，逐步走到张量布局、PyTorch 实现、训练 MFU 与 KV Cache。读完后，再看到模型配置中的 <code>hidden_size</code>、<code>num_attention_heads</code> 和 <code>num_key_value_heads</code>，就能把这些参数对应到具体的计算与显存开销。

# 1. 单头 Attention 到底在计算什么

先约定几个符号，默认省略偏置，并令 Query 和 Key 的每头维度相同。

| 符号 | 含义 |
|---|---|
| $B$ | Batch size |
| $L_q$ | Query 序列长度 |
| $L_k$ | Key/Value 序列长度 |
| $D$ | 模型隐藏维度 |
| $H$ | 标准 MHA 的头数 |
| $d$ | 每头维度；常见设置为 $d=D/H$ |

对一个头，输入被投影成 $Q$、$K$、$V$，随后计算：

$$
S=\frac{QK^\top}{\sqrt d}+M,\qquad
A=\operatorname{softmax}_{\text{key}}(S),\qquad
Z=AV
$$

$Q$ 表示当前位置用什么特征寻找信息，$K$ 表示其他位置用什么特征参与匹配，$V$ 则是匹配后真正被加权汇总的内容。$M$ 是加性 Mask：允许访问的位置加 $0$，禁止访问的位置加 $-\infty$。这是原始 Transformer 采用的 Scaled Dot-Product Attention。[Vaswani et al., 2017，§3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)

## 1.1 每一行都是一个独立的读取决策

对第 $i$ 个 Query：

$$
a_{ij}=
\frac{\exp(s_{ij})}
{\sum_{t\in\mathcal V_i}\exp(s_{it})},
\qquad
z_i=\sum_{j\in\mathcal V_i}a_{ij}v_j
$$

$\mathcal V_i$ 是它能够访问的 Key 位置集合。没有 Attention Dropout 时，有效位置上的权重之和为 $1$。

因此，一行权重回答的是“这个 Query 应该怎样读取这些 Value”，**不是下一个 Token 在词表中的概率**。单头可以同时关注多个位置，只是这些 Value 通道共用这一行权重。

Self-Attention 中，Q/K/V 来自同一序列；Cross-Attention 中，Q 来自一个序列，K/V 来自另一个序列。因果 Mask 决定能否看到未来位置。这些区别与“有几个头”是两个独立维度。

## 1.2 缩放因子为什么是 √d

可以做一个简化推导：假设 Q、K 的各个分量相互独立、均值为 $0$、方差为 $1$，则

$$
q^\top k=\sum_{r=1}^{d}q_rk_r,\qquad
\operatorname{Var}(q^\top k)=d
$$

除以 $\sqrt d$ 后，方差回到常数量级。这有助于避免头维度增大时，分数幅度随之增长，使 Softmax 过度尖锐。

这是解释缩放因子的统计模型，并不是说真实网络中的 Q/K 始终独立、标准正态。计算时使用的是**当前头的 Query/Key 维度**，而不是直接把整个隐藏维度 $D$ 放进分母。

# 2. 多头增加了什么

以下先讨论多头自注意力，所有头读取同一输入序列 $X$，第 $h$ 个头学习不同的投影：

$$
Q_h=XW_h^Q,\quad K_h=XW_h^K,\quad V_h=XW_h^V
$$

$$
Z_h=
\operatorname{softmax}\left(
\frac{Q_hK_h^\top}{\sqrt d}+M
\right)V_h
$$

最后沿通道维拼接各头输出，再做一次线性投影：

$$
Y=\operatorname{Concat}(Z_1,\ldots,Z_H)W^O
$$

这里有三个容易混淆的地方：

- **每个头都可以读取完整的输入向量。** $W_h^Q$ 的形状是 $D\times d$，不是把原始输入的某几个通道专门分配给这个头。
- 每个头都有自己的 Attention 分数，并分别执行 Softmax。拆成多个头，得到的是多套归一化后的读取方式。
- $W^O$ 是可学习的融合矩阵，不是简单平均各个头。将它按输入通道分块，可以把输出写成 $Y=\sum_h Z_hW_h^O$。

把各头的投影矩阵沿输出维拼接，便能用一次大的矩阵乘法计算全部头。因此，代码中的“先做大投影，再 reshape 成多个头”与上面的逐头公式一致。

多个 Softmax 一般不能合并成一个 Softmax，因为 Softmax 是非线性的。理解多头的重点，是不同头在融合之前保留了不同的信息选择方式；不能仅从总通道数相同，就认为它等价于一个更宽的头。

## 2.1 一个可以手算的例子

下面只观察**同一个 Query**对三个位置的读取。权重和 Value 均为教学设定，不是实际模型的可视化结果。

<figure class="mha-figure">
  <div class="head-demo">
    <div class="head-demo__row"><strong>Head 1</strong><span class="weight-high">位置 A<b>0.7</b></span><span class="weight-medium">位置 B<b>0.2</b></span><span class="weight-low">位置 C<b>0.1</b></span></div>
    <div class="head-demo__row"><strong>Head 2</strong><span class="weight-low">位置 A<b>0.1</b></span><span class="weight-medium">位置 B<b>0.2</b></span><span class="weight-high">位置 C<b>0.7</b></span></div>
  </div>
  <figcaption>同一个 Query，在两个头中可以得到不同的读取权重。颜色越深，权重越大。</figcaption>
</figure>

| 头 | 三个位置的 Value 向量 | 加权结果 |
|---|---|---|
| Head 1 | $(1,0),(0,1),(1,1)$ | $z_1=(0.8,0.3)$ |
| Head 2 | $(2,0),(0,2),(1,1)$ | $z_2=(0.9,1.1)$ |

例如，Head 1 的结果为：

$$
z_1=0.7(1,0)+0.2(0,1)+0.1(1,1)=(0.8,0.3)
$$

拼接后得到 $(0.8,0.3,0.9,1.1)$，再交给 $W^O$ 融合。到拼接这一步，两个头各自读到了什么仍然被保留着；如果先把权重或输出直接平均，就换成了另一种计算。

# 3. 从公式走到真实张量

以 $B=2$、$L=128$、$D=512$、$H=8$ 为例，每头维度 $d=64$。下面是标准 Self-Attention 的一条完整数据路径。

<figure class="mha-figure">
  <ol class="tensor-flow">
    <li><span>01 · 输入</span><strong>X</strong><code>[B, L, D]</code></li>
    <li><span>02 · 合并投影</span><strong>QKV</strong><code>[B, L, 3D]</code></li>
    <li><span>03 · 拆分与换轴</span><strong>Q, K, V</strong><code>各 [B, H, L, d]</code></li>
    <li><span>04 · 每头计算</span><strong>Attention</strong><code>[B, H, L, d]</code></li>
    <li><span>05 · 合头</span><strong>Concat</strong><code>[B, L, Hd]</code></li>
    <li><span>06 · 输出投影</span><strong>Y</strong><code>[B, L, D]</code></li>
  </ol>
  <figcaption>“拆头”拆的是投影后的通道维；序列长度 L 并没有被分给不同的头。</figcaption>
</figure>

| 步骤 | 本例形状 | 需要留意的轴 |
|---|---|---|
| 输入 $X$ | $[2,128,512]$ | 最后一维是隐藏通道 |
| 合并后的 QKV 投影 | $[2,128,1536]$ | $1536=3\times512$ |
| 分离 Q、K、V 并拆头 | 各 $[2,8,128,64]$ | 每个头仍有完整的 128 个位置 |
| 计算 $QK^\top$ | $[2,8,128,128]$ | 最后两维分别是 Query、Key 位置 |
| 计算 $AV$ | $[2,8,128,64]$ | Key 位置维被求和消去 |
| 合头与输出投影 | $[2,128,512]$ | 拼回通道，不是拼接序列 |

Cross-Attention 只需把两种长度分开：Q 为 $[B,H,L_q,d]$，K/V 为 $[B,H,L_k,d]$，分数矩阵为 $[B,H,L_q,L_k]$。Softmax 始终沿最后的 Key 位置维执行。

# 4. 一段对应公式的 PyTorch 实现

下面实现标准 MHA 的 Self-Attention 子层，适用于**无 Padding、没有 KV Cache 的整段输入**。位置编码、残差连接和归一化由外层 Transformer Block 处理。

~~~python
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_dropout=0.0):
        super().__init__()
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, causal=True):
        batch, length, hidden = x.shape

        # [B, L, 3D] -> [B, L, 3, H, d] -> [3, B, H, L, d]
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # SDPA already applies 1 / sqrt(d).
        per_head = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )

        # [B, H, L, d] -> [B, L, H, d] -> [B, L, D]
        merged = per_head.transpose(1, 2).contiguous()
        merged = merged.view(batch, length, hidden)
        return self.out_proj(merged)
~~~

<code>scaled_dot_product_attention</code> 默认按 Q 的最后一维完成缩放，不能再把 Q 额外除一次 $\sqrt d$。它也不会因为模块调用了 <code>eval()</code> 就自动关闭 Attention Dropout，因此代码在推理模式显式传入 <code>0.0</code>。具体使用哪种计算后端，由设备、数据类型、形状及其他条件决定。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 4.1 Mask 的真假含义并不统一

两个 PyTorch 接口的布尔 Mask 约定正好相反：

| 接口 | 布尔值 True 的含义 |
|---|---|
| <code>F.scaled_dot_product_attention</code> 的 <code>attn_mask</code> | 允许参与 Attention |
| <code>nn.MultiheadAttention</code> 的 <code>attn_mask</code> / <code>key_padding_mask</code> | 禁止访问或忽略该位置 |

所以不能把一个接口的布尔 Mask 原样交给另一个接口。[SDPA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)、[MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)

上面的代码让 API 生成方阵因果 Mask，避免同时传入显式 Mask。当模型引入 Padding、Cross-Attention 或 KV Cache 时，需要重新确定可访问的位置集合。

## 4.2 KV Cache 解码有另一套位置关系

整段训练时，$L_q=L_k=L$，下三角 Mask 很直观。单 Token 解码时，却通常有 $L_q=1$、$L_k=T$：这个 Query 位于历史序列的末尾，应能看到已有的全部 Key。

当前 SDPA 对非方阵 <code>is_causal=True</code> 使用左上对齐的因果关系。直接把上面的代码搬到缓存解码，可能让这个 Query 只看到序列开头。只有一个当前 Query、K/V 全部来自合法历史和当前位置时，可以使用 <code>is_causal=False</code>；分块解码、Padding 和更复杂缓存则应按绝对位置构造正确 Mask。[非方阵因果 Mask 说明](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.causal_upper_left.html)

# 5. 头数增加，计算量会成倍增加吗

**先看保持什么不变。** 对标准 MHA，若固定隐藏维度 $D$，同时令 $d=D/H$，增加头数会让每个头变窄。

以下是由矩阵形状直接得到的估算：忽略偏置、Mask、Softmax、Dropout 和反向传播，并按一次乘法加一次加法计 2 FLOPs。

## 5.1 参数量

将所有头打包后，$W^Q$、$W^K$、$W^V$ 和 $W^O$ 均为 $D\times D$，因此：

$$
P_{\text{MHA}}=4D^2
$$

固定 $D$ 时，单纯改变 $H$ 不会把这部分参数量乘以 $H$。如果每头维度固定，同时继续增加头数，那么总投影宽度也在变大，结论自然不同。

## 5.2 矩阵乘法与中间张量

整段 Self-Attention 的四次投影约需要 $8BLD^2$ FLOPs，而 $QK^\top$ 与 $AV$ 两次乘法合计：

$$
4BHL^2d=4BL^2D
$$

主矩阵乘法中的 $H$ 被 $d=D/H$ 抵消了。但显式 Attention 分数仍有 $BHL^2$ 个元素，Softmax 的工作量也与这个规模相关。**参数量接近、主要 FLOPs 接近，不等于显存和实际速度完全相同。**

例如，在 $B=1,H=32,L=8192$、每元素 2 字节的设定下，一张完整的分数矩阵就占：

$$
1\times32\times8192^2\times2
=4\,294\,967\,296\ \text{bytes}
=4\ \text{GiB}
$$

这只是一个中间张量的理论大小，不是整层训练显存，更不是整个模型的占用。实际实现可以不物化这张完整矩阵；因果计算还可能跳过上三角，改变常数开销。

# 6. 如何计算 MFU

FLOPs 告诉我们一次训练需要完成多少计算，MFU（Model FLOPs Utilization）则把它与**训练速度和硬件峰值**联系起来。这里讨论稠密 Decoder-only 模型的全参数训练；推理没有完整反向传播，不能照搬训练公式中的 $6P$。

## 6.1 从定义开始

设一次完整训练步的模型计算量为 $F_{\text{step}}$，耗时为 $\Delta t$，参与训练的 GPU 总数为 $G$，每张卡在对应精度下的理论峰值为 $F_{\text{peak}}$，则：

$$
\operatorname{MFU}
=\frac{F_{\text{step}}}
{\Delta t\cdot G\cdot F_{\text{peak}}}
=\frac{r_{\text{token}}\cdot f_{\text{token}}}
{G\cdot F_{\text{peak}}}
$$

$r_{\text{token}}$ 是整个训练任务的 Token/s，$f_{\text{token}}$ 是训练每个 Token 的模型 FLOPs。前者乘以后者，得到整个任务每秒完成的模型计算量；再除以所有卡的总峰值，才得到 MFU。

PaLM 提出的口径只计模型前向和反向所需的工作，**不把 Activation Checkpointing 额外执行的重计算重复加进分子**。把这部分执行开销也算进去，对应的是硬件 FLOPs 利用率（HFU）的计数思路。重计算仍会影响耗时，也可能通过允许更大的 Batch 改善最终吞吐。[PaLM，§4.1 与附录 B](https://arxiv.org/html/2204.02311v5#A2)

因此，MFU 不是监控面板中的 GPU 忙碌百分比。GPU 一直在工作，并不意味着它一直在以峰值速度推进模型训练。

## 6.2 把前面的 MHA FLOPs 接进来

第 5 节得到单层 MHA 的主要前向 FLOPs：

$$
F_{\text{MHA,fwd}}\approx 8BLD^2+4BL^2D
$$

一次可训练矩阵乘法的反向通常包含输入梯度、权重梯度两次矩阵乘法，所以前向加反向的主要工作量约为前向的三倍：

$$
F_{\text{MHA,train}}\approx24BLD^2+12BL^2D
$$

但是，**只拿 MHA 的 FLOPs 除以整步训练时间，算不出整模型的 MFU**。还必须纳入 FFN、输出词表投影等计算。

用 $P$ 表示稠密模型参数量、$N$ 表示层数，常见的快捷估算是：

$$
f_{\text{token}}\approx
\underbrace{6P}_{\text{参数矩阵的前向与反向}}
+
\underbrace{12NLD}_{\text{Attention 配对计算}}
$$

第一项来自每个 Token 约 $2P$ 的前向、$4P$ 的反向计算；第二项把所有层的 $QK^\top$ 与 $AV$ 计入。这里沿用 $D=H_qd$；若模型的总 Query 投影宽度不等于隐藏维度，应在第二项使用真实的 $H_qd$。这是 PaLM 常用估算式的同一写法。[PaLM 附录 B](https://arxiv.org/html/2204.02311v5#A2)

需要把两个近似条件写明：

- **$6P$ 是快捷估计。** 精确计数应按实际矩阵乘法累加：输入 Embedding Lookup 不等于完整的矩阵乘法，输出词表投影却需要计算；绑权时不能因为名称不同而重复计算参数。MoE、冻结权重或 LoRA 也不能直接套用“总参数量乘 6”。
- **$12NLD$ 使用完整方阵的 Attention 计数口径。** 若只计因果 Mask 下的有效配对，配对数变为 $L(L+1)/2$，对应项是 $6ND(L+1)$。两种口径不能混着比较，也不能只因为更换了 Attention 内核就随意改变分子。

NVIDIA 的 FLOPs 工具会根据模型结构分别计算投影、FFN 和 Attention 等项；对于混合架构，按实际结构计数比统一套一个参数公式更合适。[NeMo AutoModel FLOPs 说明](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/utils/flops_utils)

## 6.3 先把 Token 数、时间和峰值对齐

固定序列长度时，一次完整 Optimizer Step 的 Token 数为：

$$
T_{\text{step}}
=B_{\text{global}}L
=B_{\text{micro}}\cdot A_{\text{accum}}\cdot G_{\text{DP}}\cdot L
$$

其中 $B_{\text{micro}}$ 是每个数据并行副本的微批大小，$A_{\text{accum}}$ 是梯度累积次数，$G_{\text{DP}}$ 是数据并行副本数。TP、PP 或 CP 共同处理同一批样本的不同部分，不能再把这些并行度乘进 Token 数；分母的 $G$ 则应包含所有参与训练的物理 GPU。

计算前可以按下面的表核对统计范围：

| 项目 | 应采用的范围 |
|---|---|
| Token 数 | 与 FLOPs 公式相匹配的一整步全局 Token 数，不是某张卡的局部计数 |
| 训练步时间 | 同一整步的墙钟时间，涵盖全部微批、反向、通信和优化器更新 |
| 时间窗口 | 使用稳定训练区间的累计 Token 数与总耗时；说明是否包含保存检查点、评估等停顿 |
| GPU 峰值 | 对应实际卡型、训练精度和 dense/sparse 模式的理论峰值 |
| GPU 数量 | 整个任务实际占用的卡数，包括模型并行使用的卡 |

GPU 执行是异步的，计时边界要覆盖已完成的 GPU 工作，不能只计 CPU 发起调用的时间。变长输入则需要按实际长度统计，Attention 项通常涉及 $\sum_i L_i^2$，不能简单用平均长度的平方替代。

同样，SFT 中不参与 Loss 的 Prompt Token 仍然经过模型；Padding 是否真正跳过计算也取决于实现。**“有多少标签 Token”与“模型处理了多少 Token”不是同一件事。**

以 A100 为例，BF16 Tensor Core 的 dense 峰值是 **312 TFLOP/s**；规格表中的 **624 TFLOP/s** 带有稀疏条件，普通稠密训练不能使用后者作为分母。[NVIDIA A100 官方规格](https://www.nvidia.com/en-us/data-center/a100/)

## 6.4 一个完整算例

以下是一组教学假设，采用 $6P+12NLD$ 的完整方阵口径：

| 参数 | 假设值 |
|---|---:|
| 参数量 $P$ | $7\times10^9$ |
| 层数 $N$ | 32 |
| 隐藏维度 $D$ | 4096 |
| 序列长度 $L$ | 4096 |
| 全局 Batch | 128 |
| GPU | 8 张 A100，BF16 dense |
| 完整训练步耗时 | 25 秒 |

先计算每个 Token 的模型 FLOPs：

$$
f_{\text{token}}
=6\times7\times10^9+12\times32\times4096\times4096
\approx48.44\times10^9
$$

一整步处理 $128\times4096=524\,288$ 个 Token，因此：

$$
r_{\text{token}}=\frac{524\,288}{25}
=20\,971.52\ \text{Token/s}
$$

整个任务的模型计算吞吐约为 $1015.91$ TFLOP/s，而 8 张卡的总峰值是 $8\times312=2496$ TFLOP/s：

$$
\operatorname{MFU}
=\frac{1015.91}{2496}
\approx40.7\%
$$

如果只使用 $6P$ 而不计 Attention 配对项，同一组数据会得到约 **35.3%**。这不是训练突然变快或变慢，而是 FLOPs 统计口径变了。长序列下，这种差异尤其需要说明。

下面的函数可以把已有训练日志中的数据代入同一估算式；它只计算指标，不负责测量训练时间：

~~~python
def estimate_training_mfu(
    params, layers, hidden_size, seq_len,
    global_batch, step_seconds, gpu_count, peak_tflops_per_gpu,
):
    # Dense full-parameter training; full-square attention convention.
    # Assumes hidden_size == num_query_heads * head_dim.
    flops_per_token = (
        6 * params + 12 * layers * seq_len * hidden_size
    )
    tokens_per_step = global_batch * seq_len
    tokens_per_second = tokens_per_step / step_seconds
    model_flops_per_second = flops_per_token * tokens_per_second
    system_peak = gpu_count * peak_tflops_per_gpu * 1e12

    return {
        "tokens_per_second": tokens_per_second,
        "model_tflops_per_gpu": model_flops_per_second / gpu_count / 1e12,
        "mfu_percent": 100 * model_flops_per_second / system_peak,
    }
~~~

报告 MFU 时，至少同时给出**FLOPs 公式、模型与序列长度、全局 Batch、GPU 型号和数量、训练精度以及计时范围**。若结果明显超过 100%，先排查 Token 是否重复计数、时间单位、卡数和峰值，以及完整方阵与因果有效配对的口径差异；不能直接据此认为硬件超越了物理峰值。



# 7. 从 MHA 到 MQA 和 GQA

长上下文生成时，除了计算 Attention，还需要反复读取历史 K/V。MQA 和 GQA 通过共享 K/V 头，缩小缓存及相关数据读取量。

把 Query 头数记为 $H_q$，Key/Value 头数记为 $H_{kv}$：

| 结构 | $H_{kv}$ | K/V 怎样共享 |
|---|---:|---|
| MHA | $H_q$ | 每个 Query 头有对应的 K/V 头 |
| GQA | 通常介于 $1$ 与 $H_q$ 之间 | 一组 Query 头共用一个 K/V 头 |
| MQA | $1$ | 全部 Query 头共用一组 K/V |

MQA 保留多个 Query 头，不等于单头 Attention。每个 Query 头仍计算自己的读取权重，只是读取的 K/V 表示被共享。[Shazeer, 2019](https://arxiv.org/abs/1911.02150)

GQA 则在独立 K/V 与完全共享之间提供折中。它改变了模型参数化方式，不能把已有 MHA 权重随意合并，就假定模型输出不变。原论文采用了合并 K/V 投影后继续训练的方案。[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)

## 7.1 KV Cache 可以省多少

假设模型有 $N$ 层，每层结构相同，缓存长度为 $T$，每个元素占 $s$ 字节，并保存所有位置的完整 K/V。逻辑缓存大小为：

$$
M_{\text{KV}}=2BNT H_{kv}ds
$$

这里的 $2$ 对应 K 和 V。取 $B=1,N=32,T=8192,H_q=32,d=128,s=2$，直接代入可得：

| 结构 | K/V 头数 | 全模型逻辑 KV Cache |
|---|---:|---:|
| MHA | 32 | 4 GiB |
| GQA | 8 | 1 GiB |
| MQA | 1 | 128 MiB |

这些是按形状算出的容量，不是实测峰值；不含权重、激活、内存分配碎片、缓存页空洞及并行部署中的复制，也没有采用滑动窗口或 KV 量化。

同样设 $D=H_qd$，忽略偏置，Attention 投影参数量为：

$$
P=2D^2+2DH_{kv}d
$$

其中 Q 和输出投影贡献 $2D^2$，K/V 投影贡献后半项。K/V 头数减少，缓存和这两次投影都会变小；Query 头数并没有同步消失。

## 7.2 Prefill 与 Decode 要分开看

Prefill 一次处理整段输入，Attention 的主矩阵乘法规模是 $O(BL^2D)$。使用 KV Cache 的单 Token 解码则令 $L_q=1,L_k=T$，这部分计算变为 $O(BTD)$，历史 K/V 投影也能复用。

但历史 K/V 仍需要被访问。小 Batch 解码常受到权重和缓存读取带宽影响，MQA 正是针对这类成本提出的。缓存缩小 4 倍，不意味着端到端延迟必然缩小 4 倍；实际速度还取决于其他层、Batch、硬件和内核实现。[MQA 的解码分析](https://arxiv.org/html/1911.02150v1)

# 8. FlashAttention 和多头结构是什么关系

FlashAttention 优化的是**怎样执行 Attention**：通过分块计算与在线 Softmax，减少 GPU 高带宽显存和片上存储之间的数据搬运，避免把完整的 $L\times L$ 中间矩阵反复写回显存。

它保留稠密 Softmax Attention 的数学计算目标，属于精确 Attention 的实现优化；浮点计算顺序仍可能使数值出现细微差异。**它没有把稠密全注意力的主算术复杂度从二次降为线性。**[Dao et al., 2022](https://arxiv.org/abs/2205.14135)

因此，下面三件事可以同时存在：

| 层面 | 解决的问题 |
|---|---|
| 多头结构 | 同一 Query 能否保留多套读取权重 |
| GQA / MQA | 多少 Query 头共享 K/V 表示 |
| FlashAttention | 如何安排计算和存储，降低中间张量与数据搬运开销 |

讨论性能时，应先分清改动属于哪一层，再比较计算量、缓存容量和端到端延迟。

# 9. 每个头都有固定的语义职责吗

“一个头看语法，一个头看指代”可以作为直觉，但不应当被当成网络的设计规则。各头的参数从训练中学到，没有预先命名的职责，也不保证它们彼此正交或承担完全不同的功能。

Voita 等在机器翻译编码器中观察到了关注位置、句法关系和低频词等行为的头，也发现许多头可以剪除。Michel 等的研究同样发现部分头存在冗余，而不同类型的 Attention 对多头的依赖程度并不相同。这些结论有具体模型与任务范围，不能推出“任意大模型都能直接删除大部分头”。[Voita et al., 2019](https://aclanthology.org/P19-1580/)、[Michel et al., 2019](https://arxiv.org/abs/1905.10650)

固定 $D$ 时，增大 $H$ 还会压缩每头维度。关于低秩瓶颈的研究提醒我们，头数和头维度会一起影响表示能力，不能只看头数越多越好。[Bhojanapalli et al., 2020](https://proceedings.mlr.press/v119/bhojanapalli20a.html)

读一个模型的 Attention 配置时，值得同时回答四个问题：Q 有多少头、K/V 有多少头、每头多宽、哪些位置允许相互访问。把它们与具体计算内核分开，MHA 的表示方式和推理成本就更容易理解了。

# 10. 参考资料

1. [Vaswani et al. — Attention Is All You Need（2017）](https://arxiv.org/abs/1706.03762)：Scaled Dot-Product Attention 与多头结构。
2. [Shazeer — Fast Transformer Decoding: One Write-Head is All You Need（2019）](https://arxiv.org/abs/1911.02150)：MQA 与增量解码带宽。
3. [Ainslie et al. — GQA（2023）](https://arxiv.org/abs/2305.13245)：K/V 分组共享与检查点适配。
4. [Dao et al. — FlashAttention（2022）](https://arxiv.org/abs/2205.14135)：精确 Attention 的 IO 优化。
5. [Voita et al. — Analyzing Multi-Head Self-Attention（2019）](https://aclanthology.org/P19-1580/)：头的专门化与剪枝。
6. [Michel et al. — Are Sixteen Heads Really Better than One?（2019）](https://arxiv.org/abs/1905.10650)：多头冗余的实证分析。
7. [Bhojanapalli et al. — Low-Rank Bottleneck in Multi-head Attention Models（2020）](https://proceedings.mlr.press/v119/bhojanapalli20a.html)：头数与头维度的表达能力约束。
8. [PyTorch — scaled_dot_product_attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)：缩放、Mask、Dropout 与后端选择。
9. [PyTorch — MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)：模块接口与 Mask 语义。
10. [Chowdhery et al. — PaLM（2022）](https://arxiv.org/html/2204.02311v5#A2)：MFU 定义及训练 FLOPs 估算。
11. [NVIDIA — A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/)：对应精度的 dense 与 sparse 算力规格。
12. [NVIDIA — NeMo AutoModel FLOPs 工具](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/utils/flops_utils)：按模型结构计算 FLOPs 与 MFU。
