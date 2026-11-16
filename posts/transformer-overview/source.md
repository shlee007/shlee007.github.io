<title>Transformer 概述</title>

Transformer 是一种处理序列数据的神经网络架构，由 Google 在 2017 年论文《Attention Is All You Need》中提出。今天的 GPT、BERT、T5、Llama、Claude 等模型，核心都建立在 Transformer 上。



它最重要的思想是：

> 让序列中的每个词直接观察其他词，并自动判断哪些词更重要。

这套机制叫做**自注意力（Self-Attention）。**



## 1. **为什么需要 Transformer？**

Transformer 不再逐词传递信息，而是让每个词直接观察整个序列，因此它更擅长捕获长距离关系，也更适合 GPU 并行计算。



## 2. **Transformer 整体结构**

原始 Transformer 是一个Encoder-Decoder架构：

*输入文本 -> 分词 Tokenization -> 词向量+位置信息 -> Encoder 编码器 -> 上下文表示 -> Decoder 解码器 -> 下一个词的概率 -> 输出文本*

Encoder 负责理解输入，Decoder 负责生成输出，现代模型不一定同时使用两者。



## 3. **第一步：文本变成 Token**

模型不能直接处理文字，需要先使用 tokenizer 将文本切分成 Token。

例如：

```Plain Text
“我喜欢机器学习”
→ ["我", "喜欢", "机器", "学习"]
→ [314, 982, 1762, 508]
```



Token 不一定对应完整单词，也可能是：

- 一个汉字
- 一个英文单词
- 单词的一部分
- 标点或特殊符号

例如：

```Plain Text
unbelievable
→ ["un", "believ", "able"]
```

这样可以用有限词表表示几乎所有文本。



## 4. **Token Embedding：把 Token 变成向量**

每个 Token ID 会被映射为一个高维向量：

$$x_i = E[token_i]$$

假设 hidden size 为 4096，**一个 Token 会变成包含 4096 个数字的向量**。

这些数字不是人工设定的，而是在训练过程中学习出来。**含义相近的词，其向量通常也更接近。**

但只有词向量还不够，Transformer 会同时看到整个序列，本身不知道词的顺序：

> 猫吃鱼
> 
> 鱼吃猫

如果不加入位置信息，两句话可能看起来完全相同。

因此实际输入是：

$$h_i = TokenEmbedding_i + PositionEmbedding_i$$

原始 Transformer 使用正弦位置编码，**现代大模型则经常使用 RoPE（旋转位置编码），把位置信息作用到注意力中的 Query 和 Key 上。**



## 5. **核心：Self-Attention**

自注意力要解决的问题是：

> 对当前 Token 来说，序列中的其他 Token 各有多重要？

每个输入向量会经过三个不同的线性变换，得到：

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$$

也就是：

- Query：我正在寻找什么？
- Key：我包含什么信息？
- Value：如果关注我，应该读取什么内容？



### **5.1 计算相关性**

Query 与所有 Key 做点积：

$$S=QK^T$$

**点积越大，说明两个 Token 越相关**。

然后除以 $\sqrt{d_k}$：

$$S=\frac{QK^T}{\sqrt{d_k}}$$

这是为了避免向量维度较大时点积数值过大，导致 Softmax 梯度不稳定。



### **5.2 转换为注意力权重**

$$\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x}}$$

对每一行使用 Softmax 计算，把 Token 之间的相关性分数转换成注意力权重：

$$A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

例如，对于 “小明把苹果递给小红，因为她饿了。”，“她”的注意力权重可能是：

| Token | 权重 |
|-|-|
| 小明 | 0.04 |
| 把 | 0.03 |
| 苹果 | 0.04 |
| 递给 | 0.06 |
| 小红 | 0.58 |
| 因为 | 0.04 |
| 她 | 0.12 |
| 饿了 | 0.09 |

权重之和为1。



### **5.3 汇总信息**

最后用注意力权重对 Value 加权求和：

$$Z=AV$$

完整公式就是：

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

因此，每个 Token 的新表示都融合了整个上下文的信息。



## 6. **Multi-Head Attention：多头注意力**

只进行一次注意力计算，可能只能捕获一种关系。Transformer 因此同时计算多个“注意力头”：

$$head_i=\operatorname{Attention}(Q_i,K_i,V_i)$$

然后拼接：

$$\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(head_1,\ldots,head_h)W_O$$

不同注意力头可以学习不同关系。

需要注意，这些功能不是人为分配的，而是模型训练时自己学到的。

如果模型维度是4096，共有32个头，那么每个头的维度通常是：

$$d_{head}=4096/32=128$$



## 7. **前馈神经网络 FFN**

注意力负责在不同 Token 之间交换信息，之后，每个 Token 会独立通过同一个前馈网络：

$$FFN(x)=\sigma(xW_1+b_1)W_2+b_2$$

原始 Transformer 使用 ReLU，现代模型常使用 GELU 或 SwiGLU。

典型结构是先扩大维度，再压缩回来：4096 -> 16384 -> 4096。

可以理解为：

- Attention：从上下文中读取信息
- FFN：对读取的信息进行加工和记忆

很多研究也认为，FFN 中存储了大量模型学到的事实和模式。



## 8. **残差连接与归一化**

每个模块外面都有残差连接：

$$y=x+\operatorname{Module}(x)$$

这样可以：

- 保留原始信息
- 改善梯度传播
- 让很深的网络更容易训练



同时还会使用 Layer Normalization：

$$\operatorname{LayerNorm}(x) = \gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} +\beta$$

现代大模型经常使用 RMSNorm。



经典的 Pre-Norm Transformer 层是：

$$x'=x+\operatorname{Attention}(\operatorname{Norm}(x))$$

$$y=x'+\operatorname{FFN}(\operatorname{Norm}(x'))$$

一个大模型会堆叠几十甚至上百层这样的结构。



Pre-Norm 的残差通路 $x$ 是一条干净的恒等路径（identity path），从头到尾没有被 LayerNorm  干扰，梯度可以无衰减地直接回传。这正是 Pre-LN **训练更稳定、能堆很深、且通常不需要 learning rate warmup 的原因**。



## 9. **模型怎样生成下一个 Token？**

计算注意力权重：

$$A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

用权重汇总 Value，每个 token 得到融合上下文的新表示：

$$Z=AV$$

合并多个注意力头：

$$H=\operatorname{Concat}(head_1,\ldots,head_n)W_O$$

经过残差连接、归一化和FFN：

$$x'=x+\operatorname{Attention}(\operatorname{Norm}(x))$$

$$y=x'+\operatorname{FFN}(\operatorname{Norm}(x'))$$

重复经过多层 Transformer，每层不断提取和加工上下文信息。

取最后一个位置的隐藏向量。

例如输入：“法国的首都是”，使用“是”所在位置经过最后一层后的向量，映射到词表分数：

$$logits=hW_{\text{vocab}}$$

假设得到：

```Plain Text
巴黎：8.2
伦敦：3.1
北京：1.5
```

再经过 Softmax：

$$P(token)=\operatorname{softmax}(logits)=\frac{e^{logit_i}} {\sum_j e^{logit_j}}$$

得到下一个 token 的概率：

```Plain Text
巴黎：98.9%
伦敦：0.6%
北京：0.1%
其他：0.4%
```

选择或采样一个token，模型可能选择概率最高的“巴黎”，也可能按概率进行采样。

追加token并继续生成：

```Plain Text
法国的首都是 → 巴黎
法国的首都是巴黎 → 继续预测下一个 Token
```



## 10. **Transformer 怎么训练？**

Decoder-only 语言模型通常使用“预测下一个Token”的目标。

假设训练文本是：

```Plain Text
我 喜欢 学习 人工智能
```

训练样本可以看做：

```Plain Text
输入“我”              → 预测“喜欢”
输入“我 喜欢”          → 预测“学习”
输入“我 喜欢 学习”     → 预测“人工智能”
```

实际训练时，这些位置可以并行计算。一次前向传播中，同时计算所有位置的预测结果，而不是计算完前一个位置后才能计算下一个位置。因果掩码负责保证每个位置不会看到未来答案。

损失函数通常是交叉熵：

$$L=-\sum_t \log P(x_t\mid x_{<t})$$

训练过程是：

1. 输入大量文本
2. 模型预测每个位置的下一个Token
3. 比较预测结果和真实答案
4. 反向传播计算梯度
5. 使用优化器更新数十亿参数
6. 重复大量次数

模型并不是把资料原封不动存成数据库，而是在参数中学习语言模式、概念关联和统计规律。



## 11. 推理时的采样方法

模型得到概率后，不一定永远选择概率最大的 Token。

常见方法包括：

- Greedy：每次选择概率最高的 Token，稳定但可能单调。
- Temperature：调节概率分布的随机程度。
- Top-K：只在概率最高的k个token中采样。
- Top-P：只保留累积概率达到p的候选集合。
- Beam Search：同时保留多条候选路径，翻译任务重较常见。

温度公式为：

$$P_i= \operatorname{softmax} \left( \frac{logit_i}{T} \right)$$

- $T<1$：更确定、更保守；
- $T>1$：更多样、更随机；
- $T\to0$：接近直接选择最大概率。



## 12. KV Cache 为什么能加速生成？

生成第 100 个 token 时，前面 99 个 token 的 key 和 value 已经计算过。

如果每次全部重新计算，会非常浪费。因此推理系统会缓存历史 token 的 K 和 V：

- 第 1 步：计算并缓存 K₁、V₁
- 第 2 步：只新增 K₂、V₂
- 第 3 步：只新增 K₃、V₃

这就是 KV Cache。

它能显著加快生成，但上下文越长，KV Cache占用的显存也越大。现代模型会使用 MQA 或 GQA，让多个 Query 头共享较少的 Key/Value 头，以减少内存占用。



## 13. Transformer 的计算复杂度

标准自注意力需要构造一个 n x n 的注意力矩阵，其中 n 是序列长度。

时间和内存复杂度大致为：

$$O(n^2)$$

如果上下文从 4000 增加到 8000，注意力矩阵规模大约会增加到 4 倍。这也是超长上下文成本高的主要原因。

常见优化包括：

| 方法 | 优化类型 | 优化对象 | 是否改变模型行为 |
|-|-|-|-|
| FlashAttention | 计算实现优化 | 显存读写与 GPU 利用率 | 不改变，仍是精确的标准注意力 |
| Sliding Window Attention、Sparse Attention | 注意力模式优化 | 减少参与计算的 Token 对 | 会改变可见范围或连接结构 |
| Linear Attention | 数学形式优化 | 改写或近似注意力公式 | 通常不同于标准 Softmax Attention |
| KV Cache | 自回归推理缓存 | 避免重复计算历史 Token | 通常不改变计算结果，但会占用大量显存 |
| GQA、MQA | 注意力头结构优化 | 减少 KV 头及 KV Cache | 会改变模型结构，需要相应训练 |



## 14. 现代大模型对原始Transformer的改进

现在的 Transformer 通常与 2017 年版本有所不同：

| 模块 | 原始 Transformer | 现代大模型常见方案 |
|-|-|-|
| 位置编码 | 正弦编码 | RoPE、ALiBi |
| 归一化 | LayerNorm | RMSNorm |
| 激活函数 | ReLU | GELU、SwiGLU |
| 注意力 | Multi-Head Attention | GQA、MQA |
| 网络结构 | Encoder-Decoder | Decoder-only居多 |
| FFN | 普通前馈网络 | MoE |
| 计算实现 | 标准Attention | FlashAttention |

MoE（Mixture of Experts）会准备多个“专家网络”，每个 Token 只激活其中少数几个，从而增加参数量，但不让计算量同比增长。



## 15. Transformer 的优势

主要优势包括：

- 并行训练效率高
- 擅长捕获长距离依赖
- 可扩展到数十亿乃至上万亿参数
- 架构通用
- 能处理文本、图像、音频、视频和多模态数据
- 适合预训练后迁移到大量下游任务

例如 Vision Transformer 会把图片切成小块 Patch：

图片 → Patch → 向量 → Transformer

这和文本的：

文本 → Token → 向量 → Transformer

本质非常相似。



## 16. Transformer 的局限

它也存在明显问题：

- 标准注意力对序列长度是平方复杂度；
- 训练需要大量算力、数据；
- 生成是逐token进行的，推理难以完全并行；
- 语言模型优化的是“预测合理文本”，不天然保证事实正确，具有幻觉；

所谓“幻觉”，本质上与训练目标有关：模型倾向于生成统计上连贯的后续文本，而不是先查询一个绝对可靠的事实数据库。



## 17. 一句话理解Transformer

可以把 Transformer 概括为：

> 先通过 Attention 从上下文中收集信息，再通过 FFN 对信息进行加工，利用残差连接保留原始信息，重复很多层后预测下一个 token。

整个大语言模型可以简化为：

$$文本 \rightarrow Token \rightarrow Embedding \rightarrow 多层Transformer \rightarrow Token概率 \rightarrow 文本$$



# 整体架构

经典 Transformer 由 Encoder（编码器）和 Decoder（解码器）两大部分组成，各自堆叠 N 层（原论文 N=6）。

- 编码器：将输入序列转换为向量表示。
- 解码器：获取编码器的输出向量，生成序列。

编码器可以一次性获取到整个序列，而解码器的词是一个一个生成的，这个过程称为自回归（auto-regressive）。



***输入 -> [Encoder x N] -> [Decoder x N] -> 输出***



## 1. 输入处理子结构

**Input/Output Embedding**

- 把离散 token 映射为稠密向量



**Positional Encoding（位置编码）**

- 注入序列顺序信息（原版用正弦/余弦函数；后续有可学习位置编码、RoPE、ALiBi 等变体）



## 2. Encoder 层（每层包含 2 个子层）

**Multi-Head Self-Attention（多头自注意力）**

- 内部由多个缩放点积注意力(Scaled Dot-Product Attention) 并行组成
- 涉及 Q/K/V 线性投影和最终的输出线性投影



**Position-wise Feed-Forward Network（前馈网络，FFN）**

- 通常是两层全连接+激活函数（ReLU/GELU）



每个子层外面都包裹：

- Residual Connection（残差连接）
- Layer Normalization（层归一化）



其中，原版为 Post-LN：

$$x_{l+1} = \text{LayerNorm}\big(x_l + \text{Sublayer}(x_l)\big)$$



现代模型多用 Pre-LN：

$$x_{l+1} = x_l + \text{Sublayer}\big(\text{LayerNorm}(x_l)\big)$$



关键差异：

- Post-LN：LayerNorm 包在「残差和」外面，归一化作用于 $x + Sublayer(x)$。
- Pre-LN：**残差通路 $x_l$ 是一条干净的恒等路径（identity path），从头到尾没有被 LayerNorm  干扰，梯度可以无衰减地直接回传。**这正是 Pre-LN 训练更稳定、能堆很深、且通常不需要 learning rate warmup 的原因。
- Pre-LN 结构中，最后一层输出后通常还会额外加一个 final LayerNorm，因为残差通路上累加的结果本身没有被归一化过，需要在进入输出头之前再归一化一次。



## 3. Decoder 层（每层包含 3 个子层）

1. Masked Multi-Head Self-Attention（带掩码的多头自注意力）- 掩码防止看到未来 token
2. Encoder-Decoder (Cross) Attention（交叉注意力） - Query 来自 Decoder，Key/Value 来自 Encoder 输出。
3. Position-wise Feed-Forward Network（前馈网络，FFN）

同样每个子层都带残差连接+层归一化。



## 4. 注意力机制内部子结构

Multi-Head Attention 进一步拆解为：

- Scaled Dot-Product Attention（缩放点积注意力）：核心计算

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- 线性投影层（Input Linear）：把输入投影成 Q (Query)、K (Key)、V (Value) 三组矩阵
- 多头拆分/拼接：把 Q/K/V 切成 h 个头并行计算，再拼接
- 输出投影层（Output Linear）：把拼接结果映射回模型维度



## 5. 前馈网络（FFN）内部

两层线性变换 + 非线性激活：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

（原版用 ReLU；现代模型常用 GELU、SwiGLU 等）



## 6. 输出子结构

- Linear 层：映射到词表维度
- Softmax：输出下一个 token 的概率分布



## 7. 汇总（核心可复用组件）

| 组件 | 作用 |
|-|-|
| Scaled Dot-Product Attention | 注意力计算的最小单元 |
| Multi-Head Attention | 多个注意力头并行，捕捉不同子空间信息 |
| Position-wise FFN | 逐位置非线性变换 |
| Positional Encoding | 提供位置信息 |
| Residual + LayerNorm | 稳定训练、缓解梯度问题 |



# 当前主流架构（Decoder-only）

当前主流大模型 (GPT、LLaMA、Qwen、DeepSeek、Claude、Gemini 等) 几乎都采用 Decoder-only 架构。

## 1. 三种架构路线对比

| 架构 | 代表模型 | 作用 | 典型场景 |
|-|-|-|-|
| Encoder-only | BERT、RoBERTa | 文本 -> token | 理解类任务（分类、检索） |
| Encoder-Decoder | T5、BART、原始Transformer | 文本 -> token -> 文本 | 序列到序列（翻译、摘要） |
| Decoder-only | GPT系列、LLaMA、Qwen、DeepSeek、Claude | token -> 文本 | 生成式、通用大模型 |

结论：当今主流的通用大语言模型（LLM）=Decoder-only

## 2. 为什么是 Decoder-only

1. **任务范式统一为「下一个 token 预测」**Decoder-only 用自回归方式做 **Next Token Prediction**，几乎所有任务（问答、翻译、写代码、推理）都能统一成「续写」。这种简洁的训练目标让模型易于规模化，无需为不同任务设计不同结构。
2. **训练效率更高**

   - 自监督预训练只需海量纯文本，无需标注的输入-输出对；
   
     <callout emoji="❓">
     **自监督预训练的loss怎么设计？目标函数是什么？**
     核心目标函数是自回归语言建模，本质是**最大化似然/最小化交叉熵**。
     给定一段文本，把它拆成 token 序列 x1, x2, ..., xt，模型的任务是：**看着前面的token，预测下一个token**。“标签”就是文本自身的下一个词，这就是为什么不需要人工标注，文本天然自带监督信号。
     </callout>
   - 因果掩码（causal mask）让每个位置都能产生训练信号，数据利用率高；
   - 结构单一，更容易做分布式训练和工程优化。
3. **扩展性（Scaling Law）表现最好**实践证明 Decoder-only 在「参数量、数据量、算力」同步放大时，性能提升最平滑、最可预测，是目前 Scaling 效果最好的架构。

   - Scaling Laws 论文（OpenAI, 2020）*Scaling Laws for Neural Language Models - 系统研究了 Decoder-only Transformer，发现 loss 与参数量、数据量、计算量呈幂律关系（power law），且非常平滑可预测。这是"可预测地变强"的实证基础。*
   - Chinchilla 论文（DeepMind, 2022）*"Training Compute-Optimal Large Language Models"* 进一步修正了参数量与数据量的最优配比，同样基于 Decoder-only 架构验证。
   - 架构对比实验（Google, 2022）*"What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"*（Wang et al.）直接对比了 Encoder-Decoder、Decoder-only 等组合，发现 Decoder-only + 自回归目标在 zero-shot 泛化上表现最好。
   - 实践佐证GPT-3 → GPT-4、LLaMA 1/2/3、Qwen、DeepSeek 等一路放大参数和数据，性能持续稳定提升，反复验证了 Scaling 的有效性。
4. **上下文学习（In-Context Learning）能力强**Decoder-only 通过 prompt 就能实现 few-shot/zero-shot，无需微调，泛化性强。这一涌现能力在 Encoder-Decoder 上不如它显著。In-Context Learning（ICL）指的是：**模型不更新任何参数，仅通过 prompt 里给出的示例或说明，就能完成新任务。**主要有三种表现形式：

   - Zero-shot（零样本）：直接给指令，不给例子。
   - Few-shot（少样本）：在 prompt 里塞几个示例，模型“照葫芦画瓢”。
   - 指令遵循/思维链（CoT）：给出"让我们一步步思考"这类提示，模型能在上下文中展开推理。
5. **推理时 KV Cache 友好**自回归生成可以缓存历史的 Key/Value，逐 token 生成时高效，工程上易于部署。



**总结：主流大模型选择 Decoder-only，核心原因是它用「自回归生成」统一了所有任务、训练效率与扩展性最优，并天然具备强大的上下文学习能力。**