# Qwen3.6-35B-A3B SFT 训练配置逐项讲解

本文基于两个文件：

- `/Users/shawn/Downloads/train-qw36-xmlsft-nt-lrfix.yaml`
- `/Users/shawn/Downloads/sft_qwen36_a3b.sh`

结论先放前面：这不是一个“单纯 bash 跑 PyTorch”的脚本，而是一个在容器里启动 Ray 集群，再由 Ray 提交 `train_async.py` 的 Megatron/slime SFT 任务。YAML 的 `envargs` 会被平台注入成环境变量，脚本再把环境变量拼成 `MODEL_ARGS + CKPT_ARGS + SFT_ARGS + OPTIMIZER_ARGS + PARALLEL_ARGS + MISC_ARGS`。所以排查时不要只看 YAML，要看脚本最后是否真的把参数传给了 `train_async.py`。

## 1. 这次训练到底在做什么

这是一个 SFT，即 supervised fine-tuning，监督微调任务。它不是从零预训练 35B 模型，而是在已有 Qwen3.6-35B-A3B 权重上，用你的 JSONL 对话数据继续训练，让模型更倾向于生成你数据里 assistant 的回答风格。

一次训练 step 的核心链路是：

1. 从 JSONL 读一批样本。
2. 取每条样本里的 `messages` 字段。
3. 用 Qwen 的 chat template 把多轮消息变成 token 序列。
4. 构造 labels：通常只让 assistant 回复部分参与 loss，system/user/padding 不参与。
5. 把多条变长序列 packing 到一个或多个长序列块里。
6. 按 `micro_batch_size` 做 forward。
7. 算 SFT loss，本质是 next-token cross entropy。
8. backward 得到梯度。
9. 跨 GPU 做必要的通信，比如 DP 梯度规约、CP 序列通信、MoE expert all-to-all。
10. Adam optimizer 更新参数。
11. 学习率 scheduler 前进一步。
12. 记录日志；如果 step 命中 `save_interval`，保存 checkpoint。

这也是理解所有参数的主线：有些参数控制数据如何变成 token，有些控制每个 step 吃多少 token，有些控制模型怎么切到 8 张卡上，有些控制优化器怎么更新，有些控制 checkpoint 和日志。

## 2. 运行脚本的真实流程

脚本大致分成这些阶段：

1. 启动 GPU keeper，避免平台回收 GPU。
2. 设置 bash 严格模式、时区、conda。
3. 解析多机变量：`MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK`。
4. 检查 `SLIME_ROOT`、`MODEL_PATH`、`REF_LOAD`、`LOAD_PATH`、`TRAIN_DATA` 是否存在。
5. 设置 Triton/CUDA 编译缓存目录，减少重复编译。
6. `source "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"`，拿到模型结构参数 `MODEL_ARGS`。
7. 把 position embedding patch 成 Qwen3.6 的 `mrope`。
8. 安装或检查 transformers。
9. 启动 Ray head/worker。
10. rank0 上执行：

```bash
python3 train_async.py \
  --actor-num-nodes "${NNODES}" \
  --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PARALLEL_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
```

一个重要细节：`micro_batch_size`、`seq_len`、`train_iters`、`chunked_lm_head` 这些在主脚本里没有直接搜到传参位置，很可能被 `qwen3.5-35B-A3B.sh` 这个被 source 的模型脚本消费，也可能由训练框架直接读环境变量。真正排查时要在运行日志里找最终命令，确认是否出现了 `--micro-batch-size`、`--seq-length`、`--train-iters` 等参数。

## 3. `input_key: messages` 是什么

YAML 里：

```yaml
input_key: messages
```

脚本中对应：

```bash
--prompt-data "${TRAIN_DATA}"
--input-key "${INPUT_KEY:-messages}"
```

意思是：训练数据是 JSONL，每一行是一条 JSON，训练代码会从每行 JSON 里取 `messages` 这个字段作为对话内容。

典型样子：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

SFT 不应该让模型学习“复读 user prompt”，而是学习“在这些上下文后生成 assistant 回复”。所以数据进入模型后会做 loss mask：user/system token 通常只作为上下文，assistant token 才作为预测目标。你脚本里有：

```bash
--loss-type sft_loss
--loss-mask-type qwen3_5
--calculate-per-token-loss
```

这说明训练 loss 是 SFT cross entropy，并按 Qwen 的消息格式生成 mask。排查数据问题时，`input_key` 错了会导致读不到样本；messages 格式错了会导致 chat template 或 loss mask 异常；assistant 内容为空会导致有效训练 token 极少，loss 看起来不动。

## 4. `load_path`、`ref_load`、`model_path` 的区别

YAML 里有三类路径：

```yaml
model_path: .../Qwen3.6-35B-A3B
load_path: .../Qwen3.6-35B-A3B_torch_dist_tp2_pp1_cp1_ep8_mrope
ref_load:  .../Qwen3.6-35B-A3B_torch_dist_tp2_pp1_cp1_ep8_mrope
save_path: .../lhs_train_qw36
```

脚本对应：

```bash
--hf-checkpoint "${MODEL_PATH}"
--ref-load "${REF_LOAD}"
--load "${LOAD_PATH}"
--save "${SAVE_PATH}"
```

`model_path` 是 HuggingFace 格式目录，通常有 `config.json`、tokenizer 文件、若干 `safetensors` 权重分片。脚本用它加载 config、tokenizer 和 Qwen 动态代码。

`load_path` 是训练实际加载的 Megatron 分布式 checkpoint。训练不是直接读 HF safetensors，而是读已经转换成 Megatron 分布式格式的权重。

`save_path` 是本次训练输出 checkpoint 的目录。

`ref_load` 是 reference model checkpoint。对于 PPO/GRPO/DPO 这类 RLHF 或偏好训练，reference model 是冻结的基准模型，用于计算 KL 或 reference logprob，防止当前模型偏离太远。你这次是 SFT，脚本里还传了：

```bash
--disable-compute-advantages-and-returns
--debug-train-only
--loss-type sft_loss
```

所以它不是典型 RL 训练。这里保留 `ref_load` 更像是 slime 框架统一接口的要求，或用于初始化/reference actor/对齐 checkpoint。因为你的 `load_path` 和 `ref_load` 相同，首轮 SFT 就是从 base/reference 权重起步。

## 5. Megatron 分布式 checkpoint 怎么得到

原始开源模型一般是 HF checkpoint；Megatron 训练需要把权重切成与 Megatron 并行拓扑匹配或可重分片的分布式 checkpoint。

转换流程通常是：

1. 读取 HF 模型 config/tokenizer/权重。
2. 用 Megatron/Bridge/slime 构造相同结构的 Qwen MoE 模型。
3. 做参数名映射：例如 QKV 合并/拆分、MLP gate/up/down 权重映射、MoE expert 权重映射。
4. 按 TP/PP/EP 等并行方式切分权重。
5. 写出 Megatron distributed checkpoint 元数据和各 rank 分片。

你路径里的名字：

```text
Qwen3.6-35B-A3B_torch_dist_tp2_pp1_cp1_ep8_mrope
```

大概表示这个 checkpoint 是 `torch_dist` 格式，转换时的形状是 `TP=2, PP=1, CP=1, EP=8`，并且使用 `mrope`。你的训练 YAML 是 `TP=1, PP=1, CP=4, EP=8`。如果底层 distributed checkpoint 支持重分片，加载时可以从一种并行形状恢复到另一种形状；如果自定义 Qwen/MoE/mrope conversion 不完全支持，就会出现 shape mismatch、missing key、unexpected key、rank shard 不匹配等错误。

实际工程里，拿到 Megatron checkpoint 的方式通常有三种：

1. 官方或平台已经提前转换好，你直接填 `load_path`。
2. 用 Megatron Bridge 或项目里的 conversion script 从 HF 转。
3. 从前一次 Megatron 训练任务的 `save_path` 继续训。

排查转换是否正确，重点看：

- checkpoint 目录是否完整，不能只拷贝某个 rank 分片。
- 是否有最新 iteration 元数据，比如 `latest_checkpointed_iteration.txt` 一类文件。
- 转换使用的 Qwen 结构、mrope、MoE expert 数、tokenizer 是否与训练一致。
- load 日志里是否显示成功加载所有参数。

## 6. 什么是 step，和 epoch 有什么区别

在你的脚本里，`save_interval: 8` 表示每 8 个训练 iteration 保存一次；这里的 iteration/step 通常指一次 optimizer update，而不是一次 forward，也不是一条样本。

一次 optimizer step 包含：

1. 取够一个 global batch，或者取够 `global_batch_tokens`。
2. 切成多个 micro-batch。
3. 每个 micro-batch forward/backward，梯度累积起来。
4. 梯度通信和规约。
5. optimizer 更新一次模型权重。
6. learning rate scheduler 前进一步。

所以 `step=8` 的含义是模型参数已经更新了 8 次。

epoch 是数据概念，表示“完整看完一遍训练集”。`num_epoch: 3` 表示数据层面重复 3 遍。区别是：

- step 关心参数更新次数。
- epoch 关心数据集被遍历了几遍。

当所有样本长度差不多时，可以粗略说：

```text
steps_per_epoch = ceil(num_samples / global_batch_size)
```

但你这里用了 token-based batching：

```yaml
global_batch_tokens: 8388608
```

更合理的估算是：

```text
steps_per_epoch = ceil(total_tokens_one_epoch / global_batch_tokens)
train_steps ≈ steps_per_epoch * num_epoch
```

因此 step 数要用 tokenizer 实际统计后的 token 数来估。

## 7. `global_batch_size`、`global_batch_tokens`、`rollout_batch_size`、`micro_batch_size`

这几个最容易混。

`micro_batch_size: 1`：每张 GPU 在一次 forward/backward 里处理的样本数。长上下文 128K 非常吃激活显存，所以这里设成 1 很常见。micro batch 小不代表全局 batch 小，因为可以梯度累积，也可以多 GPU 并行。

`global_batch_size: 1024`：一次 optimizer update 覆盖的全局样本数目标，跨所有 GPU、所有 data parallel rank、所有梯度累积。没有 token batching 时，它就是每 step 的样本数。

`global_batch_tokens: 8388608`：一次 optimizer update 覆盖的全局 token 数目标，即 8,388,608 tokens，约 8M tokens。设置它后，训练更像“每步固定 token 预算”，而不是“每步固定样本条数”。对于变长数据，这个更稳定，因为显存、计算量、loss 统计都主要由 token 数决定。

`rollout_batch_size: 1024`：rollout 阶段一次从数据源取/处理的样本块大小。SFT 里 rollout 不是让模型生成答案，而是 `slime.rollout.sft_rollout.generate_rollout` 这类函数负责读样本、tokenize、构造训练 batch。它更像数据生产端的批大小/缓冲大小。

一句话区分：

- `micro_batch_size`：单次 forward/backward 每 GPU 吃多少样本。
- `global_batch_size`：一个 optimizer step 总共多少样本。
- `global_batch_tokens`：一个 optimizer step 总共多少 token。
- `rollout_batch_size`：数据/rollout 侧一次处理多少原始样本。

## 8. 变长数据如何满足 `global_batch_tokens`

每条训练数据 token 长度不固定，所以不可能严格保证每步正好等于 8,388,608 tokens。框架一般会这样做：

1. 先 tokenize，知道每条样本长度。
2. 按 `seq_len` 截断或过滤超长样本。
3. 把多个短样本 packing 到长序列块中。
4. 贪心或近似装箱，让每个 step 的 token 总数接近 `global_batch_tokens`。
5. 对不足的位置 padding，并用 mask 避免 padding 参与 loss。
6. 如果启用了 `data_pad_size_multiplier: 4096`，还会把长度向 4096 的倍数对齐。

所以实际 token 数是“接近预算”，不是数学上恒等。你脚本里同时打开：

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 32768
--packing-safety-margin 0.98
--data-pad-size-multiplier 4096
--global-batch-tokens 8388608
```

这说明它会动态决定每个 micro batch/step 能塞多少 token，同时留显存余量。

## 9. `max_tokens_per_gpu: 32768` 怎么来的

你的 YAML：

```yaml
seq_len: 131072
context_parallel_size: 4
max_tokens_per_gpu: 32768
```

这三个数正好满足：

```text
131072 / 4 = 32768
```

这不是巧合。CP=4 时，一个 128K 上下文会沿序列维度切到 4 个 context parallel rank 上，每个 rank 本地大约处理 32K token 的序列片段。`max_tokens_per_gpu=32768` 就是在告诉 dynamic batching：每张 GPU 本地 token 上限大约是 32K。

如果 CP=8，类似上限会变成：

```text
131072 / 8 = 16384
```

脚本默认值正好是 `16384`，说明它以前的默认注释/参数更偏向 CP=8，而你的 YAML 改成了 CP=4。

这个参数不是从模型参数量直接公式算出来的，而是结合以下因素压测出来的：

- 序列长度 `seq_len`
- CP 切分数
- GPU 显存大小
- 是否保存 optimizer 状态
- 是否开启 recompute
- 是否开启 FP8
- attention 实现
- MoE expert 权重如何切分
- padding/packing 带来的额外浪费

如果 OOM，优先降低 `max_tokens_per_gpu` 或增大 CP；如果吞吐低且显存很空，可以试着提高它。

## 10. `packing_safety_margin: 0.98`

packing 时理论上每张卡最多 32768 tokens，但实际训练还会有 padding、kernel workspace、通信 buffer、临时张量、碎片化。`0.98` 表示只用 98% 的理论容量，留 2% 安全边界。

粗略看：

```text
32768 * 0.98 ≈ 32112
```

再加上 `data_pad_size_multiplier=4096`，实际可装箱边界会受到 4096 对齐影响。这个参数能减少“看起来没超上限但实际 OOM”的情况，代价是轻微降低 packing 利用率。

## 11. TP、PP、CP、EP、ETP 如何配合

你的 YAML：

```yaml
gpus_per_node: 8
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
context_parallel_size: 4
expert_model_parallel_size: 8
expert_tensor_parallel_size: 1
```

脚本把它们传成：

```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 4
--expert-model-parallel-size 8
--expert-tensor-parallel-size 1
```

### TP: Tensor Parallel

TP 把单层里的大矩阵切到多张 GPU 上，例如 attention projection 或 MLP projection。`TP=1` 表示不切 dense 矩阵。好处是通信少、实现简单；代价是每张相关 GPU 要放完整 dense 权重。

脚本里只有 `TP > 1` 才加：

```bash
--sequence-parallel
```

你这里 TP=1，所以 sequence parallel 不启用。

### PP: Pipeline Parallel

PP 把 Transformer 层按深度切开，例如前 20 层在一组 GPU，后 20 层在另一组 GPU。`PP=1` 表示不切层。好处是没有 pipeline bubble，也没有 stage 间激活发送；代价是每个模型副本要容纳全部层。

### CP: Context Parallel

CP 沿序列长度切分激活。你的上下文长度是 131072，非常长；CP=4 后每张 GPU 本地只处理约 32768 token 的片段。CP 主要降低长上下文 attention/activation 显存压力，但不会把模型权重本身变小。

代价是 attention 需要跨 CP rank 通信，尤其 KV/梯度需要 all-gather、reduce-scatter 或 ring 通信。长上下文训练里 CP 往往是必要的。

### EP: Expert Parallel

Qwen3.6-35B-A3B 是 MoE 模型。MoE 不是每个 token 都走全部参数，而是由 router 把 token 分发给部分 expert，所以“总参数量 35B”和“每 token 激活参数 A3B”不同。

EP 把 MoE expert 权重分布到多张 GPU 上。`EP=8` 表示 expert 维度跨 8 张 GPU 切分。好处是每张 GPU 不用放所有 expert 权重；代价是 token 要通过 all-to-all 发到对应 expert 所在 GPU，通信和负载均衡变得很重要。

### ETP: Expert Tensor Parallel

ETP 是 expert 内部的 tensor parallel。`ETP=1` 表示每个 expert 内部不再切矩阵。这通常更简单，也避免 expert MLP GEMM 上额外 TP 通信。

### 它们如何同时存在

一张 GPU rank 会同时属于多个通信组：TP 组、PP 组、CP 组、EP 组、DP 组。dense attention/非 MoE 层主要受 TP/PP/CP 影响；MoE sparse MLP 层额外受 EP/ETP 影响。

需要注意：不同 Megatron 版本对 CP、EP、DP 的组构造细节不同。NVIDIA 文档也把 EP 描述为可以和 TP/CP/PP/DP/FSDP 等组合，但具体 rank folding 由实现决定。因此你上线前要在日志里找初始化打印的：

- tensor model parallel size
- pipeline model parallel size
- context parallel size
- expert model parallel size
- data parallel size
- world size

不要只用“几个数字相乘”在脑子里推断是否可行。尤其你这里单节点 8 卡、CP=4、EP=8，看起来组合很激进，必须以 Megatron 初始化日志为准。

## 12. Optimizer 和 RNG 是什么

`optimizer` 是优化器。你脚本中是：

```bash
--optimizer adam
--adam-beta1 0.9
--adam-beta2 0.98
--weight-decay 0.1
--use-distributed-optimizer
--use-precision-aware-optimizer
```

Adam 不只保存模型参数，还保存一阶动量 `m`、二阶动量 `v`，很多混合精度训练还保存 FP32 master weights。它们非常占显存/磁盘。分布式优化器会把 optimizer state 在 data parallel rank 之间切分，降低每卡 optimizer 状态内存。

`RNG` 是 random number generator state，随机数状态。训练里的随机性来自：

- 数据 shuffle
- dropout
- MoE router 中可能的随机策略
- kernel 中可能的随机或非确定行为

保存 RNG 的意义是：中断后 resume，可以尽量复现“如果不中断，下一步会看到什么数据/随机 mask”。不保存 RNG，继续训练通常还能训，但不是 bit-level 精确续训。

你的 YAML 写：

```yaml
save_optimizer: 'false'
save_rng: '0'
load_optim: '0'
load_rng: '0'
```

脚本明确做了：

```bash
--no-save-optim
--no-load-optim
--no-load-rng
```

但我在主脚本里没有看到 `SAVE_RNG` 映射到 `--no-save-rng`。也就是说，“不加载 RNG”是明确生效的；“不保存 RNG”是否生效，要看 `qwen3.5-35B-A3B.sh` 或训练框架是否读取环境变量。Megatron checkpoint 文档里 `save_rng` 默认通常是 true，所以这里建议你在实际 checkpoint 目录里确认是否保存了 RNG state。

不保存 optimizer 的后果：

- checkpoint 小很多。
- 更适合作为最终模型权重或后续新实验起点。
- 不适合严格 resume，因为 Adam 动量会丢。
- 如果中断后从该 checkpoint 继续，loss 可能短暂抖动，因为 optimizer moments 重新开始。

## 13. Checkpoint 保存逻辑

YAML：

```yaml
save_interval: 8
save_path: /mnt/cfs_bj_mt/models/experiments/lhs_train_qw36
```

脚本：

```bash
--save "${SAVE_PATH}"
--save-interval "${SAVE_INTERVAL:-100}"
```

含义：每 8 个 optimizer step 保存一次 checkpoint。若总共 130 step，理论上会在 8、16、24、...、128 保存。第 130 步是否额外保存 final checkpoint，要看训练框架是否有“退出前保存”逻辑，不能默认认为一定有。

Megatron distributed checkpoint 不是一个单文件，而是一个目录树，里面有元数据和各 rank 分片。拷贝/迁移时要整个目录一起拷。

常见 checkpoint 排查：

- 目录为空：任务没跑到第一个 save interval，或 `SAVE_PATH` 没权限。
- 只有部分文件：保存时任务被杀，checkpoint 可能损坏。
- resume 报 optimizer key 缺失：你用了 `--no-save-optim` 保存，却没加 `--no-load-optim` 读取；这个脚本已经处理了。
- shape mismatch：模型结构、mrope、MoE expert、TP/PP/EP 或 tokenizer/config 不一致。
- 续训 loss 抖动：optimizer/RNG 没保存，属于预期风险。

## 14. 学习率、warmup、decay

YAML：

```yaml
max_lr: '2.0e-05'
min_lr: '2.0e-06'
lr_warmup_fraction: '0.03'
lr_decay_iters: 115
train_iters: 130
```

脚本：

```bash
--lr "${LR}"
--lr-decay-style cosine
--min-lr "${MIN_LR}"
--lr-warmup-fraction "${LR_WARMUP_FRACTION}"
--lr-decay-iters "${LR_DECAY_ITERS}"
```

warmup 是训练最开始不要立刻用最大 LR，而是在前若干 step 逐步升到 `max_lr`。原因是初始阶段 optimizer 动量还没稳定，数据 batch/packing 也可能波动，直接大 LR 容易 loss spike 或 NaN。

Megatron 系列里 warmup 通常是线性 warmup。粗略公式：

```text
warmup_steps ≈ lr_warmup_fraction * lr_decay_iters
lr(step) ≈ max_lr * step / warmup_steps
```

你的配置：

```text
115 * 0.03 = 3.45
```

所以 warmup 约 3 到 4 个 step。如果按 `train_iters=130` 算，也是约 4 个 step。之后进入 cosine decay：

```text
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

`lr_decay_iters=115` 小于 `train_iters=130`，意味着大约前 115 step 完成 cosine 衰减，后面可能维持在 `min_lr=2e-6` 附近再训 15 step。这可能是有意让最后阶段低 LR 收敛。

但是注意：主脚本里没有直接搜到 `TRAIN_ITERS` 被传给 `train_async.py`。它可能在 sourced 的 `MODEL_ARGS` 里，也可能由 `num_epoch` 控制停止。如果日志最终命令没有 `--train-iters 130`，那 `train_iters: 130` 可能只是平台层或模型脚本层参数，不一定在主脚本直接生效。

## 15. `lr_decay_iters` 和 `train_iters` 如何事先算

因为你启用了 `global_batch_tokens`，应该按 token 算：

```text
total_train_tokens = sum(tokenized_length_after_chat_template_and_truncation) * num_epoch
train_iters = ceil(total_train_tokens / global_batch_tokens)
```

你的数字可反推：

```text
global_batch_tokens = 8,388,608
train_iters = 130
总训练 token 预算 ≈ 1,090,519,040
num_epoch = 3
单 epoch token 量 ≈ 363,506,347
```

`lr_decay_iters=115` 对应：

```text
115 * 8,388,608 ≈ 964,689,920 tokens
```

也就是说 LR 在约 0.965B token 处衰减到下界，剩余约 0.126B token 低学习率训练。

准确计算必须用同一个 tokenizer、同一个 chat template、同一个 loss mask 和截断策略预处理数据。只用文件大小或字符数估是不可靠的。

## 16. FP8 为什么能省显存，省在哪里

YAML：

```yaml
use_fp8: 'true'
fp8_format: e4m3
fp8_recipe: mxfp8
```

脚本：

```bash
--fp8-format e4m3
--fp8-recipe mxfp8
```

FP8 把部分训练计算中的 activation/weight 临时表示从 BF16/FP16 的 16 bit 降到 8 bit。它主要节省：

- forward/backward 中保存的部分 activation 或中间 tensor。
- GEMM 输入输出相关的临时 buffer。
- 某些通信带宽。
- 某些 kernel workspace 或 cache。

它通常不会把所有东西都变成 8 bit。训练的主权重、optimizer state、master weights、Adam moments 往往仍然是 BF16/FP32 或框架自己的 precision-aware 格式。所以 FP8 不是“显存直接减半”，而是对 activation-bound、GEMM-bound、长上下文场景非常有帮助。

`e4m3` 表示 4 位 exponent、3 位 mantissa，精度相对好但动态范围有限。`mxfp8` 是 microscaling 风格的 FP8 recipe，用分块 scale 降低溢出/下溢风险。

代价：

- 数值误差变大。
- 可能出现 loss spike、NaN/Inf。
- 对硬件、TransformerEngine、kernel 版本要求高。
- 最终质量可能略差，需要和 BF16 baseline 对比。

排查建议：

- 第一次跑最好有一组 BF16/FP8 off 的小步数对照。
- 观察 `train/loss` 是否突然 NaN。
- 观察是否有 FP8 amax/scale overflow 相关日志。
- 如果一开 FP8 就异常，先关 FP8，不要同时调 LR、batch、数据，避免混淆变量。

## 17. Recompute 是什么

YAML：

```yaml
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: '1'
recompute_loss_function: 'true'
```

脚本在 `full` 时传：

```bash
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--recompute-loss-function
```

正常反向传播需要保存 forward 的中间激活。长上下文下，activation 显存非常大。recompute/checkpointing 的思路是：forward 时少存一些中间结果，backward 时需要时再重新算一遍。

好处：省显存。

代价：增加计算量，训练更慢。

`full` 最省显存；`selective` 只重算部分模块，通常更快但省得少；`none` 最快但最吃显存。

你这里 `full + CP=4 + FP8` 是偏保守的长上下文显存方案。若显存很空、吞吐低，可以尝试 `selective`；若 OOM，保持 `full`。

## 18. `global_batch_tokens` 下为什么脚本特别传 `lr_decay_iters`

脚本注释写得很关键：token batch 下真实 step 数应该按总 token / 每步 token 预算计算。如果框架退回按样本数估算，cosine LR 可能过早衰减到 `min_lr`，后面大量 step 都在低 LR “空转”。

所以这段很重要：

```bash
if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
  OPTIMIZER_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi
```

你的 YAML 设置了 `lr_decay_iters: 115`，平台若正确注入成 `LR_DECAY_ITERS=115`，这个参数就会生效。

## 19. 日志会写到哪里

脚本把训练日志和资源采样写到：

```text
/mnt/cfs_bj_mt/workspace/guru4elephant/shared/jobstate/<SAVE_PATH最后一段>/
```

对你的 `save_path=/mnt/cfs_bj_mt/models/experiments/lhs_train_qw36`，目录名大概率是：

```text
/mnt/cfs_bj_mt/workspace/guru4elephant/shared/jobstate/lhs_train_qw36/
```

里面有：

- `train.log`：rank0 训练 stdout 全量日志。
- `resources.jsonl`：每 20 秒一次 GPU 利用率、显存、功耗。
- `status.json`：最近一次解析出的 step、tgs、tflops、loss、rollout_time、GPU 状态。

脚本会 grep 这些指标：

```text
step
perf/train_tgs
perf/actor_train_tflops
train/loss
perf/rollout_time
GPU utilization / memory / power
```

WandB 在 YAML 里是：

```yaml
use_wandb: 'false'
wandb_mode: disabled
```

所以默认不会上报 WandB。`tensorboard_path` 在 YAML 里有，但主脚本没有直接搜到传参位置，可能在 sourced 模型脚本或框架里消费，需看最终命令/日志确认。

## 20. 正常训练应该长什么样

正常现象：

- Ray head 正常启动。
- 单机时 `NNODES=1`，`wait_for_all_nodes` 会直接通过；多机时日志显示所有节点 alive。
- transformers import ok。
- AutoConfig/AutoTokenizer preload 成功。
- load checkpoint 成功，无 missing/unexpected/shape mismatch。
- 第 0/1 step 前可能编译很久，尤其 Triton/FlashAttention/TE kernel 首次运行。
- 过了编译期后 GPU 利用率上升，功耗稳定。
- `train/loss` 是有限数字，不是 NaN/Inf。
- `perf/train_tgs` 和 `actor_train_tflops` 在前几步后趋于稳定。
- 每 8 step 在 `save_path` 出现 checkpoint。

loss 是否“应该单调下降”不要太教条。SFT 大模型、变长 packing、token mask、数据混合会让 step loss 有噪声。更重要的是：

- 不 NaN。
- 没有持续爆炸。
- 移动平均缓慢下降或稳定。
- 和 BF16/历史同类任务相比量级合理。

## 21. 异常判断和排查

### 读不到数据

现象：

- `TRAIN_DATA ... does not exist`
- rollout 为空
- loss 为 0 或 NaN
- 有效 token 数极少

检查：

- `DATA_PATH` 是否正确注入。
- JSONL 每行是否合法 JSON。
- 每行是否有 `messages` 字段。
- `messages` 是否符合 Qwen chat template 需要的 role/content 格式。
- assistant 内容是否为空。

### checkpoint 加载失败

现象：

- shape mismatch
- missing key / unexpected key
- distributed checkpoint metadata 错误
- mrope/rope 相关参数不一致

检查：

- `MODEL_PATH` 和 `LOAD_PATH` 是否对应同一个模型版本。
- `position_embedding_type=mrope` 和 `mrope_section=11 11 10` 是否与转换一致。
- MoE expert 数、router 参数是否一致。
- checkpoint 转换时的 TP/PP/EP 与本次训练是否被框架支持重分片。
- `REF_LOAD` 和 `LOAD_PATH` 是否都存在且目录完整。

### OOM

优先尝试：

1. 降低 `max_tokens_per_gpu`，比如 32768 -> 28672 或 24576。
2. 保持/开启 `recompute_granularity=full`。
3. 确认 `micro_batch_size=1` 真生效。
4. 降低 `global_batch_tokens`，减少每 step 总 token。
5. 增大 CP，例如 CP=8，本地 token 从 32768 降到 16384。
6. 开启 optimizer CPU offload，但这会变慢；B200 192GB 场景一般不优先。
7. 检查是否保存 optimizer 或额外 reference actor 占显存。

### loss NaN 或爆炸

优先尝试：

1. 关闭 FP8 做 BF16 对照。
2. 降低 `max_lr`，例如 `2e-5 -> 1e-5`。
3. 增加 warmup step。
4. 检查数据里是否有超长、乱码、异常空样本。
5. 检查 loss mask 是否错误地训练了 user/system 或 padding。
6. 检查是否有 gradient clipping 参数；主脚本没直接看到，可能在 `MODEL_ARGS`。

### loss 不动

可能原因：

- input_key 错，读到的数据不是 messages。
- loss mask 把 assistant token 全 mask 掉。
- 学习率过早衰减到 `min_lr`。
- 实际没有加载到要训练的参数。
- 数据重复或质量太差。
- 只看 step loss 噪声太大，需要看移动平均。

### 吞吐低

可能原因：

- 首步编译/autotune，等几步再判断。
- packing 浪费严重，padding 到 4096 导致有效 token 比例低。
- `max_tokens_per_gpu` 太低。
- CP/EP 通信瓶颈。
- checkpoint 保存太频繁，`save_interval=8` 会有明显 I/O 周期。
- 数据读取/tokenize 成瓶颈。
- Ray worker 没全部起来。

### Ray/NCCL 网络问题

脚本已经打印很多网络诊断：

- `MASTER_ADDR`
- `MASTER_PORT`
- `NODE_IP`
- `NCCL_SOCKET_IFNAME`
- `ray status`
- TCP probe

如果多机卡住，先看：

- 所有 worker 是否 join。
- `MASTER_POD_IP` 是否正确。
- `no_proxy` 是否包含 master/node IP。
- NCCL 网卡是否选错。
- 端口 6379/8265 是否被挡。

## 22. 这份配置里值得特别注意的点

1. 脚本开头 echo 仍写着 `PP=2 + CP=8`，但 YAML 实际是 `PP=1 + CP=4`。以最终环境变量和日志为准。

2. `max_tokens_per_gpu=32768` 与 `seq_len=131072, CP=4` 匹配，这是合理的。

3. `load_path` 名字里是 `tp2_pp1_cp1_ep8`，训练是 `tp1_pp1_cp4_ep8`。如果 checkpoint 格式支持重分片，这是可以的；否则会加载失败。必须看 load 日志确认。

4. `save_optimizer=false` 明确会变成 `--no-save-optim`。但 `save_rng=0` 在主脚本里没有看到明确变成 `--no-save-rng`，建议确认。

5. `train_iters=130` 在主脚本里没有直接看到传参，可能在 sourced `MODEL_ARGS` 或框架层。建议看最终命令确认。

6. `tensorboard_path` 在主脚本里没有直接看到传参。若你期待 TensorBoard 曲线，要确认日志中是否真的初始化了 TensorBoard writer。

7. `save_interval=8` 对 130 step 的任务来说保存很频繁。优点是中间产物多、方便回滚；缺点是 I/O 开销明显，尤其大模型分布式 checkpoint。

8. `use_fp8=true + recompute full + CP=4` 是偏省显存配置；如果稳定性有问题，第一优先关闭 FP8 做对照。

## 23. 新人跑这类任务的检查清单

开跑前：

- 确认 JSONL 能抽样解析。
- 确认每行有 `messages`。
- 用同一 tokenizer 统计 token 数，估算 `train_iters`。
- 确认 `MODEL_PATH` 是 HF 原始模型。
- 确认 `LOAD_PATH` 是 Megatron distributed checkpoint。
- 确认 `REF_LOAD` 和 `LOAD_PATH` 首训时一致。
- 确认 checkpoint 转换支持 mrope 和 MoE。
- 确认最终日志里的 TP/PP/CP/EP/DP 与预期一致。

开跑后前 5 step：

- 看 Ray 是否正常。
- 看 checkpoint load 是否 clean。
- 看首步是否只是编译慢，不要过早杀。
- 看 GPU 显存是否接近上限。
- 看 loss 是否有限。
- 看实际 LR 是否按 warmup/cosine 变化。
- 看有效 token 数、tgs、tflops 是否合理。

跑完后：

- 检查 `save_path` 是否有 step 128 或 final checkpoint。
- 检查是否保存了 optimizer/RNG，确认它是否符合你的 resume 预期。
- 抽取最终 checkpoint 做一次加载或小推理 smoke test。
- 对比 base model 和 SFT model 在几条固定 prompt 上的输出差异。

## 参考资料

- NVIDIA Megatron Bridge Parallelisms Guide: https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html
- NVIDIA Megatron Bridge Checkpointing: https://docs.nvidia.com/nemo/megatron-bridge/0.3.0/training/checkpointing.html
- NVIDIA Megatron Bridge HF/Megatron conversion technical details: https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-tech-details.html
