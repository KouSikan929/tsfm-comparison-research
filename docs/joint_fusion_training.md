# 四模型端到端软门控联合训练

入口：`scripts/train_joint_fusion.py`；A40 脚本：`run_joint_fusion.sh`。
这是独立的新训练流程，不读取现有测试预测 CSV，不覆盖旧模型或旧实验结果。

## 本次实现的模型

| 分支 | 初始化 | 联合训练路径 |
|---|---|---|
| Kronos | `NeoQuasar/Kronos-small` + 原 tokenizer | 固定 tokenizer → 原生 embedding / Transformer / norm → 新建20步分位数预测头 |
| Lag-Llama | 官方 `lag-llama.ckpt` | 原生 lag/scaler/time features → 原生 Transformer → 新建20步分位数预测头 |
| TimesFM | `google/timesfm-2.5-200m-transformers` | HF 原生可微预测和分位数输出 |
| iTransformer | 与项目相同的2层、128维架构，随机初始化 | 原生多变量预测 → close 通道 + 新建区间宽度头 |

原生仓库保持不变，适配代码在 `scripts/fusion/experts.py`。
所有参与预测的神经主干、预测头和 gate 在同一个 AdamW 训练循环中更新。
Kronos tokenizer 固定，Kronos 原生离散输出头及 dependency layer、Lag-Llama 原生
分布输出头不参与当前预测路径，因此也固定。不是“包含 tokenizer 的所有权重全量训练”。

Kronos 和 Lag-Llama 是基于预训练表征的新预测专家，不应把这一结果命名为原生
autoregressive Kronos / Lag-Llama 的预测结果。两者的新预测头必须训练，初始输出
不能视为已有 pooled fine-tuning 的性能。本入口默认从公开预训练权重开始，
不会自动加载可能已经接触本次验证期的 pooled checkpoint。

默认还纳入 persistence、drift、20步 moving-average 三个固定专家；gate 共输出7个
连续权重。用 `--no-baselines` 切换为只有4个神经专家的对照实验。

## 信息和损失

输入保持400条观测、预测20条，评估步长10。BTC 的条数是自然日，其余通常是交易日。
输入字段固定为 open/high/low/close/volume/amount；amount 缺失时使用 close×volume。

全部14个现有市场特征进入 gate：realized_vol、rolling_vol_20d、hurst、
efficiency_ratio、max_drawdown、downside_vol、adf_stat、adf_pvalue、kpss_stat、
kpss_pvalue、skewness、kurtosis、max_abs_return、volume_ratio。
这些直接复用现有计算代码（包括其252日年化约定）；只在输入窗口计算。
按训练窗口拟合中位数填补和标准差缩放，并追加14个缺失值掩码。

gate 输入还包括各专家的标准化预测、区间宽度、专家分歧和预测步长，使用64维MLP。
softmax 沿专家维度归一化，为每个窗口的每个预测步产生不同权重。
专家表征通过各自预测头参与联合学习；当前 gate 没有直接拼接隐藏向量。
不使用稳定/中间/不稳定标签作硬路由。历史样本外误差/coverage 状态暂未加入，
避免将现有测试结果或尚未兑现的预测误差混入输入。

主损失：融合点预测的绝对误差 / 输入最后一个 close（每个窗口独立归一化）。
辅助损失：4个神经专家的归一化绝对误差（默认权重0.2）、融合及专家的
q10/q50/q90 pinball loss（默认0.1）、gate 到均匀权重的 KL 约束（默认0.0001）。
这里点预测同时作为 q50 训练目标；TimesFM 保留 HF 实现所提供的点预测通道。
区间端点用同一 gate 加权，并由 pinball 监督；这不是“概率混合分布的精确分位数”，
其覆盖率需要独立评估。固定专家区间使用历史价格差标准差和 sqrt(h) 的启发式宽度。

训练按市场均衡、有放回抽取窗口；验证逐窗口完整计算后按市场等权平均。
默认 FP32、batch=1、梯度累积8、主干学习率1e-5、新预测头与gate学习率1e-4。
不宣称小样本 smoke test 有预测精度意义。

## 时间边界

- Train 目标结束日期不晚于2024-06-30。
- Validation 目标全部位于2024-07-01至2025-06-30。
- Test 目标从2025-07-01开始，训练入口不会自动执行测试集评估。
- 跨边界的目标窗口被排除；验证输入可以引用更早的已观测历史。
- scaler 只拟合 train；所有前向路径都不读取 batch 中的 target。
- Lag-Llama 超过400条上下文所需的早期 lag 用零填充、标为未观测；不额外读取历史。

支持 `--train-end` / `--val-end` 调整较早的训练验证边界，但不能越过最终测试开始日。
这控制的是项目内训练泄漏，不保证公开预训练模型没有接触过相关历史。
原测试期已经参与方法探索，论文最终确认建议另外保留新的时间外测试期。

## A40 环境与启动

建议在 Linux 服务器新建独立 Python 3.10 环境，避免四套旧环境的依赖互相覆盖。
先安装与服务器驱动适配的 CUDA PyTorch 2.6，然后：

```bash
python -m pip install -r requirements-fusion.txt
python -m pip check
```

GluonTS **0.14.4** 是现有 Lag-Llama checkpoint 的依赖版本；新版本移除了 checkpoint
引用的 `gluonts.torch.modules.loss`，不要直接升级。Transformers 5.17.0 是本地验证版本。
GluonTS 要求 pandas<2.2、numpy<2，依赖文件已经限定。

将已有 `data/checkpoints/lag-llama/lag-llama.ckpt` 一起传到服务器。
没有该文件时可先下载官方权重：

```bash
hf download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir data/checkpoints/lag-llama
```

首次正式训练会从 HF 加载/下载 Kronos、tokenizer 和 TimesFM；已有缓存可复用。
支持 `--kronos-id`、`--tokenizer-id`、`--timesfm-id` 指定对应本地模型目录。
`--lag-checkpoint` 指向可信的官方格式 checkpoint（其旧序列化格式需要非 weights-only 读取）。
不支持把 GluonTS predictor 文件夹直接当成 Lag-Llama checkpoint。

先运行小尺寸真实模型类的离线检查：

```bash
python -m pytest tests/test_joint_fusion.py -q
python scripts/train_joint_fusion.py --tiny --device cpu --datasets SP500 BTC --epochs 1 --samples-per-epoch 2 --accumulation 2 --limit-per-market 2 --output data/fusion_runs/tiny_smoke
```

`--tiny` 采用相同原生模型类和适配器，但随机、小尺寸初始化；绝不能用于正式结果。
随后先用完整预训练权重跑一次服务器 smoke：

```bash
bash run_joint_fusion.sh --datasets SP500 --epochs 1 --samples-per-epoch 2 --accumulation 2 --limit-per-market 1 --output data/fusion_runs/a40_pretrained_smoke
```

正式训练全部10市场：

```bash
bash run_joint_fusion.sh --output data/fusion_runs/a40_joint_01
```

正式运行**不要**传 `--tiny` 或 `--limit-per-market`。
当前配置以低单步 batch 控制显存；A40 的实际峰值需在服务器 smoke 时确认。
程序要求新的输出目录，拒绝静默覆盖已有实验。

## 续训、评估和输出

```bash
python scripts/train_joint_fusion.py --device cuda --resume data/fusion_runs/a40_joint_01/last.pt --epochs 20 --output data/fusion_runs/a40_joint_01
python scripts/train_joint_fusion.py --device cuda --evaluate data/fusion_runs/a40_joint_01/best.pt --split test --output data/fusion_runs/a40_joint_01_test
```

续训发生在 epoch 边界，保留 checkpoint 中的训练配置、optimizer 和随机状态。
`--epochs` 是最终总 epoch 数，不是额外 epoch 数。续训时其它训练参数沿用保存值；
可用 `--raw-dir` 指定迁移后的数据位置。评估也使用 checkpoint 中的市场/窗口配置。

输出包括 config.json、data_manifest.json、gradient_audit.json、history.jsonl、
best.pt、last.pt；评估输出逐市场日期/预测步的预测、上下区间和每个专家权重。
验证日志的 `macro_relative_mae` 分母是预测时最后价格，不等同于旧报告
“MAE/目标平均价格”或逐点 MAPE；正式横向表格需对导出的预测使用同一指标口径。

首次训练批次自动检查**仅最终融合点预测损失**对各专家原生主干和 gate 的梯度。
checkpoint 含全部模型状态、结构配置和特征缩放器；加载不依赖原 HF 缓存，
但仍需要项目原生模型源码和上述 Python 依赖。

## 本地验证记录（2026-10-09）

- 使用完整缓存预训练权重、400→20窗口，在 CPU 完成1个训练步和独立验证。
- 仅融合点损失对 Kronos、Lag-Llama、TimesFM、iTransformer 主干和 gate 的梯度均有限且非零。
- 完整 checkpoint 离线回载后，验证指标与保存前完全一致。
- 小尺寸真实原生模型完成双市场训练，以及从 epoch 1 到 epoch 2 的断点续训。
- 测试覆盖权重归一化、主干实际更新、固定 tokenizer、target 不进入前向、时间切分、
  特征无未来依赖、缺失值处理、保存加载逐值一致及已有测试窗口键对齐。
- `pip check` 通过。尚未在 A40 上实测显存和吞吐；本地结果只证明流程可运行。
