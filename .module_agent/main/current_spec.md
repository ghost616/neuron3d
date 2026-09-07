实现 hstdn/train.py 与 hstdn/eval.py 入口骨架，先打通 config → network 构建 → 单样本 run_sample 链路（支撑 G0 调试），G1 完整协议调度后续阶段补充。
## 训练与评估入口编排

hstdn/train.py 与 hstdn/eval.py（main 模块，包根公共入口）为训练/评估 CLI 编排骨架，只做调度不做算法实现（算法一律调 hstdn.core / hstdn.exp 接口），D1-D4 首期用途是打通 config→build_network→run_sample 链路、支撑 G0 调试与单样本冒烟：

- train.py：`load_config` + `to_core_cfg` → `build_network`（固定 seed rng）→ data.synthetic 合成 10 类批次（每 epoch seed 派生，可复现）逐帧 `latency_encode` → `run_sample`（--stdp/--homeo/--norm 开关）→ 率/沉默/可塑性诊断面板 → `save_checkpoint` 落盘 npz（hstdn-network-bundle-v1：bundle 全部数组字段 + `__meta__`/`__cfg__` 0-d JSON 记录；meta 含 config 指纹 sha256、run 参数指纹、config_json、run_args）。公开接口：config_fingerprint / save_checkpoint / load_checkpoint / sample_aggregate_panel / run_training / main。`--selfcheck` 校验指纹确定性与编解码往返。运行：`python -m hstdn.train --epochs 1 --samples 10`。
- eval.py：`load_checkpoint` 重建 NetworkBundle 并做形状闸；显式 `--config` 与 checkpoint 配置指纹比对不一致即失败；独立测试集（test_seed = 1000003 + seed）全 off 冻结 run_sample；输出率/沉默聚合与类均值发放特征两两余弦（class_mean_features + exp.diagnostics.class_cosine_matrix）。运行：`python -m hstdn.eval --checkpoint <npz> --samples 10`。
- M6 调度器接入点（预留，未实现）：default.yaml protocol 节（adapt_epochs/calibrate_*/extra_loops）由未来 core.scheduler 消费，接入注释在 train._training_epochs；scheduler 落地前不宣称协议调度已实现。
- 复现与纪律：超参唯一来源 default.yaml（运行参数不落入 yaml）；固定种子驱动 build 与数据；checkpoint 载入与形状均有 G0 类断言（含统计数值的可读 AssertionError）。
G1 协议接入（M6 消费，main 模块 train/eval 扩展；协议超参唯一来源 = default.yaml protocol 节）：

- train.py --g1（run_g1_training）：逐 seed 执行 build_network（seed 确定性）→ core.scheduler.run_g1_protocol（ADAPT→CALIBRATE→COLLECT→READOUT→EVAL，M6 已实现）→ _derive_readout（公开 core API 确定性重放得到读出 W/b 与自洽 acc，因 scheduler 不暴露轮次权重）→ 每 seed 落盘 g1_train_s{seed}_*.npz（hstdn-network-bundle-v1 + extra 数组 W/b；meta 指纹含 seed/协议参数（protocol_kwargs_from_yaml 翻译自 protocol 节）/config 指纹/stage_sequence/rolled_back/best_acc/各阶段读数/derived）。种子与数据派生约定复用 exp/ablation（数据 seed = 1000+/2000+seed；默认 pool 200 = G1 门禁 light 规模，--pool 可调 800）。
- eval.py --g1（run_g1_evaluation）：load_g1_checkpoint → 独立测试集（test_seed = 5000 + 训练 seed + --seed，与协议 EVAL 2000+seed 数据流错开）全 off 冻结推理 → build_features/predict_linear_readout → 测试精度 + 率/沉默/cos，对照打印协议训练 best_acc/stage/rolled_back；报告口径与 exp.gates --g1 / exp.ablation 对齐。
- codec：save_checkpoint 增可选 extra 数组；新增 load_g1_checkpoint/_read_checkpoint；load_checkpoint 保持 (bundle, meta) 签名不变。
- 冒烟：train --g1 --classes 4 --train-samples 60 --test-samples 20 --seeds 1 与 eval --g1 --checkpoint ... --samples 20 均 exit 0（~48s/seed）；两次同参协议结果逐行一致（仅 wall 不同）。