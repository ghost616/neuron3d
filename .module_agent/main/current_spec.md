实现 hstdn/train.py 与 hstdn/eval.py 入口骨架，先打通 config → network 构建 → 单样本 run_sample 链路（支撑 G0 调试），G1 完整协议调度后续阶段补充。
## 训练与评估入口编排

hstdn/train.py 与 hstdn/eval.py（main 模块，包根公共入口）为训练/评估 CLI 编排骨架，只做调度不做算法实现（算法一律调 hstdn.core / hstdn.exp 接口），D1-D4 首期用途是打通 config→build_network→run_sample 链路、支撑 G0 调试与单样本冒烟：

- train.py：`load_config` + `to_core_cfg` → `build_network`（固定 seed rng）→ data.synthetic 合成 10 类批次（每 epoch seed 派生，可复现）逐帧 `latency_encode` → `run_sample`（--stdp/--homeo/--norm 开关）→ 率/沉默/可塑性诊断面板 → `save_checkpoint` 落盘 npz（hstdn-network-bundle-v1：bundle 全部数组字段 + `__meta__`/`__cfg__` 0-d JSON 记录；meta 含 config 指纹 sha256、run 参数指纹、config_json、run_args）。公开接口：config_fingerprint / save_checkpoint / load_checkpoint / sample_aggregate_panel / run_training / main。`--selfcheck` 校验指纹确定性与编解码往返。运行：`python -m hstdn.train --epochs 1 --samples 10`。
- eval.py：`load_checkpoint` 重建 NetworkBundle 并做形状闸；显式 `--config` 与 checkpoint 配置指纹比对不一致即失败；独立测试集（test_seed = 1000003 + seed）全 off 冻结 run_sample；输出率/沉默聚合与类均值发放特征两两余弦（class_mean_features + exp.diagnostics.class_cosine_matrix）。运行：`python -m hstdn.eval --checkpoint <npz> --samples 10`。
- M6 调度器接入点（预留，未实现）：default.yaml protocol 节（adapt_epochs/calibrate_*/extra_loops）由未来 core.scheduler 消费，接入注释在 train._training_epochs；scheduler 落地前不宣称协议调度已实现。
- 复现与纪律：超参唯一来源 default.yaml（运行参数不落入 yaml）；固定种子驱动 build 与数据；checkpoint 载入与形状均有 G0 类断言（含统计数值的可读 AssertionError）。
