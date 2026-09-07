# H-STDN（混合时空脉冲神经网络）需求设计

## 项目定位
H-STDN v3.2-final 的工程化 Python 实现。以设计文档 v3.2-final（唯一权威规范）为准绳，从零搭建 `hstdn/` 科研代码库，目标是通过 G0–G4 分级验证关卡，最终在合成 10 类、MNIST、DVS-Gesture 上验证 STDP 储备池 + 读出的端到端精度。

## 权威规范
- 唯一权威文档：H-STDN 详细设计文档 v3.2-final（含 D1–D14 设计裁决、B1–B11 勘误、§2 数据布局、§3 模块设计 M1–M6、§4 超参、§9 实施计划）。
- 四份源文档（kimi/ds/智谱/千问）已降级/归档，禁止作为实现依据。

## 首期范围（当前里程碑：D1–D4 + G0 十五项断言全绿）
按 §9 实施计划阶段推进：
1. **D1** layout.py + network.py + spatial_hash.py（平板布线 z∈[0,0.3]、per-source k=12、CSC 修正构建、增益标定、自检报告）→ G0 #4/#10/#12/#13
2. **D2** kernel.py L0（时间轮、严格不应期、输入通道 STDP、同刻经典顺序、状态重置契约）→ G0 #1/#2/#3/#6/#7/#14/#15
3. **D3** encoder.py + data/synthetic.py + 诊断面板（latency 修复排序、MNIST 10×10 池化映射）
4. **D4** plasticity.py（input_channel_stdp + post_ltp + pre_ltd + 逐神经元 homeo + 竞争归一化）→ G0 #5/#8/#9

后续阶段（不在本期）：D5 scheduler/readout/features → G1 首跑；D6 MNIST G2；D7+ L1 Numba / DVS G3 / 规模化 G4。

## 环境
Python 3.12.10，Windows 本地；依赖 numpy>=1.24, scipy>=1.10, numba>=0.59（Py3.12 兼容微调）, pyyaml, matplotlib；torch/tonic/snntorch 按 G2/G3 需要延迟引入。

## 工程纪律
- 三件套纪律：patch + changelog + G0 断言。
- ID 契约：存储层全局 ID / 状态数组层池局部 ID，转换仅在内核投递处。
- L1 Numba 分支禁止 pass 占位（未实现须显式抛 NotImplementedError）。
- G1 前冻结全部超参，禁止调参破坏归因基线。
