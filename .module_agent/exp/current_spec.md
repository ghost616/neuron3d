实现 hstdn/configs/default.yaml（§4 超参，唯一契约）与 hstdn/exp/gates.py（G0 十五项断言脚本，首期必须全绿）、diagnostics.py（诊断面板含新增诊断量）。ablation.py 与 G1+ 门禁后续阶段补充。
## 配置与超参契约

§4 超参唯一契约 `hstdn/configs/default.yaml`（exp 模块交付，G1 前冻结）：network{n_in=100,n_pool=800,e_ratio=0.8,extent=[1,1,0.3],radius_in=0.4,radius_pool=0.18,velocity=0.05,k_in=16,k_pool=12,max_delay=15}、lif{tau_mem=20,theta0=1,refractory=2,T=200,dt=1}、stdp{eta_ltp=0.008,eta_ltd=0.004,tau_trace=20,w_min=0.02,w_max=1.5}、homeo{target_rate_hz=8,eta_homeo=0.05,ema_alpha=0.1}，另有 protocol/readout_l0/features/readout_l1/encoder 节（D5/G1+ 消费）。加载辅助 `hstdn/configs/__init__.py`：load_config（pyyaml→dict+路径解析+顶层结构校验）、to_core_cfg（network/lif/stdp/homeo→core NetConfig 的唯一桥，含 √n_in→n_input_cols、extent[2]→pool_z_hi、T→window_s、dt=1 校验，未知键显式报错防漂移）。core 尚未直接读取 yaml，default.yaml 契约经由 exp 桥进入 gates/G1；core 接管后桥降级为纯校验。
## G0 十五项门禁

G0 十五项断言门禁 `hstdn/exp/gates.py` 为项目门槛权威实现（首期已交付并 15/15 全绿，exit 0），全绿方可进入 G1；每项独立可执行：`python -m hstdn.exp.gates`（批量）/ `... gates 3 7 11`（单项）/ `--list`（清单）；退出码 0/1/2。十五项编号与语义：①LIF 解析解对照（误差<2%）②传导延迟 d/v<1ms ③时间轮同槽累加 ④CSR/CSC 回指一致 ⑤STDP 方向性 ⑥输入通道活性 ⑦同刻对净更新≈+eta_ltp（D12）⑧homeostasis 闭环收敛（连续带判据，θ 平衡非饱和）⑨归一化守恒 Σw=in_sum0 ⑩ID 规范回归 ⑪内核性能 ⑫度分布 ⑬延迟上界 max_delay<L ⑭不应期严格封锁 ⑮状态重置回归（B11）。夹具纪律：micro 确定性网络由显式边表按 §2.2/§2.3 布局自建（CSR/CSC/B3/in_learn 与 core 同构），不依赖 core 内部私有函数；§4 规模网络由 default.yaml 经 to_core_cfg 构建一次后按门禁克隆复用。G0 后新增数据结构/不变量须在此登记或补对应断言。ablation.py 与 G1+ 门禁为后续阶段交付。
## 诊断面板

诊断面板 `hstdn/exp/diagnostics.py`（首期已交付，供 G1 与 gates 调试）：纯函数诊断量——发放率/沉默比例（firing_rates_hz/silence_ratio/rate_summary）、类间余弦（cos_similarity/class_cosine_matrix）、sqrt+L2 特征可比性（feature_transform(mode='sqrt_l2')/feature_cos_similarity，支持列掩码，对应 configs features 节 v0/feature_mask=pool 的读出预处理）；bundle 级（core 惰性导入）：顶界突触占比/输入→池 w 均值/可塑性面板（委托 core plasticity 原语）、延迟直方图、网络结构面板、样本级 rate+塑性汇总面板与打印助手。职责边界：只读数不承载业务断言；core 接口变更仅需同步内联调用点。
