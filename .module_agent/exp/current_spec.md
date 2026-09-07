实现 hstdn/configs/default.yaml（§4 超参，唯一契约）与 hstdn/exp/gates.py（G0 十五项断言脚本，首期必须全绿）、diagnostics.py（诊断面板含新增诊断量）。ablation.py 与 G1+ 门禁后续阶段补充。
## 配置与超参契约

§4 超参唯一契约 `hstdn/configs/default.yaml`（exp 模块交付，G1 前冻结）：network{n_in=100,n_pool=800,e_ratio=0.8,extent=[1,1,0.3],radius_in=0.4,radius_pool=0.18,velocity=0.05,k_in=16,k_pool=12,max_delay=15}、lif{tau_mem=20,theta0=1,refractory=2,T=200,dt=1}、stdp{eta_ltp=0.008,eta_ltd=0.004,tau_trace=20,w_min=0.02,w_max=1.5}、homeo{target_rate_hz=8,eta_homeo=0.05,ema_alpha=0.1}，另有 protocol/readout_l0/features/readout_l1/encoder 节（D5/G1+ 消费）。加载辅助 `hstdn/configs/__init__.py`：load_config（pyyaml→dict+路径解析+顶层结构校验）、to_core_cfg（network/lif/stdp/homeo→core NetConfig 的唯一桥，含 √n_in→n_input_cols、extent[2]→pool_z_hi、T→window_s、dt=1 校验，未知键显式报错防漂移）。core 尚未直接读取 yaml，default.yaml 契约经由 exp 桥进入 gates/G1；core 接管后桥降级为纯校验。
## G0 十五项门禁

G0 十五项断言门禁 `hstdn/exp/gates.py` 为项目门槛权威实现（首期已交付并 15/15 全绿，exit 0），全绿方可进入 G1；每项独立可执行：`python -m hstdn.exp.gates`（批量）/ `... gates 3 7 11`（单项）/ `--list`（清单）；退出码 0/1/2。十五项编号与语义：①LIF 解析解对照（误差<2%）②传导延迟 d/v<1ms ③时间轮同槽累加 ④CSR/CSC 回指一致 ⑤STDP 方向性 ⑥输入通道活性 ⑦同刻对净更新≈+eta_ltp（D12）⑧homeostasis 闭环收敛（连续带判据，θ 平衡非饱和）⑨归一化守恒 Σw=in_sum0 ⑩ID 规范回归 ⑪内核性能 ⑫度分布 ⑬延迟上界 max_delay<L ⑭不应期严格封锁 ⑮状态重置回归（B11）。夹具纪律：micro 确定性网络由显式边表按 §2.2/§2.3 布局自建（CSR/CSC/B3/in_learn 与 core 同构），不依赖 core 内部私有函数；§4 规模网络由 default.yaml 经 to_core_cfg 构建一次后按门禁克隆复用。G0 后新增数据结构/不变量须在此登记或补对应断言。ablation.py 与 G1+ 门禁为后续阶段交付。
## 诊断面板

诊断面板 `hstdn/exp/diagnostics.py`（首期已交付，供 G1 与 gates 调试）：纯函数诊断量——发放率/沉默比例（firing_rates_hz/silence_ratio/rate_summary）、类间余弦（cos_similarity/class_cosine_matrix）、sqrt+L2 特征可比性（feature_transform(mode='sqrt_l2')/feature_cos_similarity，支持列掩码，对应 configs features 节 v0/feature_mask=pool 的读出预处理）；bundle 级（core 惰性导入）：顶界突触占比/输入→池 w 均值/可塑性面板（委托 core plasticity 原语）、延迟直方图、网络结构面板、样本级 rate+塑性汇总面板与打印助手。职责边界：只读数不承载业务断言；core 接口变更仅需同步内联调用点。
## G1 消融与首跑门禁

G1 随机储备池消融与首跑门禁（R1，exp 模块 G1 配套；文档 §6 G1、§8 ablation、§11 冻结条件）：

- `hstdn/exp/ablation.py`：实验组（stdp/homeo/norm 三开关开，run_g1_protocol ADAPT 适应）vs 对照组（三开关全关随机固定储备池 + CALIBRATE + READOUT，经典 LSM）。两组唯一差异 = 三开关（对照组以 n_adapt_epochs=0 等价表达全关），同数据同种子。提供 Spec/LIGHT_SPEC(4类双种子池200 机制)/FULL_SPEC(10类≥3种子池150 官方判据)、run_ablation（多种子 mean±std + 组间对比 + evaluate_full_criterion：实验组均值≥60% 且显著高于对照=均值差≥5pp 且逐配种子 exp>ctrl）、CLI。规模注明：L0 内核 800 池逐样本数百 ms，默认消融跑门禁规模（--pool 可回 800）。
- gates.py `run_g1`（G1_SPECS id=101 登记）：--g1 轻量（快速机制验证 + 打印组间对比与提示，不断言官方判据）；--g1 --full 完整官方判据断言 PASS/FAIL；退出码/清单协议与 G0 语义一致。
- diagnostics.py 新增 G1 评估纯函数：multi_seed_summary / compare_groups / format_g1_ablation。

实证现状（诚实记录，本机 2026-09 测量）：10 类池150 noise0.16 三种子 run_g1_protocol 下 exp=56.7±15.9% vs ctrl(随机LSM)=90.3±4.9%（--full 判据未满足，FAIL）；4 类池200 noise0.18 双种子 exp=45.8±23.6 vs ctrl=64.6±26.5。即当前 core scheduler + V0 计数特征 + 本合成任务族（互不重叠字形掩码）下，冻结校准储备池已近饱和（0.87-0.98），STDP+homeo+norm 适应后部分池 θ 饱和/高沉默（exp 静默 12-51%）反而削弱计数可分离性。G1 完整判据要转绿需后续研究：时序敏感特征（v1 分箱喂入 COLLECT）、homeo/θ 饱和治理、或更利于 STDP 的任务族 —— 属 core/scheduler 与数据侧范畴（exp 不越权）。
## exp-g 判定

exp-g 判定（冻结池内可塑性 → Diehl-Cook 结构；归因 Wave 1b / §3 M4，H6 相关）：

- default.yaml network 节新增 `k_pool_learn: true`（默认 True=现状；False 触发 exp-g 冻结池内 E→E 可塑性，仅输入→池可学习、池间连接固定）。G1 冻结纪律下默认保持 True；注释说明该键为 exp-g 判定/Diehl-Cook 回归开关。
- configs/__init__.py `_NET_MAP` 增加 `k_pool_learn -> NetConfig.pool_learn`（core 字段 pool_learn，默认 True；False 时 core network 的 learn = input | (E-source & pool_learn)）。
- ablation.py：Spec 新增 `pool_learn: bool = True`（含 to_dict/默认规格处理），`_net_cfg` build_network 前 replace 透传；SeedResult 新增 calib_ok 与 w_ratio（E 源池→池权重均值 末/初 比，冻结=1，结构+STDP 冒烟验证通过）；exp-g 入口 `run_expg_comparison(light/full)`（唯一变更 pool_learn=False 且协议三开关全开=exp 协议），判据 evaluate_expg：silence<5%、CALIBRATE 全种子 ok、w_ratio≈1（1e-6）、acc≥80%；CLI `--expg`/`--freeze-pool`（--full 组合）。

实测结果（本机 2026-09，如实记录）：FULL（10 类 noise0.16 池150 adapt2 3 种子）expg=89.7±4.2%、silence 0.0%、calib 3/3 ok、w_ratio dev=0 → **healthy=True（四项全过，≥80% 达成）**；对照 exp（池内可塑性开）=56.7±15.9%（silence 34%）、ctrl(随机LSM)=90.3±4.9%（silence 0%）。light（4 类池200 双种子）：expg=72.9±20.6（silence 0.8%、calib 2/2、w_ratio dev 0，仅 acc<80% 故 light 判据 False——判据按 FULL 规模定义）。解读：冻结池内 E→E 可塑性消除了 exp 组的 θ 饱和/高沉默（34%→0%），输入→池（Diehl-Cook 前馈）路径可学习且稳定，性能≈LSM（89.7 vs 90.3，±4-5% 内不显著）；该结果将上一轮 G1 exp<ctrl 的退化归因指向池内可塑性动态，支持 H6 方向的定案依据，exp-g 是否额外超越 LSM 需更难任务族区分。
## v3.3 冻结清单

v3.3 冻结清单 exp 侧（#3 诊断升级 / #4 R7 触发器替换+度量拆分 / #5 stdp_min 档 / #9 卫生）：

- diagnostics（纯函数、core 惰性导入、无断言）：选择性指数 gini_within_neuron/per_neuron_gini（within-neuron Gini 均值；G3 终审机制级前哨）、gini_ratio（快照后/初比）；率分布 P50/P95（rate_percentiles）、有效维度（非沉默且未过热 rate<3×target）、双峰系数、rate_distribution_panel（均值率降级 mean_rate_hz_ref 参考项）；漂移/劫持度量拆分 capture_drift_snapshot/read_drift_stats（drift_frac、input_drift_frac、gini、gini_ratio、gini_std=正确 per-neuron std、w_ratio/dev、input_w_min_sat、budget_dev≈0）；R7 assess_r7（输入 w_min 饱和>20% 或 gini_ratio>1.3）。卫生：无 std_ratio 死计算残留。
- configs：stdp_min.yaml（#5，STDP 条件唯一合法档：pool_learn=False + §4 全字段显式钉死，与 default 同构仅 k_pool_learn 不同）+ PROFILES/profile_path；default.yaml 默认 k_pool_learn: true（§4 现状），stdp_min 供 R1 runner STDP 臂。
- ablation：对接 v3.3 scheduler（adapt_enabled=False=LSM 臂无 ADAPT 门禁）；Spec pool_learn 默认 False（FIX-A）；run_one_seed 自动 drift/gini/R7/budget 记账（SeedResult 扩展含 aborted 标记）；expg 三臂比较（legacy True / 冻结 False / LSM）；判据与 G1 官方判据照旧。
实测（本机 v3.3 core，如实记录）：ctrl(LSM) full 90.3±4.9 健康；expg full seed0/1=91.0%/85.0% 健康、seed2 ADAPT HARD STOP（abort 前 drift 0.448、gini_ratio 1.173 未触 R7）→ expg 判据 2/3 未全绿；exp(legacy True) 3/3 ABORT（不稳定，与上一轮结论一致）；light expg 68.8±14.7 / ctrl 64.6±26.5（budget_dev 0、ctrl drift 0、gini_ratio 1.000）。G0 十五项默认路径 15/15 exit 0 无回归；新诊断量冒烟合理。
## G2 双臂 runner

G2 双臂 runner（v3.3 冻结清单 #7，R1 第一判合法性前提）：

- `hstdn/exp/g2_runner.py`：LSM 臂（对照组：adapt_enabled=False 无学习随机固定池 + CALIBRATE+READOUT，main 默认 LSM 语义）vs STDP 臂（stdp_min.yaml 档 pool_learn=False + ADAPT 三开关开）；两臂同 seed 同网络 cfg 同数据切分、共享 run_g1_protocol（唯一差异 ADAPT 开关组）。强制同 seed 配对：正式 5 seeds（G1 n=3 在 ±4.2pp 下无功效 → G2 Step 1 升 5）、mini 2 seeds。
- 预注册判据（不事后移动）：LSM 绝对门禁 acc≥70%（R1_LSM_GATE_MIN）；R1 第一判 = 配对差 mean(diff)>0 且 >1×std(diff)（R1_DIFF_STD_FACTOR=1）；处理有效性迁移规则（评审 §2）：STDP ADAPT 后复用 exp.diagnostics.read_drift_stats——mean drift_frac≥5% → valid、判据照常；<5% → 'inconclusive-treatment' + 一次性 η 校准（G2_ETA_CALIB_FACTOR=2 仅一次，防再调滑坡）重跑。
- 数据：G2 全量要求真实 MNIST 1k 子集（data.mnist.load_mnist_subset，torch+torchvision 延迟依赖 + data/mnist 落盘）；mini/冒烟 torch 缺失时回退合成 MNIST 布局等价帧（10x10 单元=N_IN100、10 类、noise0.18）并如实标注 data_source。
- gates.py 注册 run_g2（G2_SPECS id=102）：--g2 mini（机制验证 + 输出配对差与判据字段）/-g2 --full（正式 R1 判据，需真实 MNIST）。

实测（本机 mini 冒烟，如实记录）：LSM 77.0%（gate PASS）vs STDP 79.0%，配对差 +2.0±0.0pp（paired 2/2），treatment=valid（drift 6.8% ≥5% 无需校准），R1 第一判在 mini（std=0）按规则 NOT MET（需 5 seeds 方差）→ exit 0 机制 PASS。G0 十五项 15/15 exit 0 无回归。低噪声合成体制（0.06）下 STDP 臂在 v3.3 严格 ADAPT 门禁 HARD STOP（如实记录为合成回退已知特征；已用 0.18 体制）。
