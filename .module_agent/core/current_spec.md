实现 hstdn/core/ 下 layout.py（ID/契约/断言工具）、network.py（平板布线 M1，per-source k=12、CSC 修正构建、增益标定、自检报告）、spatial_hash.py、encoder.py（M2：latency/DVS/MNIST 池化映射）、kernel.py（M3：L0 时间轮内核，含 ADR-001 输入通道 STDP、D12 经典顺序、R1 严格不应期、B11 状态重置契约）、plasticity.py（M4：input_channel_stdp + post_ltp + pre_ltd + homeo + norm）、features.py（M5 V0/V1）、readout.py（M5 L0/L1/L2）、scheduler.py（M6 状态机）。首期先覆盖 D1-D4 阶段功能。
## 布局契约与超参集中定义

ID 契约（违反即失败）：存储层（CSR/CSC/trace 索引）一律全局 ID（输入 0..N_IN-1、池 N_IN..N_IN+N_POOL-1）；状态数组访问层（V/theta/refr/counts/rate_ema/ring/csc_ptr/in_learn_ptr）一律池局部 ID；全局→局部唯一转换 pool_local_of（dst_local = dst - N_IN）仅在内核投递处。

超参/规格：NetConfig（frozen，§4 默认：N_IN=1024、N_POOL=8000、80/20 E/I、R_IN=0.4、k_in=16、R_POOL=0.15、k_pool=12、VEL=0.02、delay∈[1,15]、θ0=1.0、REFR=2、ETA_LTP=ETA_LTD=0.01、w∈[0.02,1.5]、homeo 目标 8Hz、时间轮 L=16），with_derived 派生 wheel_l/decay_v/decay_t/n_total/n_e_pool；configs/default.yaml 就绪后为其唯一来源。模块级常量 N_IN/N_POOL/N_TOTAL/WHEEL_L/ETA_LTP 等供 G0 #4/#10/#13 直接引用。dtype 规范：状态与权重 float64、ID/指针/计数 int64。断言工具 assert_*（域/形状/dtype）失败一律抛带统计数值的 AssertionError。
## 空间索引

SpatialHash3D：三维均匀网格空间哈希，cell_size=R_POOL；O(N) 构建（单趟 lexsort 分桶）、邻域 O(1) 均摊查询（仅扫描与查询球相交的常数个格子）。query_radius 返回半径内候选（距离升序、并列按索引）；nearest_k 取半径内最近 k 个并支持 exclude_self。确定性：同一构建序列结果稳定。供 M1 布线（输入→池 R_IN、池→池 R_POOL per-source）与 G0 #10（暴力对照）使用。
## 网络构建（M1）

build_network(cfg)->NetworkBundle（M1）：输入神经元 z=0 平面网格+抖动；池均匀采样 [0,1]×[0,1]×[0,0.3] 平板域（D13）；80/20 E/I 标记（D11）；输入→池 per-pool 最近 k=16（R_IN=0.4，3D）；池→池 per-source 最近 k=12（R_POOL，排除自身）；权重：输入 U(0.9,1.3)·θ0（learn）、E 源 U(0.18,0.42)·θ0（learn）、I 源 -1.2·exp(-d/0.12)（learn=False，冻结）；delay=clip(round(dist/VEL),1,15)。CSR 稳定排序（源升序→目的升序）+ csr_src_g；CSC 按目标 B3 修正构建（np.add.at(csc_ptr[1:], csr_dst-N_IN, 1) 后 cumsum，断言 csr_dst≥N_IN）；in_learn_ptr/in_learn_idx/in_sum0 仅收 learn=True 入突触。NetworkBundle：SoA 状态（V/theta/refr/counts/first_spike/rate_ema 池局部）+ 全局 trace + 时间轮 ring(L,N_POOL) + CSR/CSC 全字段。自检报告（structural_report/print_report）：孤立神经元=0、零入度=0、E 源出度≤12、延迟直方图、增益标定（输入 ~1.1θ0、E ~0.3θ0、|I|min≥E mean、rho 仅打印）。
## 模拟内核（M3）

run_sample(bundle, input_buckets, T=200, stdp_on/homeo_on/norm_on=False)（M3 L0 NumPy 时间轮内核）。严格顺序不变量：入口 B11 重置契约（V/refr/counts/first_spike/trace/ring 清零，theta/rate_ema 保持跨样本）；每步 ①V*=decayV ②V+=ring[slot]并清槽 ③输入注入 np.add.at 投递未来槽 + 当步 input_channel_stdp（ADR-001）④发放判定 (V≥θ)&(t>refr) 严格大于（R1），同刻对按经典顺序 per-spike（池局部升序）pre_ltd→置迹(trace[N_IN+spk]=1)→post_ltp（ADR-003/D12）⑤放电投递+V=0+refr=t+2 ⑥迹衰减 *=decayT；样本末按需 homeo_update/competitive_norm。时间轮 L=wheel_l=16（build 断言 max_delay<L）。reset_state 暴露 B11 重置；L1 入口 run_sample_l1 显式 raise NotImplementedError（无 pass 占位）。热路径 Python 循环仅限稀疏局部（每发放/每事件行），行内向量化。
## 输入编码（M2）

latency_encode（M2）：t = 2 + (1-I)·40、仅 I>0.12（严格）发放，按发放时刻排序后分桶（FIX-B8），输出 Dict[int, (ids, strengths)]（ids 为输入全局 ID、strengths=1.0）；MNIST：mnist_adaptive_pool 纯 NumPy 自适应均值池化（torch adaptive_avg_pool2d 语义，floor/ceil 边界）28→10×10=100 单元（D14），mnist_encode 行主序映射到 N_IN=100；DVS：dvs_patch_aggregate（D14，N_IN=512）仅留接口、G3 里程碑前显式 raise NotImplementedError；Poisson 编码默认关闭（cfg.poisson_on=False，由调度器负责）。
## 可塑性（M4）

可塑性（M4，L0）——全部作用在权威权重 csr_w（CSC 经 csc_csr_pos 映射回 CSR 位置）：
- input_channel_stdp：输入通道 pre 侧 LTD 于全部出突触（w*=(1-ETA_LTD·trace[dst])，clip），随后 trace[in_id]=1（ADR-001/D10；同刻池 post_ltp 经此 trace 实现 pre→post 增强）；
- pre_ltd：池 pre 发放 CSR 侧乘性 LTD（w*=(1-ETA_LTD·trace[csr_dst])，仅 learnable，clip）；
- post_ltp：池 post 发放 CSC 侧加性 LTP（csr_w[csr_pos] += ETA_LTP·trace[csc_src]，仅 learnable，clip[0.02,1.5]）；I 源突触冻结（D11）；
- homeo_update：逐神经元（rate=counts/window_s；rate_ema=0.9·ema+0.1·rate；θ=clip(θ+0.05·(ema-8),0.5,20)）；
- competitive_norm：竞争归一化仅 learn 入突触（按池 csr_w[idx]*=in_sum0[p]/s，clip）。
新增诊断量：capped_ratio（顶界突触占比，learnable 中 w≥w_hi 的份额）与 input_w_mean（输入→池 w 均值），plasticity_diagnostics 打包。
## 自检与验证（过渡期）

core 自检：hstdn/core/selfcheck.py（python -m hstdn.core.selfcheck）——exp/gates.py 就绪前的最小断言（逐组标注 G0 编号家族：layout #4/#13；spatial_hash #10；network #12/#13/#10；kernel #1/#2/#3/#6/#7/#14/#15；plasticity #5/#8/#9；encoder M2 冒烟），覆盖：ID/形状断言、哈希暴力对照、默认配置布线不变量与 CSR/CSC/B3/in_learn 一致性、确定性 toy 网络的时序/不应期/时间轮/STDP 顺序微测试、plasticity 原语单元断言、encoder 桶/MNIST 映射、默认配置动态冒烟（确定性重放、stdp/homeo/norm 边界）。G0 十五项断言的最终权威实现仍属 exp/gates.py。
