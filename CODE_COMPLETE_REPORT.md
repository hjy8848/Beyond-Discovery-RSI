# CODE_COMPLETE_REPORT — Beyond-Discovery RSI

日期：2026-10-23。开发基础：M1 `7acaae1`，PR #1 尚未合并。
本报告说明 **声明能力范围内的 M1–M7 工程实现与实际验证**，不宣称任意领域/任意代码的
通用发现算法已经得到验证。真实模型、神经训练和外部 benchmark 的未验证项单独列出。
H1–H4 全部仍待科学实验。运行摘要的 `COMPLETE` 仅表示本次执行结束。

## 锁定规范与提交边界

正式规范、Decision Lock、独立 `core/gate.py` 和 `core/archive.py` 相对 M1 **无修改**。
SHA-256：

- Core Design Spec：`75e970331baa00c56dcc619bbc84ae131c7e3bda51277d286c7299a075d8259f`
- Decision Lock：`1a51e0b447bc7cd07e13056698409ce9f5e5b2931a70ebac52fe2eb2d07c96ac`

| 阶段 | 独立提交 | 行为与入口 | 阶段实际测试 |
|---|---|---|---|
| M1 | `7acaae1` | `protocol.py` / `conformance.py`：通用 schema、状态、动作/观察、产物、协商；原有限算法保留为基线 | 41 项 |
| M2 | `a23b3df` | `boundary.certify_boundary/verify_certificate`：完备有限声明空间的反例证伪、搜索失败、资源未知；诊断只消费证书 | 45 项 |
| M3 | `12dc493` | `synthesis.ArtifactInventor.invent`、`experiments.design_experiment`：类型化代码重组、类比、约束放松、重构及多步区分实验 | 52 项 |
| M4 | `1df260c` | `worlds.ChallengerInventor/WorldCertifier`：基于冻结假设、边界与 train 证据生成世界，独立语义、去重与认证 | 56 项 |
| M5 | `21b3a09` | `meta.synthesize_meta`、`SynthesisRuntime.adapt`、Controller：双方可执行元策略、固定机会独立 Gate、diff 和回滚 | 62 项 |
| M6 | `d5e880c` | `models`、`trainer.LoRABackend`、`ModelResearchRuntime`：真实 HTTP/HF、Token 预留、角色独立训练及 checkpoint、主循环接入 | 68 项；真实神经路径未运行 |
| M7 | 本报告所在提交 | 统一 CLI、11 种对照/消融、双域闭环、spawn 沙盒、成本补齐、有界可辨识证明、证据包、统一审查 | 最终 79 项，全部通过 |

各阶段详情和局限见 `reports/M1.md`–`reports/M7.md`。M7 对 M6 做了角色奖励、精确采样 Token、
固定权重对照模板、实际元代码排序和失败日志的补强；对 M4 补充了独立可辨识证明及预算协商。
没有通过修改 Gate 阈值、评分尺度或可信答案来促成通过。

## M1–M7 能力核对

**M1。** Controller 消费 TaskRef、Claim、Action、Observation、Certification、Evidence、Snapshot。
不要求 finite input_space、width、expression_ops 或标量整数观察。结构化 JSON 与多步会话有
行为/失败测试。可执行语言必须真实协商；没有执行器时显式拒绝。Document 的人工样例始终只算协议样例。

**M2。** 完整 HypothesisFamily 全部被已审核反例排除才是 SPACE_REFUTED；存在另一一致解释是
SEARCH_FAILURE；枚举不完整/证明预算不足是 UNDETERMINED。证书绑定证据、家族、作用域、
逐成员反例与验证版本，独立复算拒绝伪造。有限家族证伪不等于所有可能解释空间失败。
多来源失败、共同假设和证伪证书联合触发表征重构，去诊断对照显式绕过该条件。

**M3。** 发明输出真实 AST 或实验协议，包含可执行代码与来源链。类型保持类比映射使用审核
通过的 train 资产；重构产生可复用的派生特征，放松扩大导出空间，重组构造新表达式。
实验搜索实际执行竞争假设的多步预测，选择能区分它们的动作程序；不是改名或预写正确答案。
运行时相同代码处理状态整数机制和字符串机制，Adapter 没有隐藏诊断/发明器。

**M4。** World proposal 绑定当前 Scientist、假设源程序和 train 证据。Track A 限注册原语/常量
及长度；Track B 允许更广的有界表达式组合。独立树语义与候选 VM 对照，检查可执行、非平凡、
区分性、重复、资源及可辨识性。非法返回 INVALID，预算/证据不足返回 PENDING；均不进入有效奖励。

`identification.identify_world` 额外产生可复算的比较家族哈希、区分实验、剩余行为类与成本。
证明范围为完整的小型注册语法（max_nodes≤3）加提案，行为等价按长度≤2 的公开探针定义。
在本轮协商的动作预算内无法缩至一个行为类则 PENDING。**没有证明无限假设空间或任意外部科学世界
可辨识。** 成功认证单元测试从旧的 100 次验证额度改为 10000，并新增低预算 PENDING 测试，
以覆盖新增独立证明的实际成本；没有放宽认证条件。

**M5。** MetaPolicy 是对 errors/complexity/cost/novelty/coverage 执行的类型化程序。
搜索后端与模型后端均实际运行元代码改变候选排序。元代码文本不同不自动产生增益。
固定父程序、固定权重/对手、固定 train 适应机会和分配，独立比较生成方法的验证成功率。
保存 proposed_meta、source_diff、meta_gate、meta_rollback；失败恢复后重评最终可继承版本。
联合四组合审核采用预注册 rollback-both。任务、元能力和交互项分别记录。

**M6。** HTTP 客户端调用真实兼容 Chat Completions 的服务；本地客户端调用实际 HF generate，
不做无凭据降级或自动模型替换。原始请求/响应和用量留档，无 Authorization 头。
共享 TokenLedger 执行前预留，未知网络结果保守计费。远端需显式提供包含供应商提示框架的输入 Token 上界。

LoRABackend 使用真实 torch autograd/AdamW/PEFT：SFT 或 clipped importance policy gradient，
不是 mock，也不是 PPO。RL 绑定实际行为策略身份、精确采样 Token 和 log probability；固定温度 1、
无 top-k/top-p 筛选，使用独立 GenerationConfig；RL 计算禁用 dropout。检查有限梯度和实际参数变化。
每次 fit 重新装载共享基座，角色 LoRA、优化器、经验与目录隔离。
Scientist 训练奖励来自独立解决质量；Challenger 来自经认证、去重的边界/实验成本，而非 Scientist 高分。

训练只使用已验证 TRAIN 输入及独立 TRAIN 执行奖励。更新 checkpoint 后先装载并生成可执行候选，
再走原 Gate。Program-only、LoRA-only、Hybrid 分开配置；LoRA-only 固定生成模板/提示/采样种子，
在相同机会比较训练前后编译结果，使用额外固定程序权重对照，而非与未适应的父版混算收益。
无有效奖励时不训练；缺少资源时明确失败，不制造收据。Checkpoint 校验角色、基座、数据哈希和 safetensors 校验和。

**M7。** 统一行为一致性套件覆盖五个内置 Adapter；两个状态型领域使用相同 Core、Gate、
研究运行时和指标定义。CLI 支持完整 run、conformance、suite、replay。
对照包含 fixed、fixed_challenger、随机搜索池 Challenger、冻结模型分歧优先 Challenger、
一般程序进化、原表示重组 reflection，以及四项消融。它们是明确定义的本地算法对照，
不是冒充外部论文实现；正式强基线仍需外部实验。相同上限/任务机会，报告实际成本，不填充虚假计算。

## 24 项原则对应

| Lock | 工程落实 / 科学限制 |
|---|---|
| Q1 | 可执行解释 AST 与实验协议统一版本化 |
| Q2 | 两种状态机制独立从零运行相同 Core；外部跨领域结论待实验 |
| Q3 | 程序进化与实际可选权重训练后端 |
| Q4 | 任务 artifact 与独立 meta adaptation Gate |
| Q5 | Challenger 生成新世界源程序，非只抽固定任务 |
| Q6 | 主奖励为证书支持的边界/成本；hardness 是明确的对照 |
| Q7 | 同一 base_model，角色独立状态、LoRA 与优化器 |
| Q8 | 多来源失败、共同假设、独立证伪联合诊断 |
| Q9 | 有类型、可执行、独立评分的 ResearchArtifact |
| Q10 | 五类证据驱动操作及实际模型代码生成 |
| Q11 | 多步竞争预测与可执行区分实验 |
| Q12 | 任务方法和双方元生成策略分别更新/归因 |
| Q13 | 通用协议，领域替换语义/工具而不改 Core；未知语言拒绝 |
| Q14 | 注册组合 A / 独立认证的有界开放代码 B；任意 Python 不支持 |
| Q15 | Adapter 原语执行可编程协议 |
| Q16 | 独立 Score/audit/proof/provenance/原始调用 |
| Q17 | 分区、spawn 受限 worker、资源上限、不可变 Archive；宿主可信扩展限制见下 |
| Q18 | 旧失败上的行为变化 + 独立 dev 有效性；不按字符串新颖度计分 |
| Q19 | 有效边界去重/成本；INVALID/PENDING 无有效挑战信用 |
| Q20 | 固定对手配对 + 四组合联合审核 |
| Q21 | 双候选 dev 前屏障、三态 Gate、无证据保留父版 |
| Q22 | 实际元代码、独立可选 SFT/RL、源 diff/checkpoint/rollback |
| Q23 | 推理/训练/实验/搜索/验证预算、停滞、代数停止、冻结后测试 |
| Q24 | 对照/消融执行器与两域工程闭环；论文级外部证据仍未取得 |

## 实际执行记录

- 最终完整回归：`PYTHONPATH=src python3 -m unittest discover -s tests -v`，**79/79 通过，55.073 秒**。
- 模型/训练边界：7 项通过；仅 localhost 传输及明确标注的模型协议 fixture，绝非真实 LLM 成绩。
- 状态领域 conformance：两域各 13 项通过；既有 causal/vector/document 的测试保留。
- 安全行为：畸形代码、额外字面量参数、越界值、host import、worker 死亡、有界值/载荷/执行 fuel、
  checkpoint 路径/身份/校验和、分区越界、模型截断与预算失败均有拒绝测试。
- `compileall`、`git diff --check`、离线 wheel 构建与隔离目录安装 smoke 通过。
- 两域各两代完整演示、独立 Gate、回滚、冻结测试与 Replay，详见 `evidence/demo-report.json`。
- 两域各 11 个独立方法的一代工程对照；每个方法新 Adapter、新封存运行。完整摘要和日志在 evidence 包。
- CI 配置提供 Python 3.9/3.11 Linux 测试；远端 CI 是否通过以 GitHub Checks 为准，本机只实际运行 Python 3.9.6/macOS。

两代演示：

| 领域 | Replay | 认证 | 任务 Gate | 元 Gate / 回滚 | 冻结最终分数 |
|---|---|---|---|---|---|
| stateful-scm | 211 episodes，226 records | VERIFIED、INVALID | 6 次 INCONCLUSIVE | 6 次 INCONCLUSIVE，6 次回滚 | 0.5329861111 |
| string-transducer | 136 episodes，151 records | VERIFIED、INVALID | 1 ACCEPT，5 INCONCLUSIVE | 2 ACCEPT、4 INCONCLUSIVE，4 次回滚 | 0.3819444444 |

这些小样本接受只证明 Gate 被实际执行，不证明稳定收益、元能力增长或 H1–H4。
在并行开发负载下，含 Replay 的演示分别约 25.65s / 12.62s；不是受控性能 benchmark。

成本定义：experiments 是执行/反例复现；predictions 包括生成时实际顶层 VM 预测、独立评分和重复审核；
search 在 episode 计实际枚举数量、在生成阶段保守计声明分配（包含公共排除池），缓存命中仍可能计费，
不是 CPU 指令数；verification_outputs 是可信语义输出。model_tokens 为推理 Token；训练 Token、steps、
设备和耗时单独在训练收据记录。未估算未发生的 GPU FLOPs 或美元费用。

## 复现与审计

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/demo.py --out runs/my-demo
PYTHONPATH=src python3 -m beyond_discovery.run suite --domain stateful-scm --generations 1 --out runs/my-scm-controls
PYTHONPATH=src python3 -m beyond_discovery.run suite --domain string-transducer --generations 1 --out runs/my-string-controls
```

`evidence/local-evidence-20261023.tar.gz` 保存演示和两域对照的完整 hash-chain 日志/对象、
最终测试输出、构建/安装日志。解压后可在本提交源码上 Replay `demo/stateful-scm` 或 `demo/string-transducer`。
Replay 核对实现哈希，实际重执行确定性认证/交互；不请求模型、不重新训练，不把 API 留档当逐 Token 再现。
原始审计记录含 dev/test，只供审核，不进入 Agent 提示、经验或训练。

真实模型、SFT/RL 的具体命令和依赖见 README。当前环境没有模型凭据、HF 权重、torch/transformers/peft；
没有触发真实付费调用、没有下载权重、没有产出或声称真实神经训练结果。

## 未验证与技术限制

1. **真实推理、神经训练、PEFT checkpoint 实际装载、GPU 执行和收益均未验证。** 代码贯通及拒绝路径已测；
   部署前必须用真实资源执行。HTTP 无本地 adapter 上传能力，不能用 HTTP 推理声称 LoRA 更新。
2. **没有外部 CausalGame / KernelBench 成绩。** KernelBench 明确要求独立 GPU worker、参考和计时协议；
   安装扩展 Adapter 的 entry point 支持真实接入，缺少实现/依赖/资产时 CAPABILITY_UNSUPPORTED，不模拟结果。
3. **开放认证与发明受声明语言/搜索预算限制。** 有限范围反例、行为新颖性和可辨识证明不外推到无限程序空间。
4. **神经后端是可信宿主扩展。** 有实际训练步数、推理 Token 和调用超时限制；没有训练 FLOPs/硬 GPU 时间配额。
   研究 AST 才进入 spawn worker。不能把这称为任意 Python/GPU 代码的全容器隔离。
5. **macOS 的内存限制是 20ms RSS 监控。** 可在采样间隔内超调；Linux worker 使用 RLIMIT_AS。
   worker CPU/总时限/单次调用/有界载荷均预注册，默认 60s/300s/2s/256KiB。
6. **对照共用预算上限，实际使用量不同。** 没有多 seed、等 GPU 小时、统计功效或外部强基线实验结果。
   Reflection 是明确定义的本地原表示重组对照，不能据此宣称胜过通用 LLM Reflection。
7. 非确定性 Adapter 的精确 Replay 不支持；历史日志需相同实现。真实 checkpoint Replay 还需保留对应文件和路径。
8. 科学错误由独立评分反映，资源未知/非法代码/截断/世界非法分别记录；不存在通用自然科学真值判定器。

**H1–H4：全部未验证。** 本次交付的程序通过工程测试不等于研究主张成立。
