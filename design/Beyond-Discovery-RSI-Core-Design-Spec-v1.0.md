# Beyond-Discovery RSI — Core Design Specification v1.0

**副标题：** Recursive Self-Improvement through the Invention of Scientific Experiments and Explanatory Concepts  
**状态：** 设计规范定稿（24 项用户决策已锁定）；**不是已实现或已验证的方法**。  
**日期：** 2026-10-22  
**范围：** 通用 RSI Core、Challenger/Scientist、Domain Adapter、训练与评测协议。**不规定特定领域最优策略**。

> 规范用词：**[LOCK]** 为五轮已确定的原则，不得在实现时静默改变；**[DEFAULT]** 为本规范提出的可调整实现默认值，改变必须登记版本；**[HYPOTHESIS]** 为尚待实验检验的科学主张。

## 0. 24 项决策锁定表

| 决策 | 已锁定内容 | 编号 |
|---|---|---|
| 研究对象 | 实验发明与解释概念发明统一 | Q1=C |
| 通用性 | 同一 Core 在 ≥2 领域分别独立进化 | Q2=A |
| 学习方式 | 允许权重训练和程序/策略改进 | Q3=B |
| Scientist | 任务研究能力与发明能力共同进化 | Q4=B |
| Challenger | 可以提出全新可执行世界 | Q5=C |
| Challenger 目标 | 优先发现认知边界，而非失败率 | Q6=B |
| 模型组织 | 相同基础模型、角色独立进化状态 | Q7=B |
| 定势检测 | 停滞、共同假设、反例联合诊断 | Q8=C |
| 新表征 | 必须可执行或结构化且可验证 | Q9=B |
| 创造操作 | 重组、类比、约束放松、重构等机制切换 | Q10=B |
| 挑战方法 | 区分性实验与主动证伪 | Q11=B |
| RSI 层级 | 任务方法与创造方法的元策略均进化 | Q12=B |
| 抽象 | 通用交互协议，不要求统一世界语法 | Q13=B |
| 世界创造 | 可信组合与开放式生成双轨 | Q14=C |
| Scientist 实验 | 可编程协议，使用领域提供的原语 | Q15=B |
| 反馈 | 结构化、独立、可追溯证据 | Q16=B |
| 隔离 | 沙盒、独立验证、数据分区、不可变证据 | Q17=B |
| 发现认可 | 行为新颖性 + 独立验证的有效性 | Q18=B |
| Challenger 奖励 | 有效认知边界、区分性、去重与成本 | Q19=B |
| 归因 | 固定对手的配对对照，定期交叉配对 | Q20=B |
| 选代 | 双方独立候选、证据驱动三态选择 | Q21=B |
| 元进化 | 可执行元策略；独立 LoRA/RL 可选执行 | Q22=B |
| 停止 | 预算上限 + 停滞、退化、创新率等证据 | Q23=B |
| 论文证据 | 至少两个差异大的领域、相同 Core、强消融 | Q24=B |

**原则冲突结论：** 无直接矛盾；需明确四项边界：① 开放世界不等于能自行认证；② 结构化证据需划分 Agent 可见与审计私有；③ Core 通用不意味着领域表征相同；④ 允许更新 LoRA/RL 不等于允许改写独立验证/选代规则。下文把这四项变成强制协议。

## 1. 科研问题、对象与可证伪假设

**总问题：** 在可信执行反馈下，Agent 能否主动检测当前实验/解释框架的盲区，提出此前自身搜索未覆盖的新表征与实验程序，使发现能力在未见任务上提高，并递归改进产生这种创新的方法？

**核心不是：** 最大化出题难度、生成更多提示词、累计 Skill 数量，或在自建世界上拿高分。

**[HYPOTHESIS]**
- **H1（突破）**：相同预算下，显式“认知盲区诊断 → 表征/实验重构 → 证伪”的机制，产生更多在旧版本失败任务上有效、可复用的新方法，优于普通程序进化/Reflection。
- **H2（共进化）**：针对边界产生区分性实验的可进化 Challenger，优于随机/困难优先/固定 Challenger，并给 Scientist 带来独立开发集收益。
- **H3（元能力）**：元策略在多代中进化后，以相同适应预算发现有效新方法的概率或效率提升；仅任务策略增益不足以证明 H3。
- **H4（跨领域）**：同一 Core（代码、奖励定义、选择规则不作领域性更换）在至少两个差异明显的领域分别从零运行并出现增益；**不要求一个领域的参数零样本迁移到另一个领域**。

“新发现”的层级：
1. **结构新**：与历史代码/表示不同；仅作候选，不计科学成功。
2. **行为新**：在诊断输入上产生旧候选没有的实验/预测/优化行为。
3. **有效新**：通过独立正确性检查，并在旧方法失败的受控实例上取得改善。
4. **可迁移新**：在独立开发任务族上有稳健收益，最终在冻结测试集上复核。

所谓“先前没有发现”，默认**相对于系统冻结的候选历史和预定义基线库**；不可直接声称人类史上从未发现。表征能力的“不可达”通常是**在给定旧表示语言、有限资源和预注册测试集合下的相对边界**，不是对任意 LLM 的不可能性证明。

## 2. 形式化定义

设领域为 `d`，领域适配器 `A_d` 提供任务集合 `T_d`、基础动作/实验原语 `O_d`、执行观测、合法性证明与可信结果指标。

第 `t` 代两个 Agent 的不可变快照：

- `S_t = (B, φ^S_t, M^S_t, R_t, E_t, K_t)`：Scientist；`B` 为共享基础模型；`φ` 为角色独立权重增量（如 LoRA，可为空）；`M` 为可执行元策略（诊断器、表示生成器、实验设计器、修订器）；`R` 为可执行表示/假设规格；`E` 为实验与预测/求解程序；`K` 为带证据的经验资产。
- `C_t = (B, φ^C_t, M^C_t, W_t, K^C_t)`：Challenger；`M^C` 为边界诊断、区分性实验设计、世界提案、去重与停止策略；`W` 为合法生成器/提案策略，不拥有修改真实 Oracle 的权限。

在任务 `τ` 中：

`h_old = S_t.hypothesize(τ, history)`；`c = C_t.propose(τ, h_old, evidence)`；`cert = A_d.certify(c)`；`obs = A_d.execute(cert, S_t.experiment)`；`h_new = S_t.restructure(obs, h_old)`；`e* = S_t.discriminate(h_old, h_new)`；`score = A_d.evaluate(h_new, e*, unseen-within-task)`。

`A_d` 必须能够对核心提出的候选实验进行至少一种**独立、领域可信**的有效性判断；不能把模型自评当作正确性。

认知边界的通用证据对象：

`b = (assumption, witness, source, old_prediction, observed_outcome, alternative_prediction?, validity, reproducibility, cost)`。

`assumption` 是 Agent 提出的可证伪假设，不被视为模型真实内部认知状态的证据。`witness` 是独立执行的冲突或区分性案例。若领域允许枚举/SAT/SMT，可附上“旧受限表示族无法拟合，新表示族可拟合”的形式化证书；否则只标记为实证边界。

优化目标以约束先于加权收益的方式定义：

- `valid(c)`: 独立认证通过，信息/答案未泄漏，任务可执行，非重复欺骗。
- `U_C(c | S_t)`: 在固定 Scientist 下，单位成本发现**经复现的、新颖有效的区分性边界**的效用。
- `U_S(S' | C_t)`: 固定任务分布/Challenger 下，在独立开发任务上相对于 `S_t` 的性能提升，兼顾资源成本与回归。
- `U_meta(M' | M)`: 固定发明机会集合和适应预算下，元策略产生**有效新方法**的增量收益；它不同于单次任务分数。

推荐的可记录指标（非普适固定系数）：

`boundary_yield = #unique_verified_discriminative_boundaries / total_experiment_cost`  
`invention_yield = #new_valid_transferable_methods / total_adaptation_cost`  
`Δtask = mean_grouped_score(S') - mean_grouped_score(S)`  
`Δmeta = invention_yield(M') - invention_yield(M)`。

固定 `A_d` 将领域原始分数归一为“越大越好”并标注约束、方差与物理意义；**不得跨不同领域直接平均原始分数作为算法通用性证据**。

## 3. 五种核心信息对象与版本化

所有执行均有 `content_hash / parent_hash / domain_id / split / origin / generation / agent_version / seed / verifier_version / usage`，历史不可覆盖。

1. **Claim / Hypothesis**：可证伪的预测或约束；可执行预测器或严格可检查的声明，包含适用范围和不确定性。
2. **RepresentationPatch**：旧/新表示结构、变换理由、候选程序、运行前静态检查；要求能够计算行为差异，而非只比文字。
3. **ExperimentProgram**：可调用领域原语的受限程序，包含预算、输入和观测输出。
4. **ChallengeProposal**：由任务条件、世界创建程序或对照实验组成，附预期区分假设；不携带可信评分权。
5. **EvidenceBundle**：Agent 可见观测、私有 Oracle/认证记录、预测对照、复现、预算、失效类型、因果对照 ID、审核哈希。

**[DEFAULT]** 每代最多提出固定数量的 Candidate；多个成功分支可存档以避免过早损失多样性，但只有通过选代 Gate 的版本进入正式主链。权重检查点、Prompt、工具代码、策略和元策略均在快照内明确记录。

## 4. Core 的具体机制（固定程序骨架，可进化内部策略）

### 4.1 Fixation Diagnosis（双信号）

- **性能信号：** 在结构去重的多个有效任务上连续停滞、回归或重复失败。
- **证据冲突信号：** 多个失败候选共享可陈述的假设 `a`，且存在真实观测 `w` 与 `a` 的可执行预测冲突。
- **决策：** `Diagnoser(M^S_t)` 输出 `continue_local_search / propose_restructure / insufficient_evidence`，附观测指针和置信度；不允许仅凭一条自然语言“我陷入定势”触发奖励。

### 4.2 Representation Restructuring（创造实验与概念）

通用算子类别（算子本身的实现策略允许进化）：
- `RECOMBINE`：对已验证方法的可复用子结构组合。
- `ANALOGIZE`：搜索跨任务结构映射，明确源/目标对应关系和失配条件。
- `RELAX_ASSUMPTION`：放松可验证的非必要限制，记录能力与成本变化。
- `REFRAME`：更换变量、状态、关系、分解或候选生成语法。
- `INVENT_EXPERIMENT`：编写新的受限实验协议，组合领域已有基础操作。

生成 `(R', E', M_candidate?)`，必须能够与 `(R,E)` 在一组固定诊断案例上比较可执行行为；避免把纯粹重命名算作创新。

### 4.3 Active Falsification（Challenger）

优先寻找一组条件使 `h_old` 与 `h_new` 的预测可区分，或使旧表示中的共同假设产生实测冲突；如果还没有 `h_new`，允许与 `h_alternative`（明确标为候选、未证实）比较。目标是最大化**有效辨识证据/成本**，而非制造随机难题。

### 4.4 Meta-Evolution（真正递归对象）

不仅更新任务解决程序，还允许候选补丁修改：
- `M^S`: 何时判定定势、怎样提出表示变换、怎样设计实验、怎样筛选解释；
- `M^C`: 怎样发现旧解释的可检验弱点、提出反事实世界、选择区分性干预、去重及停止。

**锁定边界：** 可修改元策略的代码/角色 LoRA，但**不能修改可信认证器、最终指标定义、外层版本选择器、数据分区或安全沙盒**。这样元策略能够演化，外层研究验证规则仍可复现。

## 5. 一代更新算法与伪代码

**[DEFAULT]** 采用“同一父代采样 → 独立产生双边候选 → 固定对手各自选代 → 联合兼容验收”的阶段屏障，不让候选先偷看另一候选产生的私有开发反馈。

```python
# A_d: domain adapter; outer selection/evaluation protocols immutable
C, S = initial_snapshots(shared_base_model)
archive = AppendOnlyArchive()
for t in range(max_generations):
    train_batch = A_d.sample("train", frozen_seed[t])
    episodes = run_real_interactions(C, S, train_batch, A_d)
    evidence = certify_and_record(episodes, A_d, archive)

    boundary_hypotheses = S.meta.diagnose(evidence)
    C_candidate = C.meta.revise(boundary_hypotheses, evidence)
    S_candidate = S.meta.revise(boundary_hypotheses, evidence)
    # Allowed: executable meta-program patches + optional separate LoRA/RL.

    C_next = gate_challenger(parent=C, candidate=C_candidate,
                            frozen_scientist=S, fixed_dev_opportunities=True,
                            certified_only=True, budget_matched=True)
    S_next = gate_scientist(parent=S, candidate=S_candidate,
                           frozen_challenger=C, fixed_dev_tasks=True,
                           separate_test=True, budget_matched=True)

    C_next, S_next = joint_compatibility_gate(
        parent_pair=(C, S), proposed_pair=(C_next, S_next),
        fixed_dev_audit=True, no_weight_updates=True)

    archive.commit_versions_and_all_attempts(C_next, S_next, evidence)
    if stop_by_preregistered_budget_or_evidence(archive):
        break
    C, S = C_next, S_next

# exactly after training/selection stops:
freeze_everything(C, S)
result = evaluate_once_on_sealed_test(C, S, A_d)
```

`gate_*` 输出 **ACCEPT / REJECT / INCONCLUSIVE**：
- `ACCEPT`：全部硬验证通过；在结构独立的开发任务上对主指标产生超出预登记最小有意义增益的正向证据；预算/回归约束满足。
- `REJECT`：代码非法、证明失败、泄漏、质量/性能严重退化或预算违规。
- `INCONCLUSIVE`：样本不足、波动太大、未证明新方法有效或收益不稳定；**默认保留父版**。
- 若两边单独通过而组合后失效，联合 Gate 可保留单侧更新或回滚二者，依据预登记规则选择，不能事后挑分数最高的组合。

## 6. 奖励、信用分配与对照

**先硬约束，再比较效用；不把可疑成功加权成有效结果。**

- **Challenger 主指标**：在冻结 `S_t`、等运行预算、相同合法来源集合上，新发现且经独立复测的区分性边界数/成本；无效世界奖励为 0 且单独统计无效率；如果认证证据不足则标为 `PENDING`，不能当有效世界用于训练。结构重复的世界/反例不重复得分。
- **Scientist 主指标**：在未参与本次候选生成的开发任务上的有效解决率、预测质量、性能或研究质量（领域提供的客观 score），同时检查先前已掌握任务的回归。新表示需要在受控旧失败例上展示行为差异，且有新任务收益，才记为有效新能力。
- **元策略主指标**：在固定的“适应任务包”上，以相同模型/训练/执行预算，比较由 `M_t` 和 `M_{t+1}` 各自产生的新方法的验证成功率或单位成本产出；不得只比较 M 版本差异或 Prompt 新颖度。
- **辅助指标**：总 Token、训练 FLOPs/GPU 小时、执行/沙盒成本、构造/认证失败、实验次数、覆盖多样性、错误复现率、一次适应的调用次数、停止前有效代数。

**归因：**
1. **C 的独立效果**：`C_t` vs `C'`，冻结 `S_t`，比较各自的独立有效边界发现率，不把来源环境不等价当作提升。
2. **S 的独立效果**：`S_t` vs `S'`，冻结 `C_t` 和开发任务/实验预算，比较有效方法迁移。
3. **交互效应**：在预先冻结的共同审核任务池上评估 `(C_t,S_t)、(C',S_t)、(C_t,S')、(C',S')`，估计交互项；不能用 C 自己生成且自己挑选的题目当作唯一审核池。
4. 任务效果和元策略效果**分别报告**，避免把被动 Skill 积累称为元层级 RSI。

统计分析以**独立世界结构/仓库/算子族/任务来源簇**为单位做配对区间或层级重采样，避免把同一世界的 2304 个预测位当作 2304 个独立样本。阈值、可接受退化范围和主指标必须在研究前预注册。

## 7. World Creation 双轨认证

**Track A：Certified Compositional**
- Adapter 维护小型可信语义组件/生成器；Challenger 提出组合/变换，由独立 Compiler/Reference 判合法、可执行、非平凡，并冻结世界/测试。

**Track B：Open Programmed Worlds**
- Challenger 可生成新环境规则、程序、约束及实验接口候选；独立沙盒编译、静态检查、属性测试、对照实现或独立可验证语义检查。
- 如果它只提交“自己的世界 + 自己写的标准答案”，不得因此自动认证；需要可信外部约束、独立参考程序、可证明性质或经可靠的交叉核验支持。
- 无独立可信真值或足够独立的行为判据：标为 `UNVERIFIED`，可归档研究，**不可进入主训练奖励、不能作为成功挑战**。

每个认证世界登记 `world_hash、generator_hash、semantics_version、source_cluster、split、certificate、resource_budget`。Challenger 不可调整 Verifier、偷读测试真值、改变评分尺度。

## 8. Domain Adapter v1.0 契约

领域自行提供具体语义，但**禁止在 Adapter 中实现“诊断定势/重构表示/进化元策略”的算法**；这些必须存在于 Core。可以有领域标准推断器作为基线/Oracle，但不能暗中代替 Scientist。

```python
from typing import Protocol, Any, Literal

Split = Literal["train", "dev", "test"]

class DomainAdapter(Protocol):
    def capabilities(self) -> dict[str, Any]:
        """Typed actions, observation schema, constraints, supported proofs."""

    def sample(self, split: Split, seed: int) -> list[Any]:
        """Return immutable task refs; sealed test never exposed to evolver."""

    def compile_challenge(self, proposal: Any) -> Any:
        """Validate/compile generated challenge; return certification or failure."""

    def begin(self, certified_task: Any, agent_view: Any) -> Any:
        """Start isolated world/session; return only permitted observation."""

    def step(self, session: Any, action: Any) -> Any:
        """Execute an allowed experimental primitive; enforce budgets."""

    def evaluate(self, submission: Any, task_ref: Any, split: Split) -> Any:
        """Private scoring, provenance, uncertainty, cost, validity."""

    def audit(self, record_ref: Any) -> Any:
        """Recompute/verify evidence without consulting the candidate LLM."""
```

Core 消费的是统一的 `TaskRef / Claim / Proposal / Action / Observation / Certification / Score / Evidence / Snapshot`，而不是领域特有的 SQL AST、GPU tensor 或因果图。Adapter 必须给出明确的 **action budget、可信反馈类型、评分方向、评测随机性、失败分类及限制**。若某领域不支持“创造世界”或“可靠评测”，要报 `CAPABILITY_UNSUPPORTED`，**不能悄悄换成只抽题并仍宣称 Q5=C 的完整能力**；可以进行受限适配实验，但必须列为 scope limitation。

## 9. RL / LoRA 与程序 Meta-Evolution 并存

**[LOCK]** 共享基座、角色独立权重增量与策略状态。`φ^C` 和 `φ^S` 的梯度更新数据、checkpoint、经验池、奖励均隔离，不采用默认为共享权重更新的方案。

**[DEFAULT]** 两种合法运行配置：
- **Program-only**：基础模型冻结，元策略代码/实验程序/表征按证据进化；用于机制消融和低成本 Pilot。
- **Program + LoRA/RL**：保持基础 checkpoint 固定，对两角色的独立 LoRA 在各自已验证训练数据上做更新；LoRA 的接受仍通过相同 Gate。需单独报告程序-only、LoRA-only、混合三种增益与资源花费。

如果使用仅提供 API 推理的闭源模型，不能声称执行了权重更新；该运行属于 Program-only。可更新模型权重是**允许的机制**，不要求每轮都更新，更不能绕过冻结测试。

**元策略因果验收默认要求**：固定新的适应机会池，只替换生成新策略的 `M`，对比它在相同预算下产生可验证新方法的能力；必要时将 LoRA 与元策略代码分别控制。防止“更强的底座/更多 Token”假扮 RSI。

## 10. 执行、保密、审计、回滚

- 所有生成代码在容器/受限进程运行，权限最小化，限 CPU/GPU、时间、内存、网络、文件系统与系统调用；不能访问评测器私有脚本、测试密钥、世界生成器隐藏答案。
- 分区：`train` 可用于反思、更新和可见任务反馈；`dev` 仅用于固定协议下的选代/开发诊断，不允许加入训练回放；`test` 封存，**冻结后最终只调用一次**（跨论文多次尝试需透明记录），绝不用于逐代调参。
- 代码非法、世界非法、正确性失败、证据不足、执行超时、模型请求截断、科学推断错误必须分开标注，避免把 G0 代码生成失败解读成能力退化或改善。
- append-only journal 保存完整请求/响应、模型和 LoRA ID、代码 diff、种子、Token、工具/验证成本、proof/cert、策略版本、选择/回滚理由；Replay 尽可能复算所有**确定性**部分，随机模型/API 结果以原始记录审计，不冒充逐 token 再现。
- `ACCEPT` 发布不可变子版；`REJECT/INCONCLUSIVE` 保留父版；预算超限、安全违规或严重污染立即中止，保留全部证据。停止只看 `train/dev` 和预设预算。

## 11. 最小正式论文实验（两个差异领域）

**[DEFAULT] 领域组合：**
- **A：交互式因果发现**：优先调查 CausalGame 是否能在不触及正式测试反馈的情况下支持足够的训练/开发挑战与可靠隐藏机制验证；研究实验发明和解释重构。若其只有少量固定场景，用独立合法 SCM 生成器进行训练，公开 Benchmark 严格作冻结测试。
- **B：GPU Kernel 优化（KernelBench）**：研究新优化表征、可执行实验协议及失败假设证伪；你的 5070 Ti 可做轻量执行，必须显式报告显存、计时和编译限制，且跨硬件数据不能直接比较。

这两个领域是否满足完全同构的发明能力指标尚需接口审查。**同一 Core 的算法、元策略算子类别、证据分类、Gate 和停止规则必须一致；适配器可替换领域语义、工具、原始得分。**

**必要对照：**
1. Base Agent / fixed strategy（无进化）。
2. Reflection-only（等预算的一般代码/Prompt 改进）。
3. Program Evolution（允许开放式自修改，但不专门做认知边界诊断；不能人为禁止其重构表示）。
4. Random/Fixed Challenger + 同一 Scientist 进化。
5. Hardness-first Challenger + 同一 Scientist 进化。
6. Full Beyond-Discovery RSI。
7. 消融：去 Fixation Diagnosis、去 Representation Restructuring、去 Active Falsification、去 Meta-Evolution；权重更新另做 Program-only / LoRA-only / 混合对照。

**预算与评测：** 同底座、同训练预算（模型调用/Token、GPU 小时与验证次数分别核算）、同推理/实验预算、同最终任务集合；独立随机种子；事前冻结指标、分区与统计单位。最终报告各代轨迹，不只选最高代。

**主结果：** 每域的外部测试表现 + 经审核的有效新方法数/成本 + 独立世界上的迁移 + 元策略发明效率 + 消融效应；可附代表性的旧假设、反例、表征 diff 与实际新方法代码。

**论文成功的必要条件：** 两领域均有相对强基线的可信收益，并能显示表现变化与具体可执行的认知重构、反例和元策略更新相关；若只产生更多世界/代码或只靠额外训练计算量，不能声称 H1–H4 成立。

## 12. 预注册淘汰与迭代纪律

- 若普通开放式程序进化在等预算下与 Full 持平，则“不足以证明表征重构模块额外价值”，必须收缩贡献或修改实质机制，而不是加长提示词。
- 若随机或难度优先 Challenger 持平，则“不足以证明认知边界驱动的挑战者必要”。
- 若新方法只在训练世界有效，未见结构/外部测试不增益，则不成立“可迁移发现能力”。
- 若多代 Skill/代码累积有效但固定 Meta-Generator 同样有效，则不成立“创造方法的机制发生递归改进”。
- 若某 Adapter 使用领域特定的隐藏专家策略自动推断答案，结果必须归入 Oracle / strong baseline，不可作为通用 Core 的贡献。

**[DEFAULT] 进入正式扩展实验前先做两域最小端到端闭环**：各自完成一轮 C+S 候选生成、证据认证、独立 Gate、联合检查、回放、冻结测试样例；不以演示成功代替统计有效性。真实 Token、GPU 时间、世界认证失败率与各分支可比性都必须先核实。

## 13. 未来代码模块建议（尚未创建）

```text
beyond_discovery/
  core/
    protocol.py       # domain-neutral schemas & capabilities
    archive.py        # immutable snapshots, provenance and journal
    diagnosis.py      # fixation/boundary analysis framework
    invention.py      # recombine/analogize/relax/reframe operators
    coevolution.py    # parent-candidate barriers and training loop
    gate.py           # paired independent selection & interaction audit
    training.py       # separate LoRA/RL backends; optional
    policy_runtime.py # sandboxed generated meta-programs
  adapters/
    causal/            # world semantics, interventions and trusted oracle
    kernel/            # generated kernels, correctness, GPU measurements
  experiments/
    preregistration/  # frozen design/budgets/splits
    baselines/        # reflection, program evolution, random/hardness
    reports/          # training/dev/test separated reports
```

**最值得先审查的入口：** `core/protocol.py`（通用边界）、`core/coevolution.py`（递归更新）、`core/gate.py`（防止虚假选代）、`core/diagnosis.py` 和 `core/invention.py`（真正的创造力机制）。所有路径目前是规划，不代表已经实现。

## 14. 证据状态与外部相关工作

**当前已实际观察（基于此前实验报告）：** 黑箱 v1 中一次从反馈归纳的策略修改；v2 中生成代码发生变化，但未证明新的隐藏状态发现能力，且 G0 曾因 Token 截断失败。它们是工程/科学风险证据，**不是 Beyond-Discovery RSI 完整机制的验证**。

**已核对的公开相关工作：** Darwin Gödel Machine（自修改 Agent）、PiEvo（原则空间扩张）、CausalGame（交互式因果发现 Benchmark）、KernelBench（可执行 Kernel 评价）。研究必须继续比较主动学习、主动实验设计、CEGIS、程序进化、开放式世界生成等；**当前不能声称新颖性已得到充分证明**。

- Darwin Gödel Machine: https://arxiv.org/abs/2505.22954
- PiEvo (ICML 2026): https://proceedings.mlr.press/v306/pu26a.html
- CausalGame (ICML 2026): https://proceedings.mlr.press/v306/chen26l.html
- KernelBench: https://github.com/ScalingIntelligence/KernelBench

**待验证：** H1–H4、有效世界的独立可认证覆盖率、两个领域的语义适配成本、模型权重进化的成本收益、元策略可迁移性和长期稳定性。

---

### 给实现者的最高优先级约束

**不能把“更难的题目”“更长的反思”“更多的代码版本”冒充新的发现能力。** 任何声称的认知边界突破，必须同时附带：旧假设、真实区分性证据、可执行新表示/实验、独立验证，以及未见任务上可重复的收益。任何声称的元层级 RSI，还必须附带“发现新方法的能力自身改善”的配对证据。
