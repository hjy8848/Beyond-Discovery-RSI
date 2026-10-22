# Beyond-Discovery RSI

独立、零第三方运行依赖的 Python Package。按照 `design/` 中已锁定的 v1.0 规范实现首个可运行工程版本；原有 `discovery_v2`、`discovery_rsi_pilot` 不作修改。规范原文与 Decision Lock 的 SHA-256 一致。

**当前状态：本地 Program-only 研究框架和协议验证环境。没有完成 KernelBench、CausalGame、真实模型训练或 H1–H4 的科学验证。** 两个内置领域都采用有限输入、确定性语义，不能充当规范要求的两个差异显著的正式论文领域。

## 运行

Python 3.9+，无需联网、密钥或 GPU。在本目录执行：

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m beyond_discovery.run run --domain causal --out runs/causal-001
PYTHONPATH=src python3 -m beyond_discovery.run replay --domain causal --out runs/causal-001
PYTHONPATH=src python3 -m beyond_discovery.run run --domain vector --out runs/vector-001
PYTHONPATH=src python3 -m beyond_discovery.run replay --domain vector --out runs/vector-001
```

每次 `run` 必须使用新的输出目录。现有目录会拒绝执行，防止覆盖证据或重复打开测试集。`replay` 重新执行确定性交互、认证与评分；不调用外部模型、不执行选代，不把随机模型响应的留档称作重新生成。

也可以用本地已有 setuptools 安装：`python3 -m pip install --no-index --no-deps --no-build-isolation -e .`，然后运行 `beyond-discovery run ...`。源码运行方式不需要安装构建工具。

```python
from beyond_discovery import RunConfig, run_rsi
from beyond_discovery.adapters.causal import CausalDiscoveryAdapter

result = run_rsi(CausalDiscoveryAdapter(), "runs/custom-001",
                 RunConfig(generations=3, seed=42))
```

## 已实现的闭环

1. 冻结配置、实现哈希、训练/开发任务及父代快照。Scientist 与 Challenger 使用同一个基座标识，各自持有独立 MetaPolicy、程序、模型增量路径和知识资产。
2. 真实执行观测；受限程序归纳；复现冲突；通过多个来源簇上的失败、共同可陈述假设与实测反例联合诊断表示限制。当前性能信号是多个结构独立任务上的重复失败，不是模型内部心理状态测量。
3. 从同一父代预先生成双方独立候选，持久化屏障后才运行开发集。创造算子在 Core，Adapter 不提供解题器。默认枚举式元策略是公开启发式基线。
4. Scientist：行为差异、受控旧失败改善、固定对手的开发收益；Challenger：固定 Scientist 下，去重且复现的边界产出/实际实验数。无效世界不计成功。
5. 按来源簇配对 bootstrap、候选数量校正、回归和预算约束进行 ACCEPT / REJECT / INCONCLUSIVE 选代。收益不足保留父版。所有候选都留档；正式继承按事先规定的第一个 ACCEPT，避免事后挑选最佳分数。
6. 单独替换 MetaPolicy，在固定父任务程序、固定适应机会与一次算子调用预算下，比较有效新方法产生成功率。任务提升不自动算元提升；元证据不足时只保留任务补丁，恢复父元策略。
7. 四组合交叉配对审计，记录交互项；组合违反兼容约束时依预注册规则双边回滚。
8. 实验、评分、验证、搜索和模型 Token 预算；停滞与代数上限停止；冻结后一次测试。所有被拒绝、无结论、认证失败的尝试均留档。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `core/contracts.py` | 冻结数据结构、角色状态和 Adapter 协议 |
| `core/controller.py` | 代际屏障、固定对手、元策略实验、兼容审计、冻结测试 |
| `core/diagnosis.py` | 多来源失败和可执行冲突的联合诊断 |
| `core/invention.py` / `evolution.py` | 创造算子、独立候选、可执行元策略调度 |
| `core/gate.py` | 硬约束、簇统计和三态选择 |
| `core/archive.py` / `replay.py` | 内容寻址、链式日志、证据复算 |
| `agents/` | 领域无关的受限程序归纳和主动区分实验 |
| `runtime/policy_runtime.py` | 无主机访问能力的有界表达式 VM |
| `runtime/llm.py` | 可注入模型客户端、严格 JSON 候选解析与原始调用记录 |
| `core/training.py` | 角色隔离的训练后端协议与 checkpoint/数据来源检查 |

## Adapter 与执行边界

`CausalDiscoveryAdapter`：合成无隐藏状态的确定性因果机制，对六个 Boolean 输入作干预，观测一个输出；独立穷举预测正确率。

`VectorProgramAdapter`：整数向量函数归纳，CPU 穷举验证加法、乘法和 max 表达式。报告正确率与指令数；没有 GPU 加速比。

生成世界支持可信组合 Track A 和开放的**有界 DSL** Track B。世界作者只提交程序，不提交可信答案。独立树解释器与栈解释器在完整有限输入域逐点核对。生成世界须有至少三个实质影响输出的变量，预置世界仅有两个；因此来源分区由可信属性保证，不通过查询开发/测试答案排除重复。生成世界之间按行为去重。

当前 Core 运行器处理有限整数输入、标量整数输出和 `probe` 原语；`DomainAdapter` 是扩展接口，尚未提供任意观察类型、任意 Python 实验程序或多步有状态世界的通用运行器。新领域若超出这些能力，需要扩展受限执行语言和交互协议，并重新冻结版本，不能仅替换一个类名后宣称已适配。

表达式 VM 禁止导入、文件、网络、反射、循环及任意函数调用，限制深度、节点、输入宽度和数值。这里的隔离来自语言没有主机能力，**不是任意 Python 代码的容器沙盒**。外部模型客户端和训练后端是可信宿主扩展，须由部署方自行隔离；不允许候选自行替换它们。`KernelBenchAdapter` 当前明确抛出 `CAPABILITY_UNSUPPORTED`。

## 模型和权重训练接入

默认无 API 调用。`JSONPolicyProposer(complete, model)` 可传给 `run_rsi(..., proposer=...)`，其中 `complete(request, output_limit)` 返回 `Completion`。`RunConfig.base_model` 必须与该模型一致。模型只接收 train 可见证据，并只能提交表征原语、实验策略和元策略的受限数据；不能改角色、基座、预算、分区、认证器或 Gate。截断、非法输出与使用量异常留档后报错，不用手写成功候选替代失败模型。网络代理、凭据与供应商调用由注入的客户端处理；接入真实网络前应按工作区 AGENTS.md 测试 65533 代理。

模型总 Token 按真实返回使用量统计，超限后不选代；当前接口无法在未知输入 Token 用量时保证远端请求绝不超额，部署客户端必须自行执行请求前 Token 预留和费用限制。默认本地运行不涉及此限制。

`TrainingBackend.fit` 必须实际产生独立角色目录中的权重增量及训练数据哈希，`train_role` 返回待验候选。当前未提供具体 LoRA/RL 优化器，也未将训练后端自动接入主循环。不能把该接口说成已经执行权重更新。推理客户端必须在部署时实现对应 checkpoint 的装载。

## 对照和证据

`--mode fixed`、`program_evolution`、`no_restructuring`、`no_meta` 提供工程对照。`program_evolution` 绕过显式诊断门槛，仍允许同样的重构算子。当前没有完整 Reflection、随机挑战者、难度优先挑战者和等计算预算论文实验套件；固定臂实际消耗较少，不宣称与 Full 已完成预算匹配。ANALOGIZE 需要已验证操作映射资产；默认不会凭空制造类比成功。

产物包括 `preregistration.json`、`journal.jsonl`、`objects/`、`frozen.json`、`final_started.json` 和 `summary.json`。对象与日志使用私有文件权限、独占创建及 fsync；哈希链能发现局部篡改，无法抵御机器所有者整体重写。原始审计目录包含 dev/test 私有记录，不应直接提供给 Agent。摘要只报告聚合结果。

资源计数定义：experiments 是 discovery 与反例独立复现实验数；predictions 包括评分、行为指纹及审核重复预测；search 是实际枚举候选数，不是 CPU 指令数；verification_outputs 统计可信解释器的所有顶层输出，包括世界认证、去重与审核。CPU/GPU 时间、训练 FLOPs 尚未测量。小样本 bootstrap 仅用于工程 Gate，不构成正式统计结论。
