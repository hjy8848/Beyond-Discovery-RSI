# Beyond-Discovery RSI

**范围更正：** 当前包含有界方法创造及研究控制元程序；尚未实现或验证任意复杂算法/研究方法的
开放式自主发明。原先无条件的 CODE-COMPLETE 表述过宽。具体扩展和未解决项见
[研究控制扩展报告](reports/RESEARCH_CONTROL_EXTENSION.md)。

按 `design/` 中 Core Design Spec v1.0 与 Decision Lock 实现的可执行研究系统。
M1–M7 的实现、验证范围与限制见 [CODE_COMPLETE_REPORT.md](CODE_COMPLETE_REPORT.md)。
24 项锁定原则、独立 Gate 和 Archive 保留。**工程闭环可运行；H1–H4 尚未得到科学验证。**

默认后端是有界、类型化程序合成，执行真实搜索和实验，不调用 LLM，也不模拟模型训练。
可选后端连接真实 HTTP LLM、本地 Hugging Face 模型，以及角色独立的 LoRA SFT / RL。
真实模型及训练路径在本开发环境未运行：缺少凭据、本地权重和训练依赖。

## 安装与本地复现

Python 3.9+。默认运行和测试只使用标准库：

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/demo.py --out runs/demo-001
PYTHONPATH=src python3 -m beyond_discovery.run conformance --domain stateful-scm
PYTHONPATH=src python3 -m beyond_discovery.run conformance --domain string-transducer
PYTHONPATH=src python3 -m beyond_discovery.run run --domain stateful-scm --generations 2 --out runs/scm-001
PYTHONPATH=src python3 -m beyond_discovery.run replay --domain stateful-scm --out runs/scm-001
```

`demo.py` 在整数状态因果机制和字符串转导机制中分别执行完整 C–S 闭环、独立证据验证、
表征/实验发明、双方元策略候选、独立选代、无充分证据时回滚、联合审计、冻结测试和 Replay。
不会为了演示成功强制 ACCEPT，也不把两个领域的分数混合成通用性结论。
本地证据保存于指定目录；每次必须使用新目录。Replay 要求相同源码实现哈希。

可选离线安装：`python3 -m pip install --no-index --no-deps --no-build-isolation -e .`。
原有 `causal`、`vector` 领域及 `FiniteResearchRuntime` 作为兼容基线保留。
`document` 可执行人工编写的协议样例，范围明确为协议验证：

```sh
PYTHONPATH=src python3 -m beyond_discovery.run run --domain document \
  --experiment examples/document-experiment.json --scope protocol-only \
  --mode fixed --generations 1 --out runs/document-001
```

## 执行与证据

- Domain Protocol 协商结构化动作、观察、输出 schema、可执行产物语言、世界轨道和预算。
  Core 不要求有限输入全集、固定 width 或整数输出；适配器声明具体能力。
- BoundaryCertificate 区分有限声明空间被反例证伪、空间内解释存在但搜索失败、资源不足未知。
  证书独立复算；单次预测错误不构成表征空间失败。
- Scientist 在 train 证据上进行重组、类比、约束放松、表征重构和实验程序合成。
  ResearchArtifact 保存实际代码、来源、父代、内容哈希和版本。类比资产只来自审核通过的 train 轨迹。
- Challenger 使用当前假设及边界证据合成世界和区分实验。独立解释器认证合法性、区分性、
  非平凡性、去重和有界可辨识性。认证失败或未定不会提供有效挑战奖励。
- MetaPolicy 可包含排序表达式，也可包含 `research-control-v1` 函数、分支、fold 和反馈驱动的
  能力调用，实际改变诊断子程序与研究步骤。固定评分公式的控制流变化也能独立评价。
  本地结构变异与模型提交均进入 train 试运行，再进入独立元 Gate。
- MetaPolicy 原有评分表达式仍参与候选排序/模型选择。固定父任务、固定对手、同一 train
  适应机会和预算，独立评价其产生的新方法；记录源代码差异和回滚。
- 双边候选在 dev 前持久化。dev 只进入独立评分/选择，不能进入生成、训练或提示。
  停止后先冻结，再打开一次最终测试；不能在同一运行目录重试最终测试。

Track A 使用注册原语及长度限制；Track B 可提交更广的有界类型化组合代码，经独立语义核对。
这里的“开放”只覆盖声明的可执行语言，不承诺认证任意 Python、外部科学事实或无限程序空间。
可辨识证书的范围是完整的小型注册比较语法加提案，行为等价按长度 ≤2 的公开探针定义；
找到 ≤8 步的独立区分实验才通过。范围外结论与预算内无法证明的结论均不冒充定理。

## 真实模型

HTTP 接入不依赖供应商 SDK。需要实际模型 ID、环境变量中的密钥，以及包含供应商提示框架
开销的输入 Token 上界。所有调用执行前预留；缺少 usage、截断、拒答、异常响应均报错。
传输结果不明确时保守消耗整个预留，不自动重试付费调用。

```sh
# 先按本机实际配置测试代理 65533；勿把凭据写入命令或仓库。
# 配置 OPENAI_API_KEY 后：
PYTHONPATH=src python3 -m beyond_discovery.run run --domain stateful-scm \
  --model-client http --base-model YOUR_PROVIDER_MODEL_ID \
  --endpoint https://api.openai.com/v1 --key-env OPENAI_API_KEY \
  --prompt-token-bound 32768 --max-model-tokens 1000000 \
  --out runs/http-001
```

`32768` 是配置示例，**不是对任意供应商或本运行提示长度的有效证明**；部署时须提供正确上界。
HTTP 模式不能装载本地 LoRA。SOCKS 代理需要安装 `.[model-http]`，并显式传入
`--proxy socks5://127.0.0.1:65533`；不允许把代理凭据写入 URL。
模型只能提交有界语言中的代码，实验程序经 Adapter 工具原语实际执行；不能改可信语义或 Gate。

## 真实 LoRA / RL

在有相应模型许可和资源的环境安装 `.[training]`，提前准备本地 HF 模型目录。
客户端和训练器均 `local_files_only=True`、`trust_remote_code=False`，不会暗中下载或切换模型。
每个角色从同一基座独立装载，训练各自 LoRA，保存 checksummed safetensors、manifest 和真实训练收据。

```sh
PYTHONPATH=src python3 -m beyond_discovery.run run --domain string-transducer \
  --model-client local --base-model /absolute/path/to/provisioned-hf-model \
  --evolution hybrid --training sft --training-steps 1 --max-training-steps 20 \
  --max-model-tokens 1000000 --device cpu --out runs/local-sft-001
```

`--training rl` 使用真实采样 Token、行为策略身份和 log probability 的 clipped importance
policy gradient；不是 PPO。只接受独立 train 执行产生的奖励，不接受 Agent 自报答案。
`--evolution program-only --training none`、`lora-only --training sft/rl`、`hybrid` 提供权重对照。
LoRA-only 固定生成模板、提示和采样种子，在同一 train 机会比较训练前后输出，独立 Gate 比较
固定程序的权重对照；训练后的执行产物仍完整归档。未训练成功不会生成假 checkpoint 或成功报告。
训练收据记录 step、Token、loss、设备和耗时；当前没有硬 GPU 时间或 FLOPs 配额。

## 对照、扩展和限制

```sh
PYTHONPATH=src python3 -m beyond_discovery.run suite --domain string-transducer \
  --generations 2 --out runs/string-controls-001
```

包含 full、fixed、fixed_challenger、random_challenger、hardness_first、program_evolution、reflection，
以及去 diagnosis、restructuring、falsification、meta 的消融。随机臂均匀抽取搜索池内合法候选；
hardness-first 按当前冻结模型的公开执行分歧排序；reflection 在原表示内重组。
同一配置上限和任务机会分别运行，全程报告实际成本，**不宣称各臂实际消耗完全相等**。
这些是清晰定义的本地算法对照，不冒充某篇论文的完整复现。HTTP 模型模式对未实现的搜索对照明确拒绝。

扩展 Adapter 可注册可信 `beyond_discovery.adapters` entry point，使用 `--external-adapter NAME`，
必须通过相同 conformance 和能力协商。KernelBench 需要独立 GPU worker、冻结参考和计时协议；
本仓库未运行外部 benchmark，缺少真实适配器/依赖时返回 `CAPABILITY_UNSUPPORTED`，不提供模拟成绩。
两个内置状态环境机制不同，但都属于可控符号研究环境，不是 H4 的外部跨领域证据。

完整状态领域运行与 Replay 在 spawn 子进程执行研究 AST，只有有界 JSON 跨越管道，
不继承评测器私有内存。worker 禁止文件、网络、子进程和 ctypes 审计操作，限制 CPU、总时限、
单次调用时限和载荷。Linux 使用地址空间限制；macOS 不支持该限制时由父进程每 20ms 检查 RSS
并终止超限 worker。监控存在采样间隔，不能声称内核级零超调内存隔离。限制写入预注册。
安全边界同时来自无主机能力的有界 AST / 工作流语言：无导入、文件、网络、反射或任意宿主代码执行。
真实模型客户端、可信 Adapter 和训练器在宿主执行，部署方需要为这些可信扩展配置资源隔离。
Archive 使用内容寻址、哈希链、私有目录、独占创建和 fsync；它不能抵抗机器所有者整体重写。
审计目录含私有 dev/test 记录，不应作为模型输入。Replay 复算确定性交互和认证、校验 checkpoint；
原始模型响应作为审计记录保留，不重放随机 API，不重新训练。非确定性 Adapter 的精确 Replay 明确不支持。

核心入口：`core/controller.py`、`boundary.py`、`synthesis.py`、`worlds.py`、`identification.py`、`meta.py`；
实际模型和优化器在 `runtime/models.py`、`model_research.py`、`trainer.py`。
阶段审查记录在 `reports/M1.md`–`M7.md`。供应商接口参考
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)，
LoRA 装载参考 [PEFT](https://huggingface.co/docs/peft/guides/peft_model_config)。

## 真实 APPS 编程评测

APPS 接入有独立 CLI。仓库锁定官方 `APPS.tar.gz` 的 SHA256 和完整任务清单，正式运行还要求
同一版本的 `manifest.json` 与数据 blob。清单包含 5,000 个官方 train 与 5,000 个官方 test
目录；train 按题目内容哈希整组切成 train/dev（4,042/958）。官方 test 在 smoke 和所有策略
选代期间保持封存。APPS 数据公开且可能进入预训练，不能视为无污染私有基准。

从 [官方 APPS release](https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz) 下载压缩包，
按 `dataset-lock.json` 的 `archive_sha256` 校验并导入。清单只放入仓库中的锁定元数据文件；题目数据、
参考答案和执行样本保留在本地目录：

```sh
curl --fail --location --output APPS.tar.gz https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
PYTHONPATH=src python3 -m beyond_discovery.benchmarks.apps.cli import \
  --archive APPS.tar.gz --sha256 6ef8e98ecca10b0159df0da4b524ecc1ca782a3b9473c57fc547ebccbbc2d0ca \
  --out /absolute/path/apps-v1
```

容器执行目前只支持 Linux 上的受信 OCI/Docker daemon。先审查锁定版本的官方 Python 评测语义，
按 `src/beyond_discovery/benchmarks/apps/container/Dockerfile` 构建镜像，`PYTHON_BASE` 必须是
审查过的 Python 3.11 镜像 SHA256 digest。运行时必须把最终 Judge 镜像写成 `name@sha256:…`；
不自动 pull、不使用宿主 Python。容器无网络、只读文件系统、非 root、丢弃 Linux capabilities、
no-new-privileges、seccomp、PID/内存/CPU/文件大小/输出/超时上限。输出由宿主可信比较器依官方
MIT evaluator 源码语义判定，隐藏输入和期望答案不会发送给提交代码。原始 APPS 的 5,000 个 test
任务中，版本清单登记 551 项因官方输入数据缺失、空/不匹配测试或 64 MiB 限制为
`CAPABILITY_UNSUPPORTED`；系统不把它们记为 WA 或 AC。

真实模型模式必须提供模型精确版本 ID、两个角色的 HTTP/HF 客户端和真实凭据/本地权重。HTTP
要求服务返回与请求版本完全匹配的模型 ID 和 token 用量。`CodingLimits` 同时限制 Scientist、
Challenger 的调用数、输入上界、单次输出上界；摘要统计推理真实用量和每次 Judge 执行成本。默认使用
同一 32,768 输入 token 上界及 2,048 输出 token 上界，单次输出最多可设为 32,768（还须满足
模型上下文长度）。APPS CLI 可传 `--max-model-tokens unlimited` 关闭累计 Token 停止条件；账本仍记录
实际用量。调用数、单次上下文长度和 Judge 容器执行限制独立于累计 Token 上限，并继续生效。

```sh
PYTHONPATH=src python3 -m beyond_discovery.benchmarks.apps.cli check \
  --store /absolute/path/apps-v1 --judge-image registry.example/apps-judge@sha256:<digest>
PYTHONPATH=src python3 -m beyond_discovery.benchmarks.apps.cli smoke \
  --store /absolute/path/apps-v1 --judge-image registry.example/apps-judge@sha256:<digest> \
  --provider http --model <immutable-model-id> --key-env APPS_API_KEY \
  --out runs/apps-smoke --verify-replay
PYTHONPATH=src python3 -m beyond_discovery.benchmarks.apps.cli suite \
  --store /absolute/path/apps-v1 --judge-image registry.example/apps-judge@sha256:<digest> \
  --provider http --model <immutable-model-id> --key-env APPS_API_KEY --out runs/apps-suite
```

`suite` 在同一 train/dev 成员、随机种子、模型/判题上限下独立执行 Direct、Self-Repair、固定
Meta RSI、进化 Meta RSI；每次都创建新的封存运行。实际 Token、Judge 次数、有效反例、算法行为变化、
任务 Gate、Meta Gate 和回滚写入 hash-chain archive。相同的是预算上限，实际模型生成长度与工具调用
次数可以不同，报告不会填充成本。

正式测试需先完成整套策略的 smoke 和冻结，并显式传入每个 frozen run：

```sh
PYTHONPATH=src python3 -m beyond_discovery.benchmarks.apps.cli final \
  --store /absolute/path/apps-v1 --judge-image registry.example/apps-judge@sha256:<digest> \
  --provider http --model <immutable-model-id> --key-env APPS_API_KEY \
  --frozen-run runs/apps-suite/direct --frozen-run runs/apps-suite/self-repair \
  --frozen-run runs/apps-suite/fixed-meta-rsi --frozen-run runs/apps-suite/evolved-meta-rsi \
  --out runs/apps-final
```

Final 命令在读取任何 test blob 前验证所有来源策略、实现哈希、数据/Judge/模型版本、冻结状态和一次性
final 标记，然后只运行冻结策略。不支持重试。同样的 model/Judge 真实路径依赖外部凭据和 Linux 镜像。
本仓库的 APPS 流程、数据清单与协议测试已实际运行；当前环境没有 APPS Judge 镜像或模型凭据，真实
APPS 模型训练/推理、隔离容器执行、官方运行时 parity 和实际成本都未运行。协议测试使用明确标注的
fixture，不代表 Benchmark 成绩或模型成功。

经独立 review 的 Oracle 配置只能来自 operator 私有文件，传入 `--oracle` 才会开放受限新输入认证。
Oracle 缺失则 Challenger 任务为 PENDING/CAPABILITY_UNSUPPORTED，不计分。APPS 公开问题也可能有
预训练污染；烟测和公开 split 上的结果不能证明无污染泛化或 H1–H4 科学假设。
