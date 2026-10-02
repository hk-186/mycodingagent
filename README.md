# MyCodingAgent

自研 AI Coding Agent——让模型像工程师一样改代码、跑测试、验证闭环。

## 功能一览

- **代码操作**：读文件、编辑文件、全局搜索、新建文件、运行 shell 命令
- **验证闭环**：改完立即自测，失败读 stderr 再修，超过轮次上限自动汇报
- **安全审批**：危险命令（删文件、写盘、Git 强制操作）按分级策略拦截或弹窗确认
- **Plan 模式**：复杂任务先出计划、等用户审批后再执行
- **记忆层**：个人偏好 + 项目约定（AGENTS.md）+ 长期语义记忆，换项目自动跟随
- **TUI 界面**：带事件日志、状态栏、审批弹窗的终端图形界面
- **Eval 回归集**：5 个真实任务级验收用例，改 Agent 必跑 eval 防退化
- **LangSmith 追踪**：可选接入，观测每次 eval 的 token/步数/耗时轨迹

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/hk-186/mycodingagent.git
cd mycodingagent

# 2. 安装（Python >= 3.11）
pip install -e ".[dev]"

# 3. 配置环境变量
copy .env.example .env        # Windows
cp .env.example .env          # Linux/macOS
```

编辑 `.env`，填入你的 API 密钥：

```
MODEL_NAME=deepseek-chat
BASE_URL=https://api.deepseek.com/v1
API_KEY=sk-xxxxxxxx
```

可选配置：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `AGENT_MAX_STEPS` | 每任务模型步数上限 | 40 |
| `AGENT_TOKEN_BUDGET` | 每任务 token 预算 | 300000 |
| `APPROVAL_MODE` | 审批模式：`minimal`/`all`/`disabled` | minimal |
| `EVAL_LANGSMITH` | 是否开启 LangSmith 追踪 | 0（关闭） |

## 启动

### 命令行模式（默认）

```bash
# 进入交互式会话
python -m mycodingagent

# 指定工作目录（改代码前切到目标项目）
python -m mycodingagent --project ./my-project
```

### TUI 模式（推荐）

```bash
python -m mycodingagent --tui
```

TUI 特点：
- 顶部标题栏 + 实时时钟
- 中部事件日志（模型思考、工具调用、结果）
- 底部状态栏（模型 / 会话 / 目录 / 审批 / 步数 / token）
- 按 `t` 展开/收起模型的思考过程
- 审批时自动切换「决策>」输入栏

## 命令速查

在 `你>` 提示符下输入：

| 命令 | 作用 |
|---|---|
| `/help` | 显示完整帮助 |
| `/new [名字]` | 开启新会话 |
| `/sessions` | 列出历史会话（TUI 弹窗选择，CLI 文本列表） |
| `/switch <id\|序号>` | 切换到指定会话（CLI 专用） |
| `/delete <id\|序号>` | 删除会话历史 |
| `/resume` | 从断点继续中断的任务 |
| `/plan <任务描述>` | Plan 模式：先出计划等审批 |
| `/memory` | 查看长期记忆 |
| `/history` | 当前会话消息统计 |
| `/pwd` | 查看工作目录 |
| `/cd <路径>` | 切换工作目录 |
| `/exit` | 退出 |

**审批弹窗下可用**（任务执行中自动弹出）：

| 输入 | 作用 |
|---|---|
| `/approve` 或 `a` | 通过当前操作 |
| `/reject <理由>` 或 `r <理由>` | 拒绝 |
| `/respond <内容>` 或 `rr <内容>` | 代答 ask_user |
| `/cancel` 或 `Ctrl+D` | 取消 |

## 典型工作流

### 1. 修 Bug

```
你> workspace/utils.py 里的 clamp 函数当 low==high 时返回错误，修复它
```

Agent 会：
1. 读文件定位问题
2. 修改代码
3. 运行 `python -m pytest` 验证
4. （若配置了 code-reviewer）审查改动
5. 汇报结果

### 2. 加功能（Plan 模式）

```
你> /plan 给 utils.py 加一个 is_palindrome 函数，支持中文
```

Agent 先出执行计划，你审批通过后逐步执行，每步可继续审批或拒绝。

### 3. 换项目

```
你> /cd ../another-project
```

工作目录、Git 工具、项目记忆（AGENTS.md）全部跟随切换。

## Eval 回归集

改 Agent 代码后，跑 eval 验证是否引入退化：

```bash
# 全量跑（约 1~2 分钟）
python scripts/run_evals.py

# 单任务
python scripts/run_evals.py --task fix-clamp-boundary

# 与基线对比
python scripts/run_evals.py --baseline evals/results/eval-xxx.json

# 只校验任务定义，不调用模型
python scripts/run_evals.py --dry-run
```

## 技术栈

- **Agent 框架**：LangGraph + LangChain
- **LLM 网关**：OpenAI-compatible API（DeepSeek / 阿里通义等）
- **持久化**：SQLite（checkpoints + 语义向量索引）
- **向量索引**：sqlite-vec（本地 embedding，无需额外服务）
- **TUI**：Textual
- **测试**：pytest
- **可选观测**：LangSmith

## 项目结构

```
mycodingagent/
├── src/mycodingagent/          # 核心源码
│   ├── agent.py                # Agent 装配（LLM + 工具 + prompt）
│   ├── cli.py                  # 命令行入口
│   ├── events.py               # 结构化事件流 + 预算执行
│   ├── tui/                    # TUI 界面
│   ├── eval/                   # Eval 回归框架
│   ├── sessions.py             # 会话管理
│   ├── tools/                  # 工具集（文件、Git、记忆、计划等）
│   └── config.py               # 集中配置
├── tests/                      # 单元测试
├── evals/tasks/                # Eval 任务定义（task.json + fixture）
├── evals/results/              # Eval 运行结果
├── scripts/run_evals.py        # Eval CLI 入口
├── .env.example                # 配置模板
└── docs/开发计划.md            # 详细开发计划
```

## 许可证

MIT
