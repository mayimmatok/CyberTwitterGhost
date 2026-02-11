# CyberTwitterGhost
使用 Wayback Twitter/X 快照构建 ShareGPT 数据集，支持过滤、清理、可选 AI 提示补全和一键 LLaMA-Factory 训练。Build ShareGPT datasets from Wayback Twitter/X snapshots with filtering, cleaning, optional AI prompt completion, and one-click LLaMA-Factory training.

---

## 前置条件 / Prerequisites

- Python 3.10 或更高版本（推荐 3.10/3.11）  
  Python 3.10+ (3.10/3.11 recommended)
- 稳定网络，可访问以下服务：  
  Stable network access to:
  - `web.archive.org`（抓取快照 / snapshot crawling）
  - OpenAI 兼容接口（仅在启用空 `human` 补全时需要 / only for empty-`human` completion）
  - `github.com`（仅在启用自动训练并克隆 LLaMA-Factory 时需要 / only for auto-training setup）
- OpenAI API Key（可选）  
  Optional OpenAI API key:
  - 用途：补全空 `human` 对话  
    Used for empty-`human` completion
  - 获取：在 OpenAI 平台创建 API Key（或使用兼容服务商 Key）  
    Get it from OpenAI dashboard (or compatible provider)
- 显卡（仅训练需要）  
  GPU is only required for training:
  - 推荐 NVIDIA RTX 5070 或更高，支持 CUDA 环境  
    NVIDIA RTX 5070+ with CUDA is recommended
  - 仅数据抓取/清洗/导出时不强制需要 GPU  
    GPU is not required for crawl/clean/export-only workflow

---

## 中文功能

- 抓取指定用户在 Wayback Machine 上的推文快照
- 过滤转推（Retweet）
- 提取对话对：
  - 回复场景：`human=被回复内容`，`gpt=用户回复`
  - 主动发言：`human=""`，`gpt=当前发言`
- 文本清洗：去 `@用户名`、去链接、繁转简
- 可选 AI 补全空 `human`（补全失败或选择不补全会删除该样本）
- 导出 ShareGPT 风格 JSON
- 可选一键接入 LLaMA-Factory 训练并启动 WebUI

---

## English Features

- Download archived tweets of target users from Wayback Machine
- Filter retweets
- Extract conversation pairs:
  - Reply case: `human=parent tweet`, `gpt=user reply`
  - Standalone case: `human=""`, `gpt=current tweet`
- Text cleanup: remove `@mentions`, remove links, convert Traditional Chinese to Simplified Chinese
- Optional AI completion for empty `human` (failed completions or cases without completions are removed)
- Export ShareGPT-style JSON
- Optional one-click LLaMA-Factory training + WebUI launch

---

## 使用方法（中文）

### 1. 运行

```bash
python tool.py
```

### 2. 按提示输入

1. `OPENAI_API_KEY`（可留空）
2. 用户名（支持多个，逗号分隔）
3. 是否启用空 `human` 补全
4. 是否在数据完成后直接训练并启动 WebUI

### 3. 结果文件

- `output/训练数据/<username>_sharegpt.json`
- `output/训练数据/all_users_sharegpt.json`

---

## Usage (English)

### 1. Run

```bash
python tool.py
```

### 2. Follow prompts

1. `OPENAI_API_KEY` (optional)
2. Username(s), comma-separated
3. Whether to enable empty-`human` completion
4. Whether to train and launch WebUI after dataset generation

### 3. Output files

- `output/训练数据/<username>_sharegpt.json`
- `output/训练数据/all_users_sharegpt.json`

---

## 环境变量 / Environment Variables

- `OPENAI_API_KEY` (for AI completion)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)

---

## 数据格式 / Data Format (ShareGPT)

```json
[
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]
```
