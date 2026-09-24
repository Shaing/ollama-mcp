# ollama-agent — 進度回報

- 回報時間：2026-09-23 17:20 (CST)
- 專案：`~/work/ollama`（ollama-agent，Python MCP server，把本機 Ollama 模型變成 Claude Code 的工具）
- 主機：<host>，RTX 4080 16 GB，62 GB RAM，Ollama 0.34.3
- 回報人：<email>（由 Claude 依程式碼、測試結果與系統狀態整理；專案尚無 commit，時間依檔案 mtime 推估）

## 一、一句話結論

六個 MCP 工具全部實作完成，單元測試與真實 Ollama 整合測試全數通過；剩下的是「接線」工作：git 初次 commit、在 Claude Code 核准 `.mcp.json`、user-scope 註冊、建立離線用的 `qwen3.6-cc` 模型。

## 二、完成項目

### 核心程式（`src/ollama_agent`，含測試共 2,506 行）

| 模組 | 內容 | 狀態 |
|---|---|---|
| `server.py` | `MCPServer` 接線、六個工具的 docstring（Claude 讀的工具說明，含 NOT for 提示）、lifespan 背景 warmup | ✅ |
| `tools/delegate.py` | `delegate_task`：自足子任務交給本機模型，支援 context_files、strong/fast tier、think | ✅ |
| `tools/review.py` | `review_diff`：結構化 code review（Pydantic 輸出、`git diff` 自動取 diff、重試、raw fallback、verdict 與 findings 一致性校正） | ✅ |
| `tools/summarize.py` | `summarize`：map-reduce 摘要（fast 做 map、strong 做 reduce），可帶 question | ✅ |
| `tools/search.py` + `index/` | `index_codebase` / `search_code`：SQLite 向量索引、增量更新（mtime/hash）、語意搜尋 | ✅ |
| `tools/status.py` | `local_models_status`：profile、tier→model、`ollama ps` GPU/CPU 分布、外來模型警告 | ✅ |
| `routing.py` | `trio` / `big` 兩個 profile，每個 process 固定；`big` 絕不同時載入第二個 LLM；warmup 由大到小 | ✅ |
| `config.py` | 環境變數 + 自寫 `.env` loader，`--check` 模式 | ✅ |
| `backend.py` | Ollama AsyncClient 薄封裝，Protocol 介面供測試替換 FakeBackend | ✅ |
| `generate.py` / `outputs.py` / `files.py` / `chunking.py` | 串流生成含 timeout 回傳部分結果、完整輸出落地 `.ollama-agent/outputs/*.md` 並裁切回傳、檔案讀取預算、行切塊 | ✅ |

### 周邊

- `bin/claude-local`：Claude Code 走 Ollama `/v1/messages` 的離線啟動器，所有 model slot 指向同一模型 ✅
- `Modelfile.qwen3.6-cc`：35B 模型 64K context 的副本定義 ✅（模型本身尚未 `ollama create`，見待辦）
- `.claude/agents/local-reviewer.md`：先用本機 review_diff、再逐條驗證的子代理 ✅
- `scripts/register.sh`（uv tool 安裝 + user-scope `claude mcp add`）、`scripts/bench.py`（VRAM 配置 + tok/s 量測）✅
- `.mcp.json`（repo 範圍註冊）、`.env.example`、`.gitignore`、`README.md`、`CLAUDE.md` ✅

### 環境準備

- Ollama 0.17.6 → 0.34.3 升級並重啟（15:00），systemd override `OLLAMA_HOST=0.0.0.0` ✅
- 模型下載：qwen3.5:4b、qwen3-embedding:0.6b、qwen3.6:35b-a3b（既有 qwen3.5:latest 9B）✅
- 實測數據（`scripts/bench.py`）：

| Profile | 模型 | 速度 | 記憶體配置 |
|---|---|---|---|
| `trio`（預設） | qwen3.5:latest 9B + qwen3.5:4b + qwen3-embedding:0.6b | 72 / 97 tok/s | 三個全駐留 GPU（約 15.0 GB），前提是 9B 先載入 |
| `big` | qwen3.6:35b-a3b（strong 與 fast 同一模型） | 36–38 tok/s | 12 GiB GPU + 9 GiB RAM，embedding 強制 CPU |

## 三、驗證結果（17:15 實跑）

| 項目 | 結果 |
|---|---|
| `uv run pytest`（fake backend） | **34 passed, 4 skipped**，1.35 s（skipped 為整合測試） |
| `OLLAMA_AGENT_INTEGRATION=1 uv run pytest tests/integration -s` | **4 passed**，6.88 s |
| ├ `test_status` | 回報 trio profile、Ollama 0.34.3、三模型駐留 11.5 GiB GPU |
| ├ `test_delegate_fast` | qwen3.5:4b 回 PONG，0.1 s |
| ├ `test_embed` | qwen3-embedding 回 1024 維向量 |
| └ `test_review_small_diff` | qwen3.5:latest 抓到 ZeroDivisionError（critical），verdict request_changes，6.47 s |
| `uv run ollama-agent --check` | 正常；trio 三模型駐留，total on GPU 11.5 GiB |
| 手動 MCP smoke | `.ollama-agent/outputs/` 有兩筆 16:55–16:56 的 `delegate_task` READY 回應 |
| `claude mcp list` | `ollama-agent` 顯示 **Pending approval** |

## 四、尚未完成／待辦（建議順序）

> 第 1–5 項與驗證已整理成 `scripts/setup_todo.sh`，在設備上執行即可（`--list` 看步驟，可只跑指定步驟，已完成的會自動略過；`tune` 步驟需要 sudo 密碼）。

1. **git 初次 commit**：目前 `master` 零 commit，全部檔案 untracked（`*.tgz`、`.ollama-agent/`、`.env` 已在 .gitignore）。
2. **核准 `.mcp.json`**：在 `~/work/ollama` 執行 `claude`，同意 ollama-agent server，確認出現 `mcp__ollama-agent__*` 六個工具並呼叫 `local_models_status`。
3. **user-scope 註冊**（選用）：`scripts/register.sh`。目前 `uv tool list` 無 ollama-agent，其他 repo 還用不到這些工具。
4. **建立離線模型**：`ollama create qwen3.6-cc -f Modelfile.qwen3.6-cc`，否則 `bin/claude-local` 預設 MODEL 會報 not found。
5. **Ollama 調校**（需 sudo）：flash-attention、q8 KV cache 已評估建議但尚未套用。
6. **整合測試補強**：`summarize`、`index_codebase`、`search_code` 目前只有 fake backend 單元測試，建議加真實 Ollama 案例。
7. **VRAM 競爭**：GPU 現為 15.1 / 16.4 GB，trio 三模型已接近滿載。切到 `big` profile 前需先卸載 trio；`local_models_status` 會對外來模型提出警告。

## 五、環境快照（17:20）

| 項目 | 值 |
|---|---|
| Ollama | 0.34.3，`*:11434`，`OLLAMA_HOST=0.0.0.0` |
| 已下載模型 | qwen3.6:35b-a3b 22 GB、qwen3.5:latest 6.6 GB、qwen3.5:4b 3.4 GB、qwen3-embedding:0.6b 639 MB（模型目錄共 31 GB） |
| 駐留中 | qwen3.5:latest（GPU）、qwen3.5:4b（GPU）、qwen3-embedding:0.6b（73% GPU） |
| GPU | 15.1 / 16.4 GB |
| 磁碟 `/` | 439 GB，已用 75%，剩 108 GB |
