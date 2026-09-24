# ollama-agent — 進度回報

- 回報時間：2026-09-24 10:55 (CST)（前次：2026-09-23 17:20）
- 專案：`~/work/ollama`（ollama-agent，Python MCP server，把本機 Ollama 模型變成 Claude Code 的工具）
- 主機：<host>，RTX 4080 16 GB，62 GB RAM，Ollama 0.34.3
- 回報人：<email>（由 Claude 依程式碼、測試結果與系統狀態整理）

## 一、一句話結論

昨天的待辦 1–7 全部完成：已 commit、兩個 scope 都可用、離線模型已建立、Ollama 調校生效、六個工具都有真實 Ollama 整合測試。另修正 `register.sh` 兩個會讓 user-scope 註冊行為錯誤的問題。

## 二、今天完成

| # | 項目 | 結果 |
|---|---|---|
| 1 | git 初次 commit | `0505543`，44 檔 |
| 2 | 核准 `.mcp.json` | `claude mcp list` 為 ✔ Connected；`local_models_status` 正常回報 trio |
| 3 | user-scope 註冊 | `uv tool` 安裝 ollama-agent 0.1.0，註冊指向 `~/.local/bin/ollama-agent`；repo 外 ✔ Connected |
| 4 | 離線模型 | `ollama create qwen3.6-cc`，`num_ctx 65536`，共用原模型權重不佔額外磁碟 |
| 5 | Ollama 調校 | override.conf 加入 `OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`，已重啟生效（原檔備份於 `override.conf.bak.*`） |
| 6 | 整合測試補強 | 新增 `summarize`（單次、map-reduce）與 `index_codebase` + `search_code` 真實案例，`d83b272` |
| 7 | VRAM 競爭 | 調校後 trio 佔 11.1 GiB（前次 12.0），餘裕變大；切 `big` 前仍需卸載 trio |

### 修正（今天發現）

| Commit | 問題 | 修正 |
|---|---|---|
| `6d3744e` | `register.sh` 用 `command -v`，VS Code 啟用 `.venv` 時 user scope 會指到 repo 內的 `.venv/bin` | 改用 `uv tool dir --bin` |
| `6d3744e` | `sudo setup_todo.sh` 因 sudo 重設 PATH 而報「缺少指令: uv claude」 | 以 root 執行時直接提示改用一般使用者（需 sudo 的指令腳本自己會呼叫） |
| `f0974e6` | `register.sh` 以 `--env OLLAMA_AGENT_PROFILE=trio` 固定 profile；實測 MCP 設定的 env 會蓋過 shell，`OLLAMA_AGENT_PROFILE=big claude` 在其他 repo 無效 | 預設不寫 env（server 預設 trio），只有執行 `register.sh` 時明確設定才固定 |

## 三、驗證結果（10:30–10:55 實跑）

| 項目 | 結果 |
|---|---|
| `uv run pytest` | **34 passed, 7 skipped**，1.40 s |
| `OLLAMA_AGENT_INTEGRATION=1 uv run pytest tests/integration` | **7 passed**，23.9 s |
| ├ `test_summarize_single_pass` | qwen3.5:latest 找出埋入的 `db-7` ERROR，3.8 s |
| ├ `test_summarize_map_reduce` | 55 KB log，4B map + 9B reduce，30K tokens 輸入，`db-7` 保留到最終摘要，14.3 s |
| └ `test_index_and_search` | 4 檔語意搜尋第一名正確（score 0.80 / 0.62，次高 ≤ 0.25）；重新索引 `unchanged=4` |
| profile 切換（stdio `initialize`） | 未設變數為 `trio`，`OLLAMA_AGENT_PROFILE=big` 為 `big` |

### 調校前後（`scripts/bench.py trio moe big64k`）

| 項目 | 9/23（調校前） | 9/24（flash-attn + q8 KV） |
|---|---|---|
| trio embedding 在 GPU 比例 | 70–73% | **100%** |
| trio 三模型合計 | 12.0 GiB | **11.2 GiB** |
| trio 生成速度（9B / 4B） | 72 / 97 tok/s | 71 / 95 tok/s |
| big 32K 生成速度 | 36–38 tok/s | **39.5 tok/s**（GPU 12.2 GiB） |
| big 64K（`qwen3.6-cc` 等效） | 未測 | 38.9 tok/s，GPU 11.9 GiB，總量僅比 32K 多 0.1 GiB |
| big 冷載入 | 15–20 s | 14–24 s |

## 四、剩餘待辦

1. **`bin/claude-local` 煙霧測試**（選用）：`ollama stop` 卸載 trio 後執行 `bin/claude-local -p "hi"`；會載入 35B，約 20 s。
2. 現有 Claude Code session 若在 `register.sh` 修正前已啟動，重開才會套用新的 user-scope 設定。

## 五、環境快照（10:55）

| 項目 | 值 |
|---|---|
| Ollama | 0.34.3，`*:11434`；`OLLAMA_HOST=0.0.0.0`、`OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0` |
| 已下載模型 | qwen3.6:35b-a3b 22 GB（+ `qwen3.6-cc` 副本）、qwen3.5:latest 6.6 GB、qwen3.5:4b 3.4 GB、qwen3-embedding:0.6b 639 MB |
| 駐留中 | qwen3.5:latest、qwen3.5:4b、qwen3-embedding:0.6b，全部 100% GPU（合計 11.1 GiB） |
| GPU | 14.9 / 16.4 GB |
| 磁碟 `/` | 439 GB，已用 75%，剩 107 GB |
| git | `master`，4 個 commit（`0505543` → `f0974e6`） |
| 程式碼 | `src` + `tests` 共 2,563 行 |
