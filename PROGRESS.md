# ollama-agent — 進度回報

- 回報時間：2026-09-29 16:25 (CST)（前次：2026-09-29 12:05）
- 專案：`~/work/ollama`（ollama-agent，Python MCP server，把本機 Ollama 模型變成 Claude Code 的工具）
- 主機：本機（RTX 4080 16 GB，62 GB RAM，Ollama 0.34.3）
- 回報人：專案維護者（由 Claude 依程式碼、測試結果與系統狀態整理）

## 一、一句話結論

昨天的待辦 1–7 全部完成：已 commit、兩個 scope 都可用、離線模型已建立、Ollama 調校生效、六個工具都有真實 Ollama 整合測試。另修正 `register.sh` 兩個會讓 user-scope 註冊行為錯誤的問題。11:30 追加：修好 MCP `summarize` 一律失敗、輸出不再寫進使用者的 repo、`claude-local` 的 context 長度設定；`claude-local` 煙霧測試通過，目前沒有已知問題。

## 二、今天完成

| # | 項目 | 結果 |
|---|---|---|
| 1 | git 初次 commit | `8d3f8e7`，44 檔 |
| 2 | 核准 `.mcp.json` | `claude mcp list` 為 ✔ Connected；`local_models_status` 正常回報 trio |
| 3 | user-scope 註冊 | `uv tool` 安裝 ollama-agent 0.1.0，註冊指向 `~/.local/bin/ollama-agent`；repo 外 ✔ Connected |
| 4 | 離線模型 | `ollama create qwen3.6-cc`，`num_ctx 65536`，共用原模型權重不佔額外磁碟 |
| 5 | Ollama 調校 | override.conf 加入 `OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`，已重啟生效（原檔備份於 `override.conf.bak.*`） |
| 6 | 整合測試補強 | 新增 `summarize`（單次、map-reduce）與 `index_codebase` + `search_code` 真實案例，`1746303` |
| 7 | VRAM 競爭 | 調校後 trio 佔 11.1 GiB（前次 12.0），餘裕變大；切 `big` 前仍需卸載 trio |

### 修正（今天發現）

| Commit | 問題 | 修正 |
|---|---|---|
| `02d4097` | `register.sh` 用 `command -v`，VS Code 啟用 `.venv` 時 user scope 會指到 repo 內的 `.venv/bin` | 改用 `uv tool dir --bin` |
| `02d4097` | `sudo setup_todo.sh` 因 sudo 重設 PATH 而報「缺少指令: uv claude」 | 以 root 執行時直接提示改用一般使用者（需 sudo 的指令腳本自己會呼叫） |
| `1825c23` | `register.sh` 以 `--env OLLAMA_AGENT_PROFILE=trio` 固定 profile；實測 MCP 設定的 env 會蓋過 shell，`OLLAMA_AGENT_PROFILE=big claude` 在其他 repo 無效 | 預設不寫 env（server 預設 trio），只有執行 `register.sh` 時明確設定才固定 |
| `b28d561` | MCP `summarize` 工具一律失敗（`Error executing tool summarize`，0 s）：`server.py` 內的工具函式名稱 `summarize` 蓋掉同名模組，`summarize.summarize(...)` 變成找函式屬性；單元測試直接呼叫模組所以沒抓到 | 模組改以 `summarize_tool` 匯入；新增經由 MCP server 呼叫的測試 `test_summarize.py::test_call_through_mcp_server`。全部 35 passed / 7 skipped，stdio 實測對真實 Ollama 成功 |
| `971828a` | 輸出和搜尋索引預設寫在 `<cwd>/.ollama-agent`、`<repo>/.ollama-agent`，user scope 會在每個用過的 repo 留下未追蹤資料夾（這次在 `~/work` 也產生了） | 改為 `$XDG_CACHE_HOME/ollama-agent`（`~/.cache/ollama-agent`），索引以 repo 路徑 hash 分檔；舊資料夾已清除、輸出已搬過去 |
| `aeb8dd6` | `claude-local` 煙霧測試時 Claude Code 警告不認得 `qwen3.6-cc`，會假設 200K context；實際 64K，超過前不會自動壓縮，Ollama 會直接截斷 | 設 `CLAUDE_CODE_MAX_CONTEXT_TOKENS=$NUM_CTX`，重跑警告消失 |

## 三、驗證結果（10:30–10:55 實跑）

| 項目 | 結果 |
|---|---|
| `uv run pytest` | **35 passed, 7 skipped**，1.40 s |
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

### 追加驗證（11:05–11:30）

| 項目 | 結果 |
|---|---|
| `uv run pytest` | **36 passed, 7 skipped**（新增索引位置測試） |
| 整合測試 | **7 passed**，23.2 s（新資料位置下） |
| 六個 MCP 工具經由 Claude Code 實際呼叫 | 全部成功；`summarize` 5.4 s、`delegate_task`(4B) 1.0 s、`search_code` 0.8 s、`review_diff` 4.1 s |
| `bin/claude-local -p` | 回覆正確；冷啟動含 35B 載入 58 s，熱啟動 4 s；35B 為 44%/56% CPU/GPU。測完已卸載並重新載入 trio |
| `ruff check`（`uv run --with ruff`） | 14 個既有風格問題（非本次引入），未處理 |

## 四、剩餘待辦

1. 在 `971828a` 之前啟動的 ollama-agent 仍會寫到舊位置，重開 Claude Code session 後生效。
2. （選用）14 個 ruff 風格問題；專案目前沒有把 ruff 列入 dev 相依。

## 五、環境快照（11:30）

| 項目 | 值 |
|---|---|
| Ollama | 0.34.3，`*:11434`；`OLLAMA_HOST=0.0.0.0`、`OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0` |
| 已下載模型 | qwen3.6:35b-a3b 22 GB（+ `qwen3.6-cc` 副本）、qwen3.5:latest 6.6 GB、qwen3.5:4b 3.4 GB、qwen3-embedding:0.6b 639 MB |
| 駐留中 | qwen3.5:latest、qwen3.5:4b、qwen3-embedding:0.6b，全部 100% GPU（合計 11.1 GiB） |
| GPU | 14.9 / 16.4 GB |
| 磁碟 `/` | 439 GB，已用 75%，剩 107 GB |
| git | `master`，9 個 commit（`8d3f8e7` → `aeb8dd6`，另有本報告） |
| 程式碼 | `src` + `tests` 共 2,573 行 |

## 六、形式化驗證（16:05 追加）

把 README、工具 docstring、`.env.example`、CLAUDE.md 裡的承諾抽成 60 餘條可檢驗的性質，寫進 `formal/SPEC.md`，分四層驗證；`formal/run.sh` 一次跑完，任何一層與 SPEC 不一致就以非 0 結束。

| 層 | 位置 | 結果 |
|---|---|---|
| SMT 證明（Z3） | `formal/smt/verify.py` | 22 條：**15 條對所有輸入成立**，7 條反例（全部是預期中的已知偏差，見下） |
| 模型檢查（TLA+/TLC） | `formal/tla/OllamaAgent.tla` + 5 個 cfg | trio 14,174 / big 8,367 / conc1 734 / nowarmup 5,750 個狀態全數通過 9 條不變式 + 2 條 liveness；`trio_foreign` 依設計違反 `NoReload`（外部模型會造成重載，證明 `local_models_status` 的警告有必要） |
| 符號執行（CrossHair） | `formal/crosshair/contracts.py` | 真實 `clip` 四條後置條件在界限內 Confirmed over all paths；`estimate_tokens`、`chunk_lines` 時限內無反例（未窮盡） |
| 性質測試 + 靜態檢查（Hypothesis / AST） | `tests/test_properties.py`、`tests/test_spec_static.py` | 16 passed、3 xfail（釘住已知偏差）；含 SQLite 索引的 model-based 狀態機測試、「每個 Ollama 呼叫都在 semaphore 內」等 AST 檢查 |

全套 `uv run pytest`：52 passed、7 skipped、3 xfailed。新增相依：`hypothesis`（dev）、`z3-solver`、`crosshair-tool`（`formal` group）；TLC 需 Java 11+，jar 首次執行自動下載到 `~/.cache/tla2tools/`。

### 發現（程式與規格不一致，皆未修，待決定）

| # | 位置 | 問題 | 嚴重度 |
|---|---|---|---|
| F1 | `tools/summarize.py` | 門檻常數假設 32K context，不看 `settings.num_ctx`；`OLLAMA_AGENT_NUM_CTX=8192` 時 45K 字元的單次摘要會被 Ollama 靜默截斷（最低安全值 15,038） | 中 |
| F6 | `chunking.chunk_lines` | 行很長時（summarize ≥1,333 字元/行、索引 ≥375）視窗每次只前進一行，map 輸入最多放大 8 倍、embedding 3 倍，且會產生完全包含在前一塊裡的重複 chunk；預設設定即可觸發 | 中 |
| F2 | `tools/review.py` | 「further cut to fit」差一個 token（Z3 已證明改成 −401 即可） | 低 |
| F7 | `warmup.py` | `MAX_CONCURRENCY=1` 時第一個工具呼叫要等整個 warmup（big 約 20 s），文件未提 | 低（文件） |
| F4 | `delegate_task` docstring | 「~90k chars」只在 `max_tokens ≤ 6,383` 成立 | 低（文件） |
| F3 / F5 / F8 / F9 | review / chunking / config / routing | 邊角設定：review 的 `num_ctx < 4497`、chunk 負 overlap、`MAX_TIMEOUT_S < 5`、big profile 被 override 打破單 LLM 而 status 不警告 | 低 |

細節、反例與建議修法見 `formal/SPEC.md` 末段。

### 修正（16:40）

F1、F2、F5、F6 已修（commit 見 git log），`formal/run.sh` 全綠；`uv run pytest` **57 passed / 7 skipped**、不再有 xfail；Z3 23 條中 20 條證明成立、3 條是保留的邊角反例（F3、F8 與 F4 的文件措辭）。

- F1：`summarize` 新增 `budgets(num_ctx, question)`，門檻隨 `num_ctx` 縮放，map chunk 不超過 reduce 輸入（fold 迴圈必收斂），`num_ctx < 2,397` 在呼叫模型前直接回錯誤；32K 時與原常數相同（Z3 S3–S5 證明）。
- F6：`chunk_lines` 的 overlap 上限為半塊，且每塊一定比前一塊多至少一行；輸出行數 ≤ 2× 輸入（Z3 C5、C6 證明，Hypothesis 驗證）。
- F2：review 的裁切改減 401 token，對所有 diff 與 `num_ctx` 都放得進 context（Z3 R1、R1b）。
- F5：`overlap_lines < 0` 拋 `ValueError`。
- 仍開放：F3、F4、F7、F8、F9（邊角設定與文件），見 `formal/SPEC.md` 末段。

## 七、`delegate_task` 的 think 額度（2026-09-29）

問題（9/24 情境測試 A1）：`think=true` 時預設 `max_tokens=4096` 可能全部花在思考，回傳「(model returned no text)」，而且思考內容沒寫進輸出檔，無從回收。當時改傳 16384 才成功，但 server 上限是 8192（初版就有），所以實際是 8192。專案維護者選「兩者都做」：

- `max_tokens` 預設改成 0（自動）：不開 think 仍是 4096；開 think 給 8192，context 不夠就縮到放得下為止，但不低於 4096。明確傳的值照用（仍夾在 64–8192）。
- 輸出檔在答案後面接模型的思考內容（`<!-- ollama-agent thinking (not part of the answer) -->` 標記），回給 Claude 的內容不含思考；meta 多記實際用的 `max_tokens`。
- 只有思考、沒有答案時回傳加一條 warning，指向輸出檔；撞到長度上限時提示加大 `max_tokens` 或關 think。

驗證：Z3 新增 D5（開 think 且用預設時 `num_predict ≥ 4096`，放得下就是 8192）、D6（新預設不會拒絕舊的固定 4096 會接受的輸入），D1、D2 改成涵蓋 think；`uv run pytest` 62 passed / 7 skipped（新增 5 個測試），`formal/run.sh` 全綠。真實 9B 重跑 A1：預設額度 8192，用了 4063 tokens、55.4 s，有答案且思考存檔；故意傳 `max_tokens=1024` 時回「(model returned no text)」加上述 warning，思考內容在輸出檔裡。

### `review_diff` 的同類問題（2026-09-29 16:25）

`review_diff` 預設開 think、上限 4096。思考吃光額度時，`truncated` 會讓它跳過「不開 think 重試一次」，直接回 fallback 的 `comment`（空內容），還附上不適用的「raise max_tokens」提示（這個工具沒有這個參數）。用真實 9B 審 `fa3a76d` 的 src diff（約 100 行）重現：第一次就發生，54 s 回空結果。

修法：開 think 且撞到長度上限、JSON 又無效時，同一個問題不開 think 再問一次（沒有可修正的 JSON，所以不加修正提示），notes 說明原因、不再轉述「raise max_tokens」；沒開 think 的長度截斷維持不重試。思考內容比照 `delegate_task` 存在輸出檔。上限維持 4096，因為 diff 的裁切與 Z3 R1 都以它為準。

驗證：SPEC 新增 V6；`uv run pytest` 64 passed / 7 skipped（新增 2 個測試，`ThinkingBackend` 移到 `conftest.py` 共用）。真實 9B：修正後同一個 diff 用預設值 17 s 就有結果（這次思考沒有超過額度）；把上限壓到 1024 強制走重試路徑，兩次都是「思考吃光 → 不開 think 重試 → 有效的 review」，各 16.5 s。
