#!/usr/bin/env bash
# ollama-agent 待辦執行腳本 — 對應 PROGRESS.md 第四節（2026-09-23）
#
# 用法:
#   scripts/setup_todo.sh              依序執行全部步驟
#   scripts/setup_todo.sh model tune   只執行指定步驟
#   scripts/setup_todo.sh --list       列出步驟
#
# 步驟（可重複執行，已完成的會自動略過）:
#   git       初次 commit（.tgz / .ollama-agent / .env / settings.local.json 已在 .gitignore）
#   mcp       核准本 repo 的 .mcp.json：寫入 .claude/settings.local.json，之後開 claude 不再跳核准
#   register  user-scope 註冊，讓其他 repo 也能用這些工具（scripts/register.sh）
#   model     建立離線用模型 qwen3.6-cc（qwen3.6:35b-a3b + 64K context）
#   tune      Ollama flash-attention + q8 KV cache（需 sudo；會重啟 ollama，已載入模型會被卸載）
#   verify    單元測試 + 整合測試 + ollama-agent --check + claude mcp list
#
# 沒放進來的: bin/claude-local 煙霧測試（會載入 35B、把 trio 擠出 GPU，要用時手動跑）。
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "請用一般使用者執行（不要加 sudo）：需要 sudo 的指令腳本會自己呼叫，並詢問密碼。" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
unset VIRTUAL_ENV || true   # 別的專案 venv 啟用中會干擾 uv

STEPS=(git mcp register model tune verify)
OVERRIDE=/etc/systemd/system/ollama.service.d/override.conf
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✔ %s\033[0m\n' "$*"; }
skip() { printf '    \033[33m– %s（略過）\033[0m\n' "$*"; }
die()  { printf '    \033[31m✘ %s\033[0m\n' "$*" >&2; exit 1; }

need() {
    local missing=()
    for c in "$@"; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
    [[ ${#missing[@]} -eq 0 ]] || die "缺少指令: ${missing[*]}"
}

# ---------------------------------------------------------------- git
do_git() {
    step "git: 初次 commit"
    if git rev-parse --verify HEAD >/dev/null 2>&1; then
        skip "已有 commit: $(git log --oneline -1)"
        return
    fi
    git add -A
    echo "    將提交 $(git diff --cached --name-only | wc -l) 個檔案"
    git commit -q -F - <<'EOF'
Initial commit: ollama-agent MCP server

Local Ollama models (qwen3.5 / qwen3.6) exposed to Claude Code as tools:
delegate_task, review_diff, summarize, index_codebase, search_code and
local_models_status. Includes the bin/claude-local offline launcher, the
local-reviewer subagent, register/bench scripts, unit and integration tests.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
    ok "$(git log --oneline -1)"
}

# ---------------------------------------------------------------- mcp
do_mcp() {
    step "mcp: 核准 .mcp.json 的 ollama-agent（.claude/settings.local.json）"
    mkdir -p .claude
    python3 - <<'EOF'
import json, pathlib
p = pathlib.Path(".claude/settings.local.json")
d = json.loads(p.read_text()) if p.exists() else {}
servers = set(d.get("enabledMcpjsonServers", []))
servers.add("ollama-agent")
d["enabledMcpjsonServers"] = sorted(servers)
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
EOF
    ok "enabledMcpjsonServers 已包含 ollama-agent；下次在本目錄開 claude 不會再要求核准"
}

# ---------------------------------------------------------------- register
do_register() {
    step "register: user-scope 註冊（uv tool install + claude mcp add --scope user）"
    scripts/register.sh
    ok "其他 repo 也可用 mcp__ollama-agent__*（切 profile: OLLAMA_AGENT_PROFILE=big claude）"
}

# ---------------------------------------------------------------- model
do_model() {
    step "model: 建立 qwen3.6-cc（35B + 64K context，供 bin/claude-local 使用）"
    if ollama show qwen3.6-cc >/dev/null 2>&1; then
        skip "qwen3.6-cc 已存在"
        return
    fi
    if ! ollama show qwen3.6:35b-a3b >/dev/null 2>&1; then
        echo "    基底 qwen3.6:35b-a3b 不在，先下載（約 22 GB）"
        ollama pull qwen3.6:35b-a3b
    fi
    ollama create qwen3.6-cc -f Modelfile.qwen3.6-cc
    ok "$(ollama list | grep '^qwen3.6-cc' || echo 'qwen3.6-cc 建立完成')"
}

# ---------------------------------------------------------------- tune
do_tune() {
    step "tune: Ollama flash-attention + q8 KV cache（需 sudo）"
    local want=(OLLAMA_HOST=0.0.0.0 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0)

    if [[ -r $OVERRIDE ]] && grep -q 'OLLAMA_FLASH_ATTENTION=1' "$OVERRIDE" \
        && grep -q 'OLLAMA_KV_CACHE_TYPE=q8_0' "$OVERRIDE"; then
        skip "override.conf 已含 flash-attention 與 q8 KV cache"
        return
    fi

    # 合併而不是覆蓋：保留既有行，只補缺的 Environment=
    local tmp; tmp="$(mktemp)"
    if [[ -r $OVERRIDE ]]; then cat "$OVERRIDE" > "$tmp"; else printf '[Service]\n' > "$tmp"; fi
    if ! grep -q '^\[Service\]' "$tmp"; then   # 既有檔案是空的或缺 section 標頭
        { printf '[Service]\n'; cat "$tmp"; } > "$tmp.new" && mv "$tmp.new" "$tmp"
    fi
    for kv in "${want[@]}"; do
        grep -q "Environment=\"${kv%%=*}=" "$tmp" || printf 'Environment="%s"\n' "$kv" >> "$tmp"
    done

    echo "    新的 override.conf:"; sed 's/^/    | /' "$tmp"
    echo "    會重啟 ollama：已載入的模型將被卸載，下次 MCP server 啟動時 warmup 會重新載入。"
    if [[ -r $OVERRIDE ]]; then
        sudo cp "$OVERRIDE" "$OVERRIDE.bak.$(date +%Y%m%d%H%M%S)"
    fi
    sudo install -m 644 "$tmp" "$OVERRIDE"
    rm -f "$tmp"
    sudo systemctl daemon-reload
    sudo systemctl restart ollama

    printf '    等待 ollama 回應 '
    for _ in $(seq 1 30); do
        if curl -fsS -m 2 "$OLLAMA_URL/api/version" >/dev/null 2>&1; then echo ok; break; fi
        printf .; sleep 1
    done
    curl -fsS -m 2 "$OLLAMA_URL/api/version" >/dev/null 2>&1 || die "ollama 30 秒內沒起來：sudo journalctl -u ollama -n 50"

    echo "    生效的環境變數:"
    systemctl show ollama -p Environment --value | tr ' ' '\n' | grep '^OLLAMA_' | sed 's/^/    | /'
    ok "ollama $(curl -fsS "$OLLAMA_URL/api/version" | sed 's/.*"version":"\([^"]*\)".*/\1/') 已重啟"
}

# ---------------------------------------------------------------- verify
do_verify() {
    step "verify: 單元測試（fake backend）"
    uv run pytest -q

    step "verify: 整合測試（真實 Ollama，會載入 trio 模型）"
    OLLAMA_AGENT_INTEGRATION=1 uv run pytest tests/integration -q

    step "verify: ollama-agent --check"
    uv run ollama-agent --check

    step "verify: claude mcp list"
    claude mcp list
    ok "驗證完成"
}

# ---------------------------------------------------------------- main
if [[ ${1:-} == "--list" || ${1:-} == "-l" ]]; then
    printf '%s\n' "${STEPS[@]}"; exit 0
fi

need git uv ollama claude python3 curl

if [[ $# -eq 0 ]]; then
    run=("${STEPS[@]}")
else
    run=("$@")
    for s in "${run[@]}"; do
        [[ " ${STEPS[*]} " == *" $s "* ]] || die "未知步驟 '$s'；可用: ${STEPS[*]}"
    done
fi

echo "ollama-agent 待辦：${run[*]}"
echo "工作目錄：$HERE"
for s in "${run[@]}"; do "do_$s"; done

cat <<EOF

全部完成。下一步：
  cd $HERE && claude            # 新 session，輸入「call local_models_status」確認六個工具可用
  bin/claude-local -p "hi"      # 選用：離線模式煙霧測試（會載入 35B，約 20 秒）
EOF
