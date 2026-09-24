from pathlib import Path

import pytest

from ollama_agent.config import ConfigError, Settings, load_dotenv
from ollama_agent.routing import PROFILES, embed_spec, gen_spec, model_for, profile_models, warmup_specs


def test_trio_mapping():
    s = Settings(profile="trio")
    assert model_for(s, "strong") == "qwen3.5:latest"
    assert model_for(s, "fast") == "qwen3.5:4b"
    assert embed_spec(s).num_gpu is None


def test_big_profile_never_loads_second_llm():
    s = Settings(profile="big")
    assert model_for(s, "strong") == model_for(s, "fast") == "qwen3.6:35b-a3b"
    assert embed_spec(s).num_gpu == 0  # embeddings pushed to CPU
    assert embed_spec(s).options() == {"num_ctx": 2048, "num_gpu": 0}
    assert [w.model for w in warmup_specs(s)] == ["qwen3.6:35b-a3b"]  # one LLM only


def test_warmup_order_largest_first():
    assert [w.model for w in warmup_specs(Settings(profile="trio"))] == ["qwen3.5:latest", "qwen3.5:4b"]


def test_overrides_win():
    s = Settings(profile="trio", strong_model="foo:1b", embed_model="bar")
    assert profile_models(s) == {"strong": "foo:1b", "fast": "qwen3.5:4b", "embed": "bar"}


def test_gen_spec_options_carry_ctx_and_caps():
    s = Settings(num_ctx=8192, keep_alive="5m")
    spec = gen_spec(s, "fast", think=True, temperature=0.3, num_predict=77)
    assert spec.options() == {"num_ctx": 8192, "temperature": 0.3, "num_predict": 77}
    assert spec.keep_alive == "5m" and spec.think is True


def test_every_profile_has_all_tiers():
    for name, p in PROFILES.items():
        assert p.strong and p.fast and p.embed, name


def test_from_env_and_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for k in list(__import__("os").environ):
        if k.startswith("OLLAMA_"):
            monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text("OLLAMA_AGENT_PROFILE=big\nOLLAMA_AGENT_NUM_CTX=16384 # comment\n")
    load_dotenv(tmp_path / ".env")
    s = Settings.from_env(cwd=tmp_path)
    assert s.profile == "big" and s.num_ctx == 16384
    assert s.data_dir == tmp_path / ".ollama-agent"


def test_invalid_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("OLLAMA_AGENT_PROFILE", "huge")
    with pytest.raises(ConfigError):
        Settings.from_env(cwd=tmp_path)
