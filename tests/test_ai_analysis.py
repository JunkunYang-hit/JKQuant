from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from jkquant.ai.cache import AIAnalysisCache
from jkquant.ai.deepseek import DeepSeekClient
from jkquant.ai.prompts import SYSTEM_PROMPT, user_prompt


def _analysis() -> dict:
    return {
        "analysis_date": "2026-09-10", "overall_risk_level": "中",
        "market_summary": "测试", "portfolio_observations": [],
        "concentration_risks": [], "candidates": [], "data_limitations": [],
        "disclaimer": "仅供量化研究参考，不构成投资建议。",
    }


def test_ai_cache_round_trip(tmp_path: Path) -> None:
    cache = AIAnalysisCache(tmp_path / "ai.sqlite3")
    saved = cache.put(
        date(2026, 9, 10), "deepseek", "test-model", "v1", "hash",
        _analysis(), {"candidates": []}, {"total_tokens": 12},
    )
    assert saved["cached"] is False
    loaded = cache.get(date(2026, 9, 10), "deepseek", "test-model", "v1", "hash")
    assert loaded is not None
    assert loaded["cached"] is True
    assert loaded["analysis"]["overall_risk_level"] == "中"
    assert cache.latest(date(2026, 9, 10))["usage"]["total_tokens"] == 12


def test_prompt_requires_json_and_rejects_fabricated_news() -> None:
    prompt = user_prompt({"analysis_date": "2026-09-10", "candidates": []})
    assert "json" in SYSTEM_PROMPT.lower()
    assert "严禁编造" in SYSTEM_PROMPT
    assert "INPUT_DATA=" in prompt


def test_deepseek_client_uses_json_mode(monkeypatch) -> None:
    captured = {}

    class Response:
        status_code = 200
        ok = True
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": json.dumps(_analysis(), ensure_ascii=False)}}],
                "usage": {"total_tokens": 9},
            }

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setattr("jkquant.ai.deepseek.requests.post", fake_post)
    client = DeepSeekClient({"model": "test-model", "max_retries": 0})
    result, usage = client.complete("system json", "user")
    assert result["analysis_date"] == "2026-09-10"
    assert usage["total_tokens"] == 9
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["headers"]["Authorization"] == "Bearer not-a-real-key"


def test_deepseek_connection_check_does_not_generate_tokens(monkeypatch) -> None:
    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]}

    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setattr("jkquant.ai.deepseek.requests.get", lambda *args, **kwargs: Response())
    result = DeepSeekClient({"model": "deepseek-v4-flash"}).test_connection()
    assert result["connected"] is True
    assert result["models"] == ["deepseek-flash", "deepseek-v4-pro"]
