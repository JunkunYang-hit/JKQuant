from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from jkquant.ai.cache import AIAnalysisCache
from jkquant.ai.deepseek import DeepSeekClient
from jkquant.ai.prompts import SYSTEM_PROMPT, user_prompt
from jkquant.ai.service import run_ai_analysis


def _analysis() -> dict:
    return {
        "analysis_date": "2026-09-10", "overall_risk_level": "中",
        "market_summary": "测试", "portfolio_observations": [],
        "concentration_risks": [], "candidates": [], "data_limitations": [],
        "user_question_answer": {
            "question": "", "answer": "", "supporting_data": [], "limitations": [],
        },
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
    prompt = user_prompt({"analysis_date": "2026-09-10", "candidates": [], "user_question": "谁的风险较低？"})
    assert "json" in SYSTEM_PROMPT.lower()
    assert "严禁编造" in SYSTEM_PROMPT
    assert "INPUT_DATA=" in prompt
    assert "谁的风险较低" in prompt


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


def test_service_passes_selected_model_and_custom_question(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class Cache:
        def get(self, *args):
            return None

        def put(self, selected_date, provider, model, version, input_hash, analysis, context, usage):
            captured.update({"model": model, "context": context})
            return {"analysis": analysis, "model": model, "cached": False}

    class Client:
        def __init__(self, settings):
            captured["client_model"] = settings["model"]

        def complete(self, system, prompt, temperature, max_tokens):
            captured["prompt"] = prompt
            analysis = _analysis()
            analysis["user_question_answer"] = {
                "question": "哪些波动较低？", "answer": "测试回答",
                "supporting_data": [], "limitations": [],
            }
            return analysis, {}

    monkeypatch.setattr("jkquant.ai.service.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("jkquant.ai.service.build_analysis_context", lambda *args: {
        "analysis_date": "2026-09-10", "candidates": [],
    })
    monkeypatch.setattr("jkquant.ai.service._cache", lambda config: Cache())
    monkeypatch.setattr("jkquant.ai.service.DeepSeekClient", Client)
    config = {
        "_config_dir": str(tmp_path),
        "ai": {"provider": "deepseek", "model": "deepseek-flash",
               "models": ["deepseek-flash", "deepseek-v4-pro"]},
    }
    run_ai_analysis(
        config, date(2026, 9, 10), model="deepseek-v4-pro",
        user_question="  哪些波动较低？  ",
    )
    assert captured["client_model"] == "deepseek-v4-pro"
    assert captured["model"] == "deepseek-v4-pro"
    assert captured["context"]["user_question"] == "哪些波动较低？"
    assert "哪些波动较低" in captured["prompt"]
