from __future__ import annotations

import json
import os
import time
from typing import Any

import requests


class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.api_key = os.getenv(config.get("api_key_env", "DEEPSEEK_API_KEY"), "").strip()
        if not self.api_key:
            raise DeepSeekError(
                f"未配置 {config.get('api_key_env', 'DEEPSEEK_API_KEY')}，请写入项目根目录 .env"
            )
        self.base_url = str(config.get("base_url", "https://api.deepseek.com")).rstrip("/")
        self.model = str(config.get("model", "deepseek-v4-flash"))
        self.thinking = str(config.get("thinking", "disabled"))
        if self.thinking not in {"enabled", "disabled"}:
            raise DeepSeekError("ai.thinking 仅支持 enabled 或 disabled")
        self.timeout = int(config.get("timeout_seconds", 180))
        self.max_retries = int(config.get("max_retries", 2))

    def test_connection(self) -> dict[str, Any]:
        """Validate authentication without generating tokens."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}, timeout=min(self.timeout, 30),
            )
        except requests.RequestException as exc:
            raise DeepSeekError(f"无法连接DeepSeek：{exc}") from exc
        if not response.ok:
            raise DeepSeekError(
                f"DeepSeek连接测试失败（HTTP {response.status_code}）：{response.text[:300]}"
            )
        payload = response.json()
        models = [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
        return {"connected": True, "models": models, "configured_model": self.model}

    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 16000) -> tuple[dict[str, Any], dict[str, Any]]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise DeepSeekError(f"DeepSeek 暂时不可用（HTTP {response.status_code}）")
                if not response.ok:
                    detail = response.text[:300]
                    raise DeepSeekError(f"DeepSeek 请求失败（HTTP {response.status_code}）：{detail}")
                payload = response.json()
                choice = payload["choices"][0]
                content = choice["message"]["content"]
                if not content or not content.strip():
                    usage = payload.get("usage", {})
                    raise DeepSeekError(
                        "DeepSeek 返回空内容"
                        f"（finish_reason={choice.get('finish_reason', 'unknown')}，"
                        f"completion_tokens={usage.get('completion_tokens', 0)}）"
                    )
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError as exc:
                    finish_reason = choice.get("finish_reason", "unknown")
                    raise DeepSeekError(
                        "DeepSeek返回的JSON不完整"
                        f"（finish_reason={finish_reason}，completion_tokens="
                        f"{payload.get('usage', {}).get('completion_tokens', 0)}）：{exc}"
                    ) from exc
                self._validate(analysis)
                return analysis, payload.get("usage", {})
            except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError, DeepSeekError) as exc:
                last_error = exc
                if attempt >= self.max_retries or (
                    isinstance(exc, DeepSeekError) and (
                        ("HTTP 4" in str(exc) and "HTTP 429" not in str(exc))
                        or "JSON不完整" in str(exc)
                    )
                ):
                    break
                time.sleep(min(2 ** attempt, 4))
        raise DeepSeekError(f"AI分析失败：{last_error}") from last_error

    @staticmethod
    def _validate(result: dict[str, Any]) -> None:
        required = {
            "analysis_date", "overall_risk_level", "market_summary", "candidates",
            "portfolio_observations", "concentration_risks", "data_limitations", "disclaimer",
        }
        if not isinstance(result, dict) or not required.issubset(result):
            raise DeepSeekError("DeepSeek 返回的JSON缺少必要字段")
        if not isinstance(result["candidates"], list):
            raise DeepSeekError("DeepSeek 返回的候选明细格式不正确")
