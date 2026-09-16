from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests


class GigaChatConfigurationError(RuntimeError):
    """Raised when GigaChat credentials are missing or invalid."""


class GigaChatRequestError(RuntimeError):
    """Raised when a request to GigaChat fails."""


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class AIResponse:
    text: str
    model: str
    provider: str = "GigaChat"


class GigaChatClient:
    """Small REST client for GigaChat suitable for the MVP.

    Two authorization modes are supported:
    1. GIGACHAT_AUTH_KEY -> OAuth token is requested and cached.
    2. GIGACHAT_ACCESS_TOKEN / GC_TOKEN -> an already issued access token is used.

    The second mode preserves compatibility with the earlier local setup where
    the token was stored as GC_TOKEN.
    """

    def __init__(self) -> None:
        self.auth_key = os.getenv("GIGACHAT_AUTH_KEY", "").strip()
        self.access_token = (
            os.getenv("GIGACHAT_ACCESS_TOKEN", "").strip()
            or os.getenv("GC_TOKEN", "").strip()
        )
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max").strip() or "GigaChat-2-Max"
        self.base_url = os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1").rstrip("/")
        self.auth_url = os.getenv(
            "GIGACHAT_AUTH_URL",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        ).strip()
        self.timeout = int(os.getenv("AI_TIMEOUT_SECONDS", "60"))
        self.verify_ssl = _env_bool("GIGACHAT_VERIFY_SSL", False)
        self._cached_token = ""
        self._cached_token_expires_at = 0.0

        if not self.auth_key and not self.access_token:
            raise GigaChatConfigurationError(
                "Не задан GIGACHAT_AUTH_KEY или GIGACHAT_ACCESS_TOKEN/GC_TOKEN"
            )

    @property
    def credential_mode(self) -> str:
        return "access_token" if self.access_token else "oauth"

    def _oauth_token(self) -> str:
        now = time.time()
        if self._cached_token and now < self._cached_token_expires_at - 60:
            return self._cached_token

        auth_header = self.auth_key
        if not auth_header.lower().startswith("basic "):
            auth_header = "Basic " + auth_header

        try:
            response = requests.post(
                self.auth_url,
                headers={
                    "Authorization": auth_header,
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self.scope},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            token = str(data["access_token"])

            raw_expires_at = data.get("expires_at")
            expires_at = 0.0
            if raw_expires_at is not None:
                expires_at = float(raw_expires_at)
                if expires_at > 10_000_000_000:
                    expires_at /= 1000.0
            if expires_at <= now:
                expires_at = now + 30 * 60

            self._cached_token = token
            self._cached_token_expires_at = expires_at
            return token
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            detail = ""
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                detail = f" Ответ API: {exc.response.text[:500]}"
            raise GigaChatRequestError(
                f"Не удалось получить токен GigaChat: {exc}.{detail}"
            ) from exc

    def _token(self) -> str:
        return self.access_token or self._oauth_token()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def chat(self, messages: list[dict[str, str]]) -> AIResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1400,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            text = str(data["choices"][0]["message"]["content"])
            model = str(data.get("model") or self.model)
            return AIResponse(text=text, model=model)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            detail = ""
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                detail = f" Ответ API: {exc.response.text[:500]}"
            raise GigaChatRequestError(f"Ошибка GigaChat: {exc}.{detail}") from exc

    def check_connection(self) -> list[str]:
        """Return model ids visible to the current token."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={
                    "Authorization": f"Bearer {self._token()}",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            models = data.get("data", [])
            return [str(item.get("id")) for item in models if item.get("id")]
        except (requests.RequestException, TypeError, ValueError) as exc:
            detail = ""
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                detail = f" Ответ API: {exc.response.text[:500]}"
            raise GigaChatRequestError(
                f"Не удалось проверить подключение к GigaChat: {exc}.{detail}"
            ) from exc
