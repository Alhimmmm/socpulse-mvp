from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning


class GigaChatConfigurationError(RuntimeError):
    """Raised when GigaChat configuration is missing or invalid."""


class GigaChatRequestError(RuntimeError):
    """Raised when a request to GigaChat fails."""


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AIResponse:
    text: str
    model: str
    provider: str = "GigaChat"


class GigaChatClient:
    """GigaChat REST client with an in-memory OAuth access-token cache.

    Configuration stores only the long-lived Authorization Key. The short-lived
    access token is requested from the OAuth endpoint, cached in memory, refreshed
    shortly before expiry and retried once if the API returns HTTP 401.
    """

    TOKEN_REFRESH_MARGIN_SECONDS = 60
    DEFAULT_TOKEN_TTL_SECONDS = 30 * 60

    def __init__(self) -> None:
        self.auth_key = os.getenv("GIGACHAT_AUTH_KEY", "").strip()
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max").strip() or "GigaChat-2-Max"
        self.base_url = os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1").rstrip("/")
        self.auth_url = os.getenv(
            "GIGACHAT_AUTH_URL",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        ).strip()
        self.timeout = int(os.getenv("AI_TIMEOUT_SECONDS", "60"))
        self.verify_ssl = _env_bool("GIGACHAT_VERIFY_SSL", False)

        if not self.auth_key:
            raise GigaChatConfigurationError(
                "Не задан GIGACHAT_AUTH_KEY. Добавьте ключ авторизации GigaChat в файл .env."
            )

        if not self.verify_ssl:
            urllib3.disable_warnings(InsecureRequestWarning)

        self._cached_token = ""
        self._cached_token_expires_at = 0.0
        self._token_lock = threading.Lock()

    def _authorization_header(self) -> str:
        key = self.auth_key.strip()
        if key.lower().startswith("basic "):
            return "Basic " + key[6:].strip()
        return "Basic " + key

    @staticmethod
    def _normalize_expires_at(raw_value: Any, now: float) -> float:
        if raw_value is None:
            return now + GigaChatClient.DEFAULT_TOKEN_TTL_SECONDS
        try:
            expires_at = float(raw_value)
        except (TypeError, ValueError):
            return now + GigaChatClient.DEFAULT_TOKEN_TTL_SECONDS

        # GigaChat commonly returns Unix time in milliseconds.
        if expires_at > 10_000_000_000:
            expires_at /= 1000.0
        if expires_at <= now:
            return now + GigaChatClient.DEFAULT_TOKEN_TTL_SECONDS
        return expires_at

    def _request_new_access_token(self) -> str:
        now = time.time()
        try:
            response = requests.post(
                self.auth_url,
                headers={
                    "Authorization": self._authorization_header(),
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self.scope},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.Timeout as exc:
            raise GigaChatRequestError("GigaChat OAuth не ответил вовремя. Повторите попытку.") from exc
        except requests.SSLError as exc:
            raise GigaChatRequestError(
                "Ошибка SSL при подключении к GigaChat OAuth. Проверьте сертификаты или GIGACHAT_VERIFY_SSL."
            ) from exc
        except requests.RequestException as exc:
            raise GigaChatRequestError("Не удалось подключиться к сервису авторизации GigaChat.") from exc

        if response.status_code in {401, 403}:
            raise GigaChatConfigurationError(
                "GigaChat отклонил Authorization Key. Проверьте GIGACHAT_AUTH_KEY и GIGACHAT_SCOPE."
            )
        if response.status_code >= 500:
            raise GigaChatRequestError("Сервис авторизации GigaChat временно недоступен.")

        try:
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            token = str(data["access_token"]).strip()
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise GigaChatRequestError("GigaChat вернул некорректный ответ при получении access token.") from exc

        if not token:
            raise GigaChatRequestError("GigaChat не вернул access token.")

        self._cached_token = token
        self._cached_token_expires_at = self._normalize_expires_at(data.get("expires_at"), now)
        return token

    def _get_access_token(self) -> str:
        now = time.time()
        if self._cached_token and now < self._cached_token_expires_at - self.TOKEN_REFRESH_MARGIN_SECONDS:
            return self._cached_token

        with self._token_lock:
            now = time.time()
            if self._cached_token and now < self._cached_token_expires_at - self.TOKEN_REFRESH_MARGIN_SECONDS:
                return self._cached_token
            return self._request_new_access_token()

    def _invalidate_access_token(self) -> None:
        with self._token_lock:
            self._cached_token = ""
            self._cached_token_expires_at = 0.0

    def _api_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, *, retry_401: bool = True, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = dict(kwargs.pop("headers", {}))
        headers.update(self._api_headers())

        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise GigaChatRequestError("GigaChat не ответил вовремя. Повторите попытку.") from exc
        except requests.SSLError as exc:
            raise GigaChatRequestError(
                "Ошибка SSL при подключении к GigaChat. Проверьте сертификаты или GIGACHAT_VERIFY_SSL."
            ) from exc
        except requests.RequestException as exc:
            raise GigaChatRequestError("Не удалось подключиться к GigaChat API.") from exc

        if response.status_code == 401 and retry_401:
            self._invalidate_access_token()
            return self._request(method, path, retry_401=False, **kwargs)
        if response.status_code in {401, 403}:
            raise GigaChatRequestError("GigaChat отклонил временный access token или доступ к API.")
        if response.status_code == 429:
            raise GigaChatRequestError("Достигнут лимит запросов GigaChat. Подождите немного и повторите.")
        if response.status_code >= 500:
            raise GigaChatRequestError("GigaChat временно недоступен. Повторите попытку позже.")

        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GigaChatRequestError(f"GigaChat вернул ошибку HTTP {response.status_code}.") from exc
        return response

    def chat(self, messages: list[dict[str, str]]) -> AIResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1400,
        }
        response = self._request("POST", "/chat/completions", json=payload)
        try:
            data: dict[str, Any] = response.json()
            text = str(data["choices"][0]["message"]["content"])
            model = str(data.get("model") or self.model)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GigaChatRequestError("GigaChat вернул ответ в неожиданном формате.") from exc
        return AIResponse(text=text, model=model)

    def check_connection(self) -> list[str]:
        response = self._request("GET", "/models")
        try:
            data: dict[str, Any] = response.json()
            models = data.get("data", [])
            return [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]
        except (TypeError, ValueError) as exc:
            raise GigaChatRequestError("GigaChat вернул некорректный список моделей.") from exc
