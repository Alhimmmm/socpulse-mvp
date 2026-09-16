from unittest.mock import Mock, patch

import pytest

from src.ai import GigaChatClient, GigaChatConfigurationError


def clear_gigachat_env(monkeypatch):
    for name in [
        "GIGACHAT_AUTH_KEY",
        "GIGACHAT_SCOPE",
        "GIGACHAT_MODEL",
        "GIGACHAT_BASE_URL",
        "GIGACHAT_AUTH_URL",
        "GIGACHAT_VERIFY_SSL",
        "AI_TIMEOUT_SECONDS",
        "GIGACHAT_ACCESS_TOKEN",
        "GC_TOKEN",
    ]:
        monkeypatch.delenv(name, raising=False)


def token_response(token="token-1", expires_at=2_000_000_000_000):
    response = Mock(status_code=200)
    response.json.return_value = {"access_token": token, "expires_at": expires_at}
    response.raise_for_status.return_value = None
    return response


def api_response(status=200, payload=None):
    response = Mock(status_code=status)
    response.json.return_value = payload if payload is not None else {"data": []}
    response.raise_for_status.return_value = None
    return response


def test_client_requires_authorization_key(monkeypatch):
    clear_gigachat_env(monkeypatch)
    with pytest.raises(GigaChatConfigurationError, match="GIGACHAT_AUTH_KEY"):
        GigaChatClient()


def test_basic_prefix_is_added_once(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "abc123")
    client = GigaChatClient()
    assert client._authorization_header() == "Basic abc123"

    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "Basic abc123")
    client = GigaChatClient()
    assert client._authorization_header() == "Basic abc123"


def test_access_token_is_cached(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "abc123")
    client = GigaChatClient()
    with patch("src.ai.time.time", return_value=1_700_000_000), patch(
        "src.ai.requests.post", return_value=token_response(expires_at=1_700_001_800_000)
    ) as post:
        assert client._get_access_token() == "token-1"
        assert client._get_access_token() == "token-1"
        assert post.call_count == 1


def test_millisecond_expiry_is_normalized(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "abc123")
    client = GigaChatClient()
    with patch("src.ai.time.time", return_value=1_700_000_000), patch(
        "src.ai.requests.post", return_value=token_response(expires_at=1_700_001_800_000)
    ):
        client._get_access_token()
    assert client._cached_token_expires_at == 1_700_001_800


def test_expired_token_is_refreshed(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "abc123")
    client = GigaChatClient()
    first = token_response("token-1", 1_700_000_100_000)
    second = token_response("token-2", 1_700_001_900_000)
    with patch("src.ai.requests.post", side_effect=[first, second]) as post:
        with patch("src.ai.time.time", return_value=1_700_000_000):
            assert client._get_access_token() == "token-1"
        with patch("src.ai.time.time", return_value=1_700_000_050):
            assert client._get_access_token() == "token-2"
        assert post.call_count == 2


def test_401_invalidates_token_and_retries_once(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "abc123")
    client = GigaChatClient()
    client._cached_token = "old-token"
    client._cached_token_expires_at = 9_999_999_999

    unauthorized = api_response(status=401)
    ok = api_response(status=200, payload={"data": [{"id": "GigaChat-2-Max"}]})

    with patch("src.ai.requests.post", return_value=token_response("new-token")), patch(
        "src.ai.requests.request", side_effect=[unauthorized, ok]
    ) as request:
        models = client.check_connection()

    assert models == ["GigaChat-2-Max"]
    assert request.call_count == 2
    assert client._cached_token == "new-token"


def test_model_is_read_from_environment(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "abc123")
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-2")
    client = GigaChatClient()
    assert client.model == "GigaChat-2"
