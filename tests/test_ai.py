import pytest

from src.ai import GigaChatClient, GigaChatConfigurationError


def clear_gigachat_env(monkeypatch):
    for name in [
        "GIGACHAT_AUTH_KEY",
        "GIGACHAT_ACCESS_TOKEN",
        "GC_TOKEN",
        "GIGACHAT_MODEL",
        "GIGACHAT_BASE_URL",
        "GIGACHAT_AUTH_URL",
        "GIGACHAT_VERIFY_SSL",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_client_requires_credentials(monkeypatch):
    clear_gigachat_env(monkeypatch)
    with pytest.raises(GigaChatConfigurationError):
        GigaChatClient()


def test_legacy_gc_token_is_supported(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GC_TOKEN", "legacy-token")
    client = GigaChatClient()
    assert client.credential_mode == "access_token"
    assert client._token() == "legacy-token"
    assert client.model == "GigaChat-2-Max"


def test_auth_key_uses_oauth_mode(monkeypatch):
    clear_gigachat_env(monkeypatch)
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "base64-key")
    client = GigaChatClient()
    assert client.credential_mode == "oauth"
    assert client.auth_key == "base64-key"
