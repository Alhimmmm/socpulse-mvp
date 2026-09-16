$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Создан .env. Укажите GIGACHAT_AUTH_KEY и запустите скрипт еще раз." -ForegroundColor Yellow
    exit 0
}

streamlit run app.py
