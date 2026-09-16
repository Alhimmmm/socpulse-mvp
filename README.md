# СоцПульс AI — MVP на GigaChat

Демонстрационный сервис для мониторинга и оценки социальных программ. MVP показывает управленческий дашборд, автоматически подсвечивает риск программ и использует **GigaChat** как ИИ-аналитика.

## Сценарий демонстрации

1. Открыть портфель социальных программ.
2. Увидеть программы с отклонениями по сроку, результату и бюджету.
3. Выбрать проблемную программу.
4. Нажать **«Сформировать аналитическую записку»**.
5. GigaChat получает только агрегированные показатели программы и формирует краткий управленческий вывод.
6. Руководитель может задать дополнительный вопрос по выбранной программе.

## Что входит в MVP

- демонстрационный CSV или загрузка своего CSV;
- общий дашборд портфеля;
- план/факт бюджета, получателей и KPI;
- прозрачный детерминированный риск-индикатор;
- карточка программы;
- аналитическая записка через GigaChat;
- вопрос к GigaChat по выбранной программе;
- проверка подключения к GigaChat API из интерфейса;
- Dockerfile и unit-тесты.

В MVP намеренно нет авторизации пользователей, PostgreSQL, сложного ML и интеграций с ведомственными ИС. Цель — законченный демонстрационный пользовательский сценарий, который можно защитить за один день.

## Стек

- Python 3.11+
- Streamlit
- pandas
- Plotly
- GigaChat REST API

## GigaChat

По умолчанию используется модель `GigaChat-2-Max`.

Текущий целевой API URL:

```text
https://api.giga.chat/v1
```

Для получения access token используется OAuth endpoint:

```text
https://ngw.devices.sberbank.ru:9443/api/v2/oauth
```

### Вариант 1 — Authorization key

Создайте `.env` из `.env.example` и заполните:

```env
GIGACHAT_AUTH_KEY=ваш_ключ_авторизации
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat-2-Max
GIGACHAT_VERIFY_SSL=false
```

Приложение само получает access token и кэширует его до истечения срока действия.

### Вариант 2 — совместимость с нашей прежней настройкой

Ранее мы использовали уже выданный токен (`GC_TOKEN`). Этот режим сохранён:

```env
GC_TOKEN=ваш_access_token
GIGACHAT_MODEL=GigaChat-2-Max
GIGACHAT_VERIFY_SSL=false
```

Можно также использовать более явное имя `GIGACHAT_ACCESS_TOKEN`.

> Access token GigaChat короткоживущий. Для постоянной работы предпочтительнее `GIGACHAT_AUTH_KEY`.

## Запуск

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# заполнить .env
streamlit run app.py
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# заполнить .env
streamlit run app.py
```

Открыть: `http://localhost:8501`.

## Docker

```bash
docker build -t socpulse-mvp .
docker run --rm -p 8501:8501 --env-file .env socpulse-mvp
```

## CSV

Обязательные столбцы:

```text
program_id,name,region,budget_plan,budget_actual,beneficiaries_plan,beneficiaries_actual,kpi_plan,kpi_actual,start_date,end_date
```

Даты: `YYYY-MM-DD`.

## Как считается риск

MVP сравнивает долю прошедшего срока программы, фактическое достижение получателей и KPI, освоение бюджета и разрыв между расходованием средств и фактическим результатом.

На выходе три статуса:

- `В норме`;
- `Требует внимания`;
- `Высокий риск`.

Это намеренно объяснимый rule-based индикатор. GigaChat не определяет риск сам — он объясняет уже рассчитанные показатели и помогает руководителю интерпретировать их.

## Тесты

```bash
pytest -q
```
