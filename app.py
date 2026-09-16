from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.ai import GigaChatClient, GigaChatConfigurationError, GigaChatRequestError
from src.analytics import enrich_programs, portfolio_metrics, program_context, validate_programs
from src.prompts import analysis_messages, question_messages

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="СоцПульс AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(89,72,210,.14), rgba(0,160,170,.10));
        border: 1px solid rgba(120,120,140,.20);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .35rem 0 0; opacity: .75; }
    .hint {
        border-radius: 12px;
        padding: .75rem 1rem;
        background: rgba(120,120,140,.08);
        border: 1px solid rgba(120,120,140,.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} млрд ₽"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн ₽"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f} тыс. ₽"
    return f"{value:.0f} ₽"


def load_source(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.read_csv(BASE_DIR / "data" / "sample_programs.csv")
    return pd.read_csv(uploaded)


with st.sidebar:
    st.subheader("Источник данных")
    uploaded = st.file_uploader("CSV с программами", type=["csv"])
    st.caption("Если файл не загружен, используется демонстрационный набор.")

    st.divider()
    st.subheader("GigaChat")
    st.caption("Модель: " + os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"))
    has_credentials = bool(
        os.getenv("GIGACHAT_AUTH_KEY")
        or os.getenv("GIGACHAT_ACCESS_TOKEN")
        or os.getenv("GC_TOKEN")
    )
    st.caption("Доступ: " + ("настроен" if has_credentials else "не настроен"))
    if st.button("Проверить подключение", use_container_width=True):
        try:
            client = GigaChatClient()
            with st.spinner("Проверяю GigaChat API..."):
                models = client.check_connection()
            st.success("GigaChat API доступен")
            if models:
                st.caption("Доступные модели: " + ", ".join(models[:6]))
        except (GigaChatConfigurationError, GigaChatRequestError) as exc:
            st.error(str(exc))

    st.divider()
    st.caption("MVP: дашборд + риск-индикатор + GigaChat-аналитик")

st.markdown(
    """
    <div class="hero">
      <h1>СоцПульс AI</h1>
      <p>Мониторинг социальных программ: план/факт, раннее выявление рисков и аналитические записки через GigaChat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    source = load_source(uploaded)
    missing = validate_programs(source)
    if missing:
        st.error("В CSV не хватает обязательных столбцов: " + ", ".join(missing))
        st.stop()
    data = enrich_programs(source)
except Exception as exc:
    st.error(f"Не удалось обработать данные: {exc}")
    st.stop()

metrics = portfolio_metrics(data)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Программ", metrics["programs"])
c2.metric("Высокий риск", metrics["high_risk"])
c3.metric("Требуют внимания", metrics["attention"])
c4.metric("Освоено", money(metrics["budget_actual"]))
c5.metric("Средний результат", f"{metrics['avg_delivery'] * 100:.0f}%")

tab_overview, tab_programs, tab_ai, tab_data = st.tabs(
    ["Обзор", "Программы", "GigaChat-аналитик", "Данные"]
)

with tab_overview:
    left, right = st.columns([1.5, 1])
    chart_df = data.copy()
    chart_df["Результат, %"] = (chart_df["delivery_progress"] * 100).round(1)
    chart_df["Плановый ход, %"] = (chart_df["schedule_progress"] * 100).round(1)
    with left:
        fig = px.bar(
            chart_df,
            x="name",
            y=["Результат, %", "Плановый ход, %"],
            barmode="group",
            labels={"value": "%", "name": "Программа"},
            title="Фактический результат относительно хода реализации",
        )
        fig.update_layout(legend_title_text="Показатель", xaxis_title="", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        status_counts = data["status"].value_counts().rename_axis("Статус").reset_index(name="Количество")
        fig_status = px.pie(
            status_counts,
            names="Статус",
            values="Количество",
            hole=0.55,
            title="Портфель по уровню риска",
        )
        st.plotly_chart(fig_status, use_container_width=True)

    risk_view = data[["name", "region", "status", "risk_score", "delivery_progress", "budget_progress"]].copy()
    risk_view["Результат"] = (risk_view.pop("delivery_progress") * 100).round(1)
    risk_view["Бюджет"] = (risk_view.pop("budget_progress") * 100).round(1)
    risk_view = risk_view.rename(
        columns={"name": "Программа", "region": "Регион", "status": "Статус", "risk_score": "Риск-балл"}
    )
    st.dataframe(
        risk_view.sort_values(["Риск-балл", "Результат"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Результат": st.column_config.ProgressColumn("Результат", min_value=0, max_value=100, format="%.0f%%"),
            "Бюджет": st.column_config.ProgressColumn("Бюджет", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

with tab_programs:
    selected_name = st.selectbox("Выберите программу", data["name"].tolist(), key="program_details")
    row = data.loc[data["name"] == selected_name].iloc[0]

    st.subheader(selected_name)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Статус", row["status"])
    s2.metric("Результат", f"{row['delivery_progress'] * 100:.0f}%")
    s3.metric("Ход периода", f"{row['schedule_progress'] * 100:.0f}%")
    s4.metric("Риск-балл", int(row["risk_score"]))

    b1, b2, b3 = st.columns(3)
    b1.metric("Бюджет", money(row["budget_actual"]), f"из {money(row['budget_plan'])}", delta_color="off")
    b2.metric(
        "Получатели",
        f"{int(row['beneficiaries_actual']):,}".replace(",", " "),
        f"из {int(row['beneficiaries_plan']):,}".replace(",", " "),
        delta_color="off",
    )
    b3.metric("KPI", f"{row['kpi_actual']:.0f}", f"из {row['kpi_plan']:.0f}", delta_color="off")

    detail = pd.DataFrame(
        {
            "Показатель": ["Бюджет", "Получатели", "KPI", "Ход периода"],
            "Выполнение, %": [
                row["budget_progress"] * 100,
                row["beneficiaries_progress"] * 100,
                row["kpi_progress"] * 100,
                row["schedule_progress"] * 100,
            ],
        }
    )
    fig_detail = px.bar(
        detail,
        x="Показатель",
        y="Выполнение, %",
        range_y=[0, max(110, detail["Выполнение, %"].max() + 10)],
    )
    st.plotly_chart(fig_detail, use_container_width=True)

    st.markdown(
        """
        <div class="hint">
        <b>Как считается риск в MVP:</b> система сравнивает темп достижения результата с темпом прохождения периода,
        а также проверяет разрыв между освоением бюджета и фактическим результатом. Это прозрачный детерминированный
        индикатор, а не ML-модель. Для пилота такой подход проще объяснить и проверить.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_ai:
    ai_program = st.selectbox("Программа для анализа", data["name"].tolist(), key="program_ai")
    ai_row = data.loc[data["name"] == ai_program].iloc[0]
    context = program_context(ai_row)

    st.caption("В GigaChat передается только агрегированная карточка выбранной программы — без персональных данных.")

    if st.button("Сформировать аналитическую записку", type="primary", use_container_width=True):
        try:
            client = GigaChatClient()
            with st.spinner("GigaChat анализирует показатели..."):
                response = client.chat(analysis_messages(context))
            st.session_state["analysis_result"] = response.text
            st.session_state["analysis_meta"] = f"{response.provider} · {response.model}"
        except (GigaChatConfigurationError, GigaChatRequestError) as exc:
            st.error(str(exc))

    if st.session_state.get("analysis_result"):
        st.markdown(st.session_state["analysis_result"])
        st.caption(st.session_state.get("analysis_meta", ""))
        st.download_button(
            "Скачать записку",
            data=st.session_state["analysis_result"].encode("utf-8"),
            file_name="socpulse_analysis.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    question = st.text_input(
        "Задайте вопрос по выбранной программе",
        placeholder="Почему программа попала в зону риска?",
    )
    if st.button("Спросить GigaChat", disabled=not question.strip()):
        try:
            client = GigaChatClient()
            with st.spinner("GigaChat формирует ответ..."):
                response = client.chat(question_messages(context, question.strip()))
            st.markdown(response.text)
            st.caption(f"{response.provider} · {response.model}")
        except (GigaChatConfigurationError, GigaChatRequestError) as exc:
            st.error(str(exc))

with tab_data:
    st.subheader("Исходные данные")
    st.dataframe(source, use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать демонстрационный CSV",
        data=(BASE_DIR / "data" / "sample_programs.csv").read_bytes(),
        file_name="sample_programs.csv",
        mime="text/csv",
    )
    st.caption(
        "Обязательные столбцы: "
        "program_id, name, region, budget_plan, budget_actual, beneficiaries_plan, beneficiaries_actual, "
        "kpi_plan, kpi_actual, start_date, end_date"
    )
