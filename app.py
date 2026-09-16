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
LOGO_PATH = BASE_DIR / "assets" / "socialnavigator-ai-logo.png"
BRAND_NAVY = "#083B73"
BRAND_BLUE = "#0B78C4"
BRAND_CYAN = "#05BFD1"
BRAND_PALE = "#EAF6FB"
STATUS_COLORS = {
    "В норме": "#1B9E77",
    "Требует внимания": "#F2A900",
    "Высокий риск": "#D64545",
}

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="СоцНавигаторAI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _bridge_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets as env vars for the framework-agnostic client."""
    names = [
        "GIGACHAT_AUTH_KEY",
        "GIGACHAT_SCOPE",
        "GIGACHAT_MODEL",
        "GIGACHAT_BASE_URL",
        "GIGACHAT_AUTH_URL",
        "GIGACHAT_VERIFY_SSL",
        "AI_TIMEOUT_SECONDS",
    ]
    try:
        for name in names:
            if not os.getenv(name) and name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except Exception:
        # Local launch without .streamlit/secrets.toml is a normal scenario.
        return


_bridge_streamlit_secrets()

st.markdown(
    f"""
    <style>
    :root {{
      --navy: {BRAND_NAVY};
      --blue: {BRAND_BLUE};
      --cyan: {BRAND_CYAN};
      --pale: {BRAND_PALE};
    }}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        font-family: "Segoe UI", "Arial", sans-serif;
    }}
    .block-container {{ padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1450px; }}
    [data-testid="stImage"] img {{
        display: block;
        width: 100% !important;
        height: auto !important;
        max-height: 235px;
        object-fit: contain !important;
        object-position: center;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid rgba(8,59,115,.10); }}
    [data-testid="stMetric"] {{
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(11,120,196,.12);
        border-radius: 16px;
        padding: .8rem .9rem;
        box-shadow: 0 6px 22px rgba(8,59,115,.06);
    }}
    [data-testid="stMetricValue"] {{ color: var(--navy); font-size: 1.7rem; }}
    [data-testid="stMetricLabel"] {{ color: #557086; }}
    .brand-hero {{
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.55rem 1.25rem;
        border-radius: 22px;
        background: linear-gradient(118deg, #F8FCFF 0%, #EAF6FB 52%, #E8FBFA 100%);
        border: 1px solid rgba(11,120,196,.16);
        box-shadow: 0 10px 30px rgba(8,59,115,.07);
        margin-bottom: 1rem;
    }}
    .brand-hero:after {{
        content: "";
        position: absolute;
        width: 250px; height: 250px;
        border-radius: 50%;
        right: -80px; top: -110px;
        border: 34px solid rgba(5,191,209,.10);
    }}
    .brand-title {{ color: var(--navy); font-size: 2.15rem; font-weight: 800; margin: 0; letter-spacing: -.02em; }}
    .brand-ai {{ color: var(--cyan); }}
    .brand-subtitle {{ color: #31526E; font-size: 1.02rem; margin: .28rem 0 .8rem; }}
    .brand-slogan {{ color: var(--navy); font-weight: 650; margin: 0 0 .75rem; }}
    .chip {{
        display: inline-block; margin-right: .38rem; margin-top: .18rem;
        padding: .26rem .58rem; border-radius: 999px;
        font-size: .79rem; font-weight: 650;
        color: var(--navy); background: rgba(255,255,255,.72);
        border: 1px solid rgba(11,120,196,.16);
    }}
    .section-note {{
        border-radius: 14px; padding: .85rem 1rem;
        background: #F7FBFD; border: 1px solid rgba(11,120,196,.12);
        color: #31526E;
    }}
    .status-card {{
        border-radius: 16px; padding: .9rem 1rem;
        background: white; border: 1px solid rgba(11,120,196,.12);
        box-shadow: 0 5px 18px rgba(8,59,115,.05);
    }}
    .status-label {{ font-size: .8rem; color: #6A8194; text-transform: uppercase; letter-spacing: .04em; }}
    .status-value {{ font-size: 1.22rem; font-weight: 750; margin-top: .18rem; }}
    .brand-footer {{ text-align: center; color: #718697; font-size: .82rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(11,120,196,.10); }}
    div[data-baseweb="tab-list"] {{ gap: .25rem; }}
    button[data-baseweb="tab"] {{ border-radius: 12px 12px 0 0; }}

    /* Phone layout. Desktop keeps the styles above; these rules apply only to a
       narrow viewport and let Streamlit's sidebar stay collapsible. */
    @media (max-width: 680px) {{
        .block-container {{
            padding: .75rem .8rem 1.5rem;
            max-width: 100%;
        }}
        [data-testid="stImage"] img {{ max-height: 160px; }}
        [data-testid="stMetric"] {{
            border-radius: 13px;
            padding: .65rem .7rem;
        }}
        [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
        .brand-hero {{
            padding: 1rem;
            border-radius: 16px;
        }}
        .brand-hero:after {{ width: 170px; height: 170px; right: -95px; top: -90px; }}
        .brand-title {{ font-size: 1.6rem; line-height: 1.1; }}
        .brand-subtitle {{ font-size: .92rem; line-height: 1.45; }}
        .brand-slogan {{ font-size: .93rem; line-height: 1.35; }}
        .chip {{ font-size: .72rem; padding: .22rem .45rem; }}
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap;
            gap: .65rem;
        }}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 calc(50% - .325rem) !important;
            min-width: calc(50% - .325rem) !important;
            width: calc(50% - .325rem) !important;
        }}
        /* Charts and their legend need the whole row on a narrow screen. */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:only-child {{
            flex-basis: 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }}
        div[data-baseweb="tab-list"] {{
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
        }}
        button[data-baseweb="tab"] {{
            flex: 0 0 auto;
            padding-left: .65rem;
            padding-right: .65rem;
            white-space: nowrap;
        }}
        [data-testid="stDataFrame"] {{ border-radius: 10px; overflow-x: auto; }}
        .section-note {{ padding: .75rem .8rem; font-size: .92rem; line-height: 1.45; }}
        .brand-footer {{ margin-top: 1.25rem; font-size: .75rem; line-height: 1.45; }}
    }}
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


@st.cache_resource(show_spinner=False)
def get_gigachat_client() -> GigaChatClient:
    # One client per Streamlit process = one shared in-memory OAuth token cache.
    return GigaChatClient()


def status_card(label: str, value: str, color: str = BRAND_NAVY) -> None:
    st.markdown(
        f"""
        <div class="status-card">
          <div class="status-label">{label}</div>
          <div class="status-value" style="color:{color}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width="stretch")
    else:
        st.markdown("## 🧭 СоцНавигаторAI")
        st.caption("Курс на результат без отклонений")

    st.divider()
    st.subheader("Источник данных")
    uploaded = st.file_uploader("CSV с программами", type=["csv"])
    if uploaded is None:
        st.info("Используется демонстрационный набор")
    else:
        st.success(f"Загружен файл: {uploaded.name}")

    st.divider()
    st.subheader("GigaChat")
    st.caption("Модель: " + os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"))
    has_credentials = bool(os.getenv("GIGACHAT_AUTH_KEY"))
    st.caption("Авторизация: " + ("настроена" if has_credentials else "не настроена"))

    if st.button("Проверить подключение", width="stretch"):
        try:
            with st.spinner("Проверяю GigaChat API..."):
                models = get_gigachat_client().check_connection()
            st.success("GigaChat API доступен")
            if models:
                st.caption("Доступно моделей: " + str(len(models)))
        except (GigaChatConfigurationError, GigaChatRequestError) as exc:
            st.error(str(exc))

    st.divider()
    st.caption("MVP · Streamlit · GigaChat · Explainable risk")

brand_mark, brand_copy = st.columns([1.15, 4], gap="large", vertical_alignment="center")
with brand_mark:
    st.image(str(LOGO_PATH), width="stretch")
with brand_copy:
    st.markdown(
        """
        <div class="brand-hero">
          <div class="brand-title">СоцНавигатор<span class="brand-ai">AI</span></div>
          <div class="brand-slogan">Курс на результат без отклонений</div>
          <div class="brand-subtitle">Цифровой мониторинг социальных программ: план/факт, раннее выявление риска и управленческая аналитика через GigaChat.</div>
          <span class="chip">🧭 мониторинг курса</span>
          <span class="chip">📊 план / факт</span>
          <span class="chip">⚠️ ранний риск</span>
          <span class="chip">✨ GigaChat-аналитик</span>
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

st.caption("Показатели рассчитаны на текущую дату. Риск-индикатор объяснимый и не является ML-моделью.")

tab_overview, tab_programs, tab_ai, tab_data = st.tabs(
    ["📊 Обзор портфеля", "🧭 Карточка программы", "✨ GigaChat-аналитик", "🗂 Данные"]
)

with tab_overview:
    left, right = st.columns([1.55, 1])
    chart_df = data.copy()
    chart_df["Результат, %"] = (chart_df["delivery_progress"] * 100).round(1)
    chart_df["Ход периода, %"] = (chart_df["schedule_progress"] * 100).round(1)

    with left:
        fig = px.bar(
            chart_df,
            x="name",
            y=["Результат, %", "Ход периода, %"],
            barmode="group",
            color_discrete_sequence=[BRAND_BLUE, "#A9D8EA"],
            title="Результат относительно хода реализации",
        )
        fig.update_layout(
            legend_title_text="Показатель",
            xaxis_title="",
            yaxis_title="%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        fig.update_xaxes(tickangle=-18, gridcolor="rgba(8,59,115,.05)")
        fig.update_yaxes(gridcolor="rgba(8,59,115,.08)")
        st.plotly_chart(fig, width="stretch")

    with right:
        status_counts = data["status"].value_counts().rename_axis("Статус").reset_index(name="Количество")
        fig_status = px.pie(
            status_counts,
            names="Статус",
            values="Количество",
            hole=0.60,
            color="Статус",
            color_discrete_map=STATUS_COLORS,
            title="Портфель по уровню риска",
        )
        fig_status.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=55, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig_status, width="stretch")

    st.subheader("Приоритет внимания")
    risk_view = data[["name", "region", "status", "risk_score", "delivery_progress", "schedule_progress", "budget_progress"]].copy()
    risk_view["Результат, %"] = (risk_view.pop("delivery_progress") * 100).round(1)
    risk_view["Ход периода, %"] = (risk_view.pop("schedule_progress") * 100).round(1)
    risk_view["Бюджет, %"] = (risk_view.pop("budget_progress") * 100).round(1)
    risk_view = risk_view.rename(
        columns={"name": "Программа", "region": "Регион", "status": "Статус", "risk_score": "Риск-балл"}
    ).sort_values(["Риск-балл", "Результат, %"], ascending=[False, True])
    st.dataframe(
        risk_view,
        width="stretch",
        hide_index=True,
        column_config={
            "Результат, %": st.column_config.ProgressColumn("Результат", min_value=0, max_value=100, format="%.0f%%"),
            "Ход периода, %": st.column_config.ProgressColumn("Ход периода", min_value=0, max_value=100, format="%.0f%%"),
            "Бюджет, %": st.column_config.ProgressColumn("Бюджет", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

with tab_programs:
    selected_name = st.selectbox("Выберите программу", data["name"].tolist(), key="program_details")
    row = data.loc[data["name"] == selected_name].iloc[0]
    status_color = STATUS_COLORS.get(str(row["status"]), BRAND_NAVY)

    st.subheader(selected_name)
    st.caption(f"{row['region']} · {row['start_date'].date().isoformat()} — {row['end_date'].date().isoformat()}")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        status_card("Статус", str(row["status"]), status_color)
    with s2:
        status_card("Фактический результат", f"{row['delivery_progress'] * 100:.0f}%", BRAND_BLUE)
    with s3:
        status_card("Ход периода", f"{row['schedule_progress'] * 100:.0f}%", BRAND_NAVY)
    with s4:
        status_card("Риск-балл", str(int(row["risk_score"])), status_color)

    st.write("")
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
        color="Показатель",
        color_discrete_sequence=[BRAND_NAVY, BRAND_BLUE, BRAND_CYAN, "#A9D8EA"],
        range_y=[0, max(110, detail["Выполнение, %"].max() + 10)],
    )
    fig_detail.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
    )
    fig_detail.update_yaxes(gridcolor="rgba(8,59,115,.08)")
    st.plotly_chart(fig_detail, width="stretch")

    lag_pp = float(row["lag"]) * 100
    budget_gap_pp = float(row["budget_efficiency_gap"]) * 100
    st.markdown(
        f"""
        <div class="section-note">
        <b>Навигация по отклонениям.</b> Разрыв относительно хода периода: <b>{lag_pp:+.1f} п.п.</b>.
        Разрыв «бюджет — результат»: <b>{budget_gap_pp:+.1f} п.п.</b>.
        Риск рассчитывается прозрачными правилами по сроку, KPI, получателям и бюджету — его можно проверить по исходным данным.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_ai:
    ai_program = st.selectbox("Программа для анализа", data["name"].tolist(), key="program_ai")
    ai_row = data.loc[data["name"] == ai_program].iloc[0]
    context = program_context(ai_row)
    program_id = str(ai_row["program_id"])

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Статус", ai_row["status"])
    a2.metric("Результат", f"{ai_row['delivery_progress'] * 100:.0f}%")
    a3.metric("Ход периода", f"{ai_row['schedule_progress'] * 100:.0f}%")
    a4.metric("Бюджет", f"{ai_row['budget_progress'] * 100:.0f}%")

    st.info("В GigaChat передаётся только агрегированная карточка выбранной программы — без персональных данных граждан.")

    if st.button("✨ Сформировать аналитическую записку", type="primary", width="stretch"):
        try:
            with st.spinner("GigaChat анализирует показатели и отклонения..."):
                response = get_gigachat_client().chat(analysis_messages(context))
            results = st.session_state.setdefault("analysis_results", {})
            results[program_id] = {
                "text": response.text,
                "meta": f"{response.provider} · {response.model}",
            }
        except (GigaChatConfigurationError, GigaChatRequestError) as exc:
            st.error(str(exc))

    result = st.session_state.get("analysis_results", {}).get(program_id)
    if result:
        st.markdown("### Аналитическая записка")
        st.markdown(result["text"])
        st.caption(result["meta"])
        st.download_button(
            "Скачать записку",
            data=result["text"].encode("utf-8"),
            file_name=f"socnavigator_{program_id}_analysis.md",
            mime="text/markdown",
            width="stretch",
        )

    st.divider()
    st.markdown("#### Быстрые вопросы для демонстрации")
    q1, q2, q3 = st.columns(3)
    if q1.button("Почему программа в зоне риска?", width="stretch"):
        st.session_state["ai_question"] = "Почему программа находится в текущей зоне риска? Назови показатели, которые сильнее всего на это влияют."
    if q2.button("Есть ли дисбаланс бюджета?", width="stretch"):
        st.session_state["ai_question"] = "Есть ли дисбаланс между освоением бюджета и фактическим результатом? Объясни по цифрам."
    if q3.button("Что проверить руководителю?", width="stretch"):
        st.session_state["ai_question"] = "Какие три вещи руководителю стоит проверить в первую очередь по этой программе?"

    question = st.text_input(
        "Или задайте свой вопрос",
        key="ai_question",
        placeholder="Например: за счёт чего сформировалось отклонение?",
    )
    if st.button("Спросить GigaChat", disabled=not question.strip(), width="stretch"):
        try:
            with st.spinner("GigaChat формирует ответ..."):
                response = get_gigachat_client().chat(question_messages(context, question.strip()))
            st.markdown("### Ответ")
            st.markdown(response.text)
            st.caption(f"{response.provider} · {response.model}")
        except (GigaChatConfigurationError, GigaChatRequestError) as exc:
            st.error(str(exc))

with tab_data:
    st.subheader("Исходные данные")
    st.caption("Данные демонстрационные. Для своего сценария можно загрузить CSV того же формата через боковую панель.")
    st.dataframe(source, width="stretch", hide_index=True)
    st.download_button(
        "Скачать демонстрационный CSV",
        data=(BASE_DIR / "data" / "sample_programs.csv").read_bytes(),
        file_name="sample_programs.csv",
        mime="text/csv",
    )
    st.caption(
        "Обязательные столбцы: program_id, name, region, budget_plan, budget_actual, beneficiaries_plan, "
        "beneficiaries_actual, kpi_plan, kpi_actual, start_date, end_date"
    )

st.markdown(
    """
    <div class="brand-footer">
      СоцНавигаторAI · Курс на результат без отклонений · демонстрационный MVP
    </div>
    """,
    unsafe_allow_html=True,
)
