import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="StrokePrediction.AI", page_icon="🧠", layout="wide")

gender_map = {"Мужской": "Male", "Женский": "Female"}
married_map = {"В браке": "Yes", "Свободен(-на)": "No"}
work_map = {
    "Наёмный труд": "Private",
    "Самозанятый / ИП / Фриланс": "Self-employed",
    "Госсектор / Бюджетник": "Govt_job",
    "Ребёнок": "children",
    "Никогда не работал(а)": "Never_worked"
}
residence_map = {"Город": "Urban", "Село": "Rural"}
smoke_map = {
    "Никогда не курил(а)": "never smoked",
    "Бросил(а) курить": "formerly smoked",
    "Курю в настоящее время": "smokes"
}

HISTORY_FILE = "data/history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def local_css():
    st.markdown("""
    <style>
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #ccdaee 50%,  #ffffff 100%);
        padding: 100px 20px;
        border-bottom: 1px solid #e0e4e8;
        margin-top: -60px;
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center !important;
    }
    .hero h1 {
        font-size: 4rem;
        font-weight: 800;
        color: #1a2639;
        margin-bottom: 10px;
        letter-spacing: -1px;
        text-align: center !important;
        width: 100%;
    }
    .hero p {
        font-size: 1.3rem;
        color: #2980b9;
        font-weight: 500;
        text-align: center !important;
        width: 100%;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff !important;
        padding: 30px 40px !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
        border: 1px solid #e0e4e8 !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background-color: #2980b9 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #1a2639 !important;
        transform: translateY(-2px);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2c3e50;
        color: #bdc3c7;
        text-align: center;
        padding: 12px;
        font-size: 0.85rem;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    h2, h3 { color: #1a2639 !important; }
    .stApp { padding-bottom: 60px; }
    </style>
    """, unsafe_allow_html=True)

local_css()

@st.cache_resource
def load_model():
    return joblib.load('models/stroke_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.markdown("""
<div class="hero">
    <h1>StrokePrediction.AI</h1>
    <p>Модель логистической регрессии для оценки риска инсульта</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = load_history()

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if st.session_state.history:
        st.subheader("📋 История ваших тестирований")
        hist_df = pd.DataFrame(st.session_state.history[-5:][::-1])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.write("")

    with st.form("patient_data"):
        st.info("🔒 **Гарантируется конфиденциальность данных.** Мы не сохраняем ваши персональные данные.")
        st.subheader("Заполните данные пациента")

        age = st.number_input("Возраст (лет)", min_value=1, max_value=105, value=50, step=1)

        gender_ru = st.selectbox("Пол", list(gender_map.keys()))
        married_ru = st.selectbox("Семейное положение", list(married_map.keys()))

        c1, c2 = st.columns(2)
        with c1:
            work_ru = st.selectbox(
                "Тип занятости",
                list(work_map.keys()),
                help="Если возраст меньше 14 лет, тип занятости автоматически станет «Ребёнок». "
                     "Если возраст 14 лет и более, «Ребёнок» заменится на «Никогда не работал(а)»")
        with c2:
            residence_ru = st.selectbox("Место жительства", list(residence_map.keys()))

        c3, c4 = st.columns(2)
        with c3:
            height = st.number_input("Рост (см)", min_value=100, max_value=250, value=170, step=1)
        with c4:
            weight = st.number_input("Вес (кг)", min_value=20, max_value=300, value=70, step=1)

        avg_glucose = st.number_input("Средний уровень глюкозы в крови (мг/дл)", min_value=50.0, max_value=300.0,
                                      value=100.0, step=0.1)
        smoke_ru = st.selectbox("Статус курения", list(smoke_map.keys()))

        st.write("**Хронические заболевания:**")
        c5, c6 = st.columns(2)
        with c5:
            hypertension = st.checkbox("Гипертония")
        with c6:
            heart_disease = st.checkbox("Болезни сердца")

        st.write("")
        submitted = st.form_submit_button("Получить прогноз")

if submitted:
    bmi = weight / ((height / 100) ** 2)

    if bmi < 10 or bmi > 80:
        w1, w2, w3 = st.columns([1, 2, 1])
        with w2:
            st.warning(
                "⚠️ Внимание: Введенные параметры роста и веса указывают на экстремальные показатели ИМТ. Прогноз может быть неточным.")

    if age < 14:
        actual_work_ru = "Ребёнок"
    else:
        actual_work_ru = work_ru if work_ru != "Ребёнок" else "Никогда не работал(а)"

    if actual_work_ru != work_ru:
        w1, w2, w3 = st.columns([1, 2, 1])
        with w2:
            if age < 14:
                st.info("ℹ️ Возраст меньше 14 лет — тип занятости автоматически изменён на «Ребёнок».")
            else:
                st.info("ℹ️ Возраст 14 лет и старше — тип занятости «Ребёнок» заменён на «Никогда не работал(а)».")

    input_data = pd.DataFrame([{
        'id': 0,
        'gender': gender_map[gender_ru],
        'age': age,
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'ever_married': married_map[married_ru],
        'work_type': work_map[actual_work_ru],
        'Residence_type': residence_map[residence_ru],
        'avg_glucose_level': avg_glucose,
        'bmi': round(bmi, 1),
        'smoking_status': smoke_map[smoke_ru]
    }])

    proba = model.predict_proba(input_data)[0][1]
    proba_percent = proba * 100

    new_entry = {
        "Возраст": age,
        "Пол": gender_ru,
        "Семейное положение": married_ru,
        "Тип занятости": actual_work_ru,
        "Место жительства": residence_ru,
        "Рост (см)": height,
        "Вес (кг)": weight,
        "ИМТ": round(bmi, 1),
        "Глюкоза": avg_glucose,
        "Курение": smoke_ru,
        "Гипертония": "Да" if hypertension else "Нет",
        "Болезни сердца": "Да" if heart_disease else "Нет",
        "Риск (%)": f"{proba_percent:.1f}%"
    }

    st.session_state.history.append(new_entry)
    save_history(st.session_state.history)

    st.divider()

    res_left, res_center, res_right = st.columns([1, 2.5, 1])

    with res_center:
        st.subheader("📊 Результат анализа")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_percent,
            number={'suffix': "%", 'font': {'size': 50, 'color': '#1a2639'}},
            title={'text': "Вероятность инсульта", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2980b9" if proba_percent < 50 else "#e74c3c"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#e8f5e9'},
                    {'range': [50, 75], 'color': '#fff9c4'},
                    {'range': [75, 100], 'color': '#ffebee'}
                ]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        if proba_percent < 50:
            st.success(
                "✅ **Риск низкий.** Ваши показатели в пределах нормы. Рекомендуется проходить профилактический осмотр раз в год.")
        elif proba_percent < 75:
            st.warning(
                "⚠️ **Риск повышен.** Рекомендуется обратиться к врачу-терапевту в ближайшее время для дополнительного обследования.")
        else:
            st.error(
                "🚨 **Высокий риск!** Рекомендуется обратиться к врачу (неврологу/кардиологу) **как можно скорее** для детальной диагностики.")

        st.divider()

        st.subheader("Ваши индивидуальные факторы риска")
        st.markdown(
            "*Показан относительный вклад каждого фактора в общую картину. Красные повышают риск, зеленые — снижают.*")

        clf = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']

        X_trans = preprocessor.transform(input_data)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = X_trans.flatten()

        feature_names = preprocessor.get_feature_names_out()
        coefs = clf.coef_[0]

        contributions = X_trans * coefs

        feature_ru = {
            'num__age': 'Возраст',
            'num__avg_glucose_level': 'Глюкоза',
            'num__bmi': 'ИМТ',
            'cat__gender_Male': 'Пол: Мужской',
            'cat__gender_Other': 'Пол: Другой',
            'cat__hypertension_1': 'Гипертония (есть)',
            'cat__heart_disease_1': 'Болезни сердца (есть)',
            'cat__ever_married_Yes': 'В браке',
            'cat__work_type_Private': 'Работа: Наёмный труд',
            'cat__work_type_Self-employed': 'Работа: Самозанятый',
            'cat__work_type_Govt_job': 'Работа: Госсектор',
            'cat__work_type_children': 'Работа: Ребёнок',
            'cat__work_type_Never_worked': 'Работа: Не работал',
            'cat__Residence_type_Urban': 'Живет в городе',
            'cat__smoking_status_formerly smoked': 'Курил(а) в прошлом',
            'cat__smoking_status_never smoked': 'Никогда не курил(а)',
            'cat__smoking_status_smokes': 'Курит сейчас',
            'cat__smoking_status_Unknown': 'Курение: Неизвестно'
        }

        df_contrib = pd.DataFrame({'Feature': feature_names, 'Contribution': contributions})

        df_contrib = df_contrib[
            (df_contrib['Contribution'].abs() > 0.01) & (df_contrib['Feature'] != 'cat__work_type_children')]

        df_contrib['Feature_RU'] = df_contrib['Feature'].apply(lambda x: feature_ru.get(x, x.split('__')[-1]))

        total_impact = df_contrib['Contribution'].abs().sum()
        if total_impact > 0:
            df_contrib['SignedPercent'] = (df_contrib['Contribution'] / total_impact) * 100
        else:
            df_contrib['SignedPercent'] = 0

        df_contrib = df_contrib.sort_values(by='SignedPercent', ascending=True)

        colors = ['#e74c3c' if x > 0 else '#27ae60' for x in df_contrib['SignedPercent']]

        fig_imp = go.Figure(go.Bar(
            x=df_contrib['SignedPercent'],
            y=df_contrib['Feature_RU'],
            orientation='h',
            marker_color=colors,
            text=df_contrib['SignedPercent'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside'
        ))
        fig_imp.update_layout(
            xaxis_title="Влияние на риск (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        st.subheader("Прогноз риска на 5 лет вперед")

        future_ages = [age + i for i in range(6)]
        labels = [f"{a} лет" for a in future_ages]

        probs_current = []
        probs_quit = []

        input_current = input_data.copy()
        input_quit = input_data.copy()
        input_quit.loc[0, 'smoking_status'] = 'formerly smoked'

        for a in future_ages:
            input_current.loc[0, 'age'] = a
            probs_current.append(model.predict_proba(input_current)[0][1] * 100)

            input_quit.loc[0, 'age'] = a
            probs_quit.append(model.predict_proba(input_quit)[0][1] * 100)

        fig_future = go.Figure()

        fig_future.add_trace(go.Bar(
            x=labels, y=probs_current,
            name='Текущий образ жизни',
            marker_color='#2980b9',
            text=[f"{p:.1f}%" for p in probs_current],
            textposition='auto'
        ))

        if smoke_ru == "Курю в настоящее время":
            fig_future.add_trace(go.Bar(
                x=labels, y=probs_quit,
                name='Если бросить курить',
                marker_color='#2ecc71',
                text=[f"{p:.1f}%" for p in probs_quit],
                textposition='auto'
            ))
            title_text = "Ваш риск: сейчас vs отказ от курения"
        else:
            title_text = "Как изменится ваш риск с возрастом"

        fig_future.update_layout(
            title=title_text,
            xaxis_title="Возраст пациента",
            yaxis_title="Вероятность инсульта (%)",
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_future, use_container_width=True)

        if smoke_ru == "Курю в настоящее время":
            st.info("💡 На графике показано, как отказ от курения может повлиять на снижение риска в ближайшие годы.")

st.markdown("""
<div class="footer">
    Сгенерировано ИИ. <b>Не является действительным медицинским средством.</b> Обязательно проконсультируйтесь с врачом.
</div>
""", unsafe_allow_html=True)