import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# Настройка страницы сайта
st.set_page_config(page_title="FIFA Predictor", layout="centered")
st.title("⚽ Бесплатный Предсказатель Матчей FIFA")

# 1. Загрузка бесплатных данных из открытого GitHub (Английская Премьер-Лига)
@st.cache_data
def load_data():
    url = "https://githubusercontent.com"
    df = pd.read_csv(url)
    df = df.rename(columns={'Team 1': 'Home_Team', 'Team 2': 'Away_Team', 'FT': 'Score'})
    df[['Home_Goals', 'Away_Goals']] = df['Score'].str.split('-', expand=True).astype(int)
    return df[['Home_Team', 'Away_Team', 'Home_Goals', 'Away_Goals']]

try:
    df = load_data()
    teams = np.unique(np.concatenate([df['Home_Team'], df['Away_Team']]))
    n_teams = len(teams)
    team_map = {team: i for i, team in enumerate(teams)}

    # 2. Математическая модель (Пуассон)
    def loss_function(params, df, team_map, n_teams):
        home_adv = params[0]
        att_skills = params[1:1+n_teams]
        def_skills = params[1+n_teams:1+2*n_teams]
        log_lik = 0
        for _, row in df.iterrows():
            h_idx = team_map[row['Home_Team']]
            a_idx = team_map[row['Away_Team']]
            lambda_home = np.exp(home_adv + att_skills[h_idx] - def_skills[a_idx])
            lambda_away = np.exp(att_skills[a_idx] - def_skills[h_idx])
            log_lik += poisson.logpmf(row['Home_Goals'], lambda_home)
            log_lik += poisson.logpmf(row['Away_Goals'], lambda_away)
        return -log_lik

    init_params = np.zeros(1 + 2 * n_teams)
    res = minimize(loss_function, init_params, args=(df, team_map, n_teams), method='BFGS')
    home_adv, att, _def = res.x[0], res.x[1:1+n_teams], res.x[1+n_teams:1+2*n_teams]

    # 3. Интерфейс выбора команд на сайте
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Хозяева поля (Home)", teams, index=0)
    with col2:
        away_team = st.selectbox("Гости (Away)", teams, index=1)

    if st.button("Рассчитать прогноз матча", type="primary"):
        h_idx, a_idx = team_map[home_team], team_map[away_team]
        lambda_h = np.exp(home_adv + att[h_idx] - _def[a_idx])
        lambda_a = np.exp(att[a_idx] - _def[h_idx])
        
        # Расчет вероятностей
        h_probs = poisson.pmf(range(6), lambda_h)
        a_probs = poisson.pmf(range(6), lambda_a)
        matrix = np.outer(h_probs, a_probs)
        
        draw = np.sum(np.diag(matrix)) * 100
        home_win = np.sum(np.tril(matrix, -1)) * 100
        away_win = np.sum(np.triu(matrix, 1)) * 100

        # Вывод красивого результата
        st.subheader("📊 Шансы на победу:")
        st.write(f"🏆 Победа **{home_team}**: {home_win:.1f}%")
        st.write(f"🤝 Ничья: {draw:.1f}%")
        st.write(f"🏆 Победа **{away_team}**: {away_win:.1f}%")
        st.info(f"Ожидаемый средний счет: {lambda_h:.1f} - {lambda_a:.1f}")

except Exception as e:
    st.error("Не удалось загрузить бесплатные данные футбола. Проверьте интернет-соединение.")
