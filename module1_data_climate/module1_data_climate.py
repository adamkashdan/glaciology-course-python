"""
Модуль 1: Анализ данных и простые климатические модели
==========================================================

Курс по гляциологии — практическая реализация на Python.
На основе структуры курса и материалов проекта Penny Ice Cap GeoAgent
(радарные данные NASA Operation IceBridge, MCoRDS L2).

Содержание:
1. Анализ временных рядов потери массы (аналог данных ICESat-2)
2. Очистка и анализ радарных данных ложа ледника (аналог MCoRDS L2, pandas)
3. Модель положительных градусо-дней (PDD) для оценки абляции
4. Феноменологические (ad-hoc) модели ледниковых циклов:
   Колдер (Calder), Имбри (Imbrie), Пайар (Paillard)
5. Плейстоценовый ледяной слой (PIL): толщина и реология (enhancement factor)

Требуемые библиотеки: numpy, pandas, matplotlib, scipy
Установка: pip install numpy pandas matplotlib scipy

ВАЖНОЕ МЕТОДИЧЕСКОЕ ЗАМЕЧАНИЕ: реальные данные ICESat-2 и MCoRDS L2 не
включены в этот учебный репозиторий (требуют скачивания с NASA
Earthdata). Ниже используются СИНТЕТИЧЕСКИЕ данные, структурно и
статистически подобные реальным (тот же формат столбцов, тот же
характер шума и пропусков), чтобы отрабатывать инструментарий pandas
без необходимости доступа к внешним серверам.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

RHO_ICE = 917.0
G = 9.81
SEC_PER_YEAR = 365.25 * 24 * 3600

rng = np.random.default_rng(42)

# =============================================================================
# 1. ВРЕМЕННЫЕ РЯДЫ ПОТЕРИ МАССЫ (аналог ICESat-2 / GRACE)
# =============================================================================
#
# Синтетический аналог сводки баланса массы ледниковых щитов: линейный
# тренд потери массы + сезонный цикл (накопление зимой, абляция летом) +
# шум наблюдений. Порядки величины трендов взяты из открытых оценок
# NASA/IMBIE для Гренландии (~280 Гт/год) и Антарктиды (~125 Гт/год) —
# используются здесь только для калибровки амплитуды, а не как точные
# заявленные значения на конкретную дату.


def generate_mass_balance_timeseries(years=10, trend_Gt_per_year=-280.0,
                                       seasonal_amplitude_Gt=150.0, noise_Gt=15.0):
    """
    Генерирует синтетический месячный временной ряд аномалии массы льда
    (структурно подобный продуктам ICESat-2 / GRACE), Гт (гигатонны).
    """
    n_months = years * 12
    t_years = np.arange(n_months) / 12.0
    trend = trend_Gt_per_year * t_years
    seasonal = seasonal_amplitude_Gt * np.sin(2 * np.pi * t_years - np.pi / 2)  # минимум летом
    noise = rng.normal(0, noise_Gt, n_months)
    mass_anomaly = trend + seasonal + noise
    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")
    return pd.DataFrame({"date": dates, "mass_anomaly_Gt": mass_anomaly, "t_years": t_years})


def plot_mass_balance_timeseries(df, title="Синтетический аналог ICESat-2: аномалия массы льда"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["date"], df["mass_anomaly_Gt"], color="#1f77b4", lw=1, alpha=0.6, label="Ежемесячные данные")

    # Скользящее среднее (сглаженный тренд без сезонного цикла)
    rolling = df["mass_anomaly_Gt"].rolling(12, center=True).mean()
    ax.plot(df["date"], rolling, color="#d62728", lw=2.5, label="Скользящее среднее (12 мес.)")

    ax.set_xlabel("Дата")
    ax.set_ylabel("Аномалия массы льда, Гт")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/icesat2_mass_balance.png", dpi=150)
    plt.close(fig)

    trend_fit = np.polyfit(df["t_years"], df["mass_anomaly_Gt"], 1)
    return trend_fit[0]  # Гт/год


# =============================================================================
# 2. АНАЛИЗ РАДАРНЫХ ДАННЫХ ЛОЖА ЛЕДНИКА (аналог MCoRDS L2, pandas)
# =============================================================================
#
# MCoRDS (Multichannel Coherent Radar Depth Sounder) измеряет высоту
# поверхности и ложа ледника вдоль траектории полёта. Пропуски данных
# в реальных продуктах NASA часто кодируются значением -9999.
# Толщина льда: THICKNESS = SURFACE - BOTTOM (высоты в метрах над
# эллипсоидом), при этом BOTTOM/-9999 означает "нет данных" (например,
# из-за сильного рассеяния сигнала во влажном/деформируемом ложе).


def generate_synthetic_mcords_profile(n_points=2000, missing_frac=0.03):
    """
    Синтетический аналог траектории радарного профиля MCoRDS L2
    (структура столбцов как в реальных продуктах IceBridge).
    """
    distance_km = np.linspace(0, 80, n_points)
    surface = 1800 + 400 * np.exp(-distance_km / 30) + rng.normal(0, 5, n_points)
    thickness_true = 300 + 250 * np.sin(distance_km / 15) ** 2 + rng.normal(0, 20, n_points)
    thickness_true = np.clip(thickness_true, 20, None)
    bottom = surface - thickness_true

    df = pd.DataFrame({
        "DISTANCE_KM": distance_km,
        "ELEVATION": surface + rng.normal(0, 2, n_points),  # высота самолёта/GPS-антенны (условно)
        "SURFACE": surface,
        "BOTTOM": bottom,
    })

    # Имитация пропусков данных (сильное рассеяние, шум сигнала)
    missing_idx = rng.choice(n_points, size=int(n_points * missing_frac), replace=False)
    df.loc[missing_idx, "BOTTOM"] = -9999.0

    return df


def clean_and_analyze_mcords(df):
    """
    Очистка радарных данных MCoRDS от значений-пропусков (-9999) и
    расчёт толщины льда, как в реальном пайплайне penny-geoagent.
    """
    df = df.copy()
    df.loc[df["BOTTOM"] <= -9998, "BOTTOM"] = np.nan
    df["THICKNESS"] = df["SURFACE"] - df["BOTTOM"]

    n_missing = df["BOTTOM"].isna().sum()
    stats = {
        "n_points_total": len(df),
        "n_missing": int(n_missing),
        "missing_frac": n_missing / len(df),
        "mean_thickness_m": df["THICKNESS"].mean(),
        "max_thickness_m": df["THICKNESS"].max(),
        "surface_bottom_corr": df[["SURFACE", "BOTTOM"]].corr().iloc[0, 1],
    }
    return df, stats


def plot_mcords_profile(df):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(df["DISTANCE_KM"], df["SURFACE"], df["BOTTOM"], color="#a6cee3", alpha=0.5, label="Лёд")
    ax.plot(df["DISTANCE_KM"], df["SURFACE"], color="#1f77b4", lw=1.5, label="Поверхность")
    ax.plot(df["DISTANCE_KM"], df["BOTTOM"], color="#8c510a", lw=1.5, label="Ложе (после очистки -9999)")

    missing = df["BOTTOM"].isna()
    if missing.any():
        ax.scatter(df.loc[missing, "DISTANCE_KM"], df.loc[missing, "SURFACE"] - 50,
                   color="red", s=10, zorder=5, label="Пропуски данных (было -9999)")

    ax.set_xlabel("Расстояние вдоль траектории полёта, км")
    ax.set_ylabel("Высота, м")
    ax.set_title("Радарный профиль ложа ледника (аналог MCoRDS L2, после очистки pandas)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/mcords_radar_profile.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 3. МОДЕЛЬ ПОЛОЖИТЕЛЬНЫХ ГРАДУСО-ДНЕЙ (PDD)
# =============================================================================
#
# PDD переводит температурные наблюдения в толщину растаявшего льда:
#
#   PDD = sum_i max(T_i - T_m, 0)     (сумма по дням года)
#   M_melt = f_snow * PDD   (пока не растаял снег)
#           или f_ice * PDD  (после стаивания снежного покрова -> голый
#                             лёд, ниже альбедо -> быстрее тает)
#
# f_ice > f_snow (лёд темнее свежего снега -> поглощает больше радиации).
# T_m — пороговая температура таяния (здесь используется калибровка
# для конкретного случая Penny Ice Cap, T_m = 270.4 K = -2.75°C —
# отражает эмпирическую поправку на то, что таяние может начинаться
# при среднесуточной температуре чуть ниже 0°C из-за суточного хода).


def daily_temperature_series(n_days=365, T_mean_annual_C=-15.0, T_amplitude_C=18.0,
                              lapse_rate=-0.006, elevation_m=0.0, noise_C=3.0):
    """Синтетический суточный ход температуры (годовой цикл + шум)."""
    day = np.arange(n_days)
    T = T_mean_annual_C + T_amplitude_C * (-np.cos(2 * np.pi * day / 365.25)) \
        + lapse_rate * elevation_m + rng.normal(0, noise_C, n_days)
    return day, T + 273.15  # в Кельвинах


def pdd_melt_model(T_kelvin, T_m=270.4, f_snow=3.0, f_ice=8.0, snow_reservoir_mm=300.0):
    """
    Расчёт таяния по модели PDD с учётом истощения снежного покрова.

    Параметры
    ----------
    T_kelvin : np.ndarray
        Суточные температуры, К
    T_m : float
        Пороговая температура таяния, К (270.4 К — калибровка Penny Ice Cap)
    f_snow, f_ice : float
        Коэффициенты таяния снега/льда, мм в.э. / (K * день)
    snow_reservoir_mm : float
        Начальный запас снега (в водном эквиваленте), мм

    Возвращает
    -------
    dict с суточным и накопленным таянием, а также днём перехода
    "снег -> лёд" (истощение снежного покрова)
    """
    dPDD = np.clip(T_kelvin - T_m, 0, None)  # градусо-дни (в кельвинах = градусах Цельсия по интервалу)

    daily_melt = np.zeros_like(dPDD)
    snow_remaining = snow_reservoir_mm
    snow_depleted_day = None

    for i, dpdd in enumerate(dPDD):
        potential_melt_snow = f_snow * dpdd
        if snow_remaining > 0:
            melt_from_snow = min(potential_melt_snow, snow_remaining)
            snow_remaining -= melt_from_snow
            remaining_pdd_equiv = (potential_melt_snow - melt_from_snow) / f_snow if f_snow > 0 else 0
            melt_from_ice = f_ice * remaining_pdd_equiv if snow_remaining <= 0 else 0.0
            daily_melt[i] = melt_from_snow + melt_from_ice
            if snow_remaining <= 0 and snow_depleted_day is None:
                snow_depleted_day = i
        else:
            daily_melt[i] = f_ice * dpdd

    return {
        "PDD_daily": dPDD,
        "melt_daily_mm": daily_melt,
        "melt_cumulative_mm": np.cumsum(daily_melt),
        "snow_depleted_day": snow_depleted_day,
        "total_melt_mm": daily_melt.sum(),
    }


def plot_pdd_model():
    day, T = daily_temperature_series(elevation_m=1500.0)
    result = pdd_melt_model(T)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(day, T - 273.15, color="#1f77b4", lw=1)
    axes[0].axhline(270.4 - 273.15, color="gray", ls="--", lw=1, label="T_m = -2.75°C")
    axes[0].set_ylabel("Температура, °C")
    axes[0].set_title("Синтетический суточный ход температуры")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(day, result["melt_daily_mm"], color="#d62728", lw=1)
    if result["snow_depleted_day"]:
        axes[1].axvline(result["snow_depleted_day"], color="orange", ls="--", lw=1.5,
                         label=f"Истощение снега (день {result['snow_depleted_day']})")
        axes[1].legend(fontsize=8)
    axes[1].set_ylabel("Суточное таяние, мм в.э.")
    axes[1].set_title("Модель PDD: суточное таяние (снег -> лёд)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(day, result["melt_cumulative_mm"], color="#2ca02c", lw=2)
    axes[2].set_xlabel("День года")
    axes[2].set_ylabel("Накопленное таяние, мм в.э.")
    axes[2].set_title(f"Суммарное годовое таяние: {result['total_melt_mm']:.0f} мм в.э.")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/pdd_model.png", dpi=150)
    plt.close(fig)

    return result


# =============================================================================
# 4. ФЕНОМЕНОЛОГИЧЕСКИЕ (AD-HOC) МОДЕЛИ ЛЕДНИКОВЫХ ЦИКЛОВ
# =============================================================================
#
# Синтетическая инсоляционная "прокси"-кривая, качественно
# воспроизводящая структуру циклов Миланковича (эксцентриситет ~100 тыс.
# лет, наклон оси ~41 тыс. лет, прецессия ~23 тыс. лет). Используется
# как общий форсинг для всех трёх моделей ниже. Это упрощённая учебная
# аппроксимация, а не астрономически точное решение (напр. Laskar et al.).


def milankovitch_insolation_proxy(t_ka):
    """
    t_ka — время в тысячах лет (может быть отрицательным = "тысяч лет
    назад" при использовании как t_ka = -age_kyr).
    Возвращает безразмерную аномалию инсоляции (условные единицы,
    среднее ~0, амплитуда ~1).
    """
    ecc = 0.55 * np.cos(2 * np.pi * t_ka / 100.0)
    obliquity = 0.30 * np.cos(2 * np.pi * t_ka / 41.0 + 0.4)
    precession = 0.15 * np.cos(2 * np.pi * t_ka / 23.0 + 1.1)
    return ecc + obliquity + precession


# --- 4.1 Модель Колдера (Calder, 1974): релейная (switch) модель ---
#
# Ледник реагирует на инсоляцию асимметрично: тает быстро (при тёплом
# форсинге), нарастает медленно (при холодном форсинге).

def calder_model_rhs(t_ka, V, k_melt, k_grow):
    I = milankovitch_insolation_proxy(t_ka)
    if I > 0:  # тёплый форсинг -> таяние (быстро)
        dVdt = -k_melt * I
    else:  # холодный форсинг -> рост (медленно)
        dVdt = -k_grow * I  # I<0 => -k_grow*I>0, рост
    return [dVdt]


def run_calder_model(t_ka_span, V0=1.0, k_melt=0.08, k_grow=0.01, n_eval=2000):
    t_eval = np.linspace(*t_ka_span, n_eval)
    sol = solve_ivp(calder_model_rhs, t_ka_span, [V0], t_eval=t_eval,
                     args=(k_melt, k_grow), method="RK45", max_step=0.5)
    V = np.clip(sol.y[0], 0, None)
    return t_eval, V


# --- 4.2 Модель Имбри (Imbrie, 1980): релаксационная модель с задержкой ---
#
# tau * dV/dt = V_eq(I) - V
# Ледник стремится к равновесному объёму V_eq(I), но с конечным
# временем релаксации tau -> отклик отстаёт по фазе от форсинга
# (характерная задержка климатической системы).

def imbrie_model_rhs(t_ka, V, tau_ka, alpha):
    I = milankovitch_insolation_proxy(t_ka)
    V_eq = 1.0 - alpha * I  # тёплый форсинг (I>0) -> меньше равновесный объём
    dVdt = (V_eq - V[0]) / tau_ka
    return [dVdt]


def run_imbrie_model(t_ka_span, V0=1.0, tau_ka=8.0, alpha=0.6, n_eval=2000):
    t_eval = np.linspace(*t_ka_span, n_eval)
    sol = solve_ivp(imbrie_model_rhs, t_ka_span, [V0], t_eval=t_eval,
                     args=(tau_ka, alpha), method="RK45", max_step=0.5)
    V = np.clip(sol.y[0], 0, None)
    return t_eval, V


# --- 4.3 Модель Пайара (Paillard, 1998): мультистабильная пороговая модель ---
#
# ВАЖНОЕ МЕТОДИЧЕСКОЕ ЗАМЕЧАНИЕ: ниже приведена УПРОЩЁННАЯ иллюстративная
# реализация идеи Пайара (три состояния — межледниковье i, мягкое
# оледенение g, полное оледенение G; пороговые переходы между ними в
# зависимости от инсоляции и накопленного объёма льда), а не дословное
# воспроизведение оригинальных параметров Paillard (1998, Nature,
# "The timing of Pleistocene glaciations from a simple multi-state
# climate model"). Модель качественно воспроизводит пилообразную
# асимметрию (медленный рост -> быстрый переход в межледниковье).

PAILLARD_STATES = {"i": 0, "g": 1, "G": 2}
# Скорость релаксации к целевому объёму в каждом состоянии (1/тыс.лет)
PAILLARD_RATES = {"i": -0.30, "g": 0.08, "G": 0.04}
PAILLARD_TARGETS = {"i": 0.0, "g": 0.6, "G": 1.0}


def paillard_transition(state, V, I):
    """
    Пороговые правила перехода между состояниями (упрощённые).

    ПРИМЕЧАНИЕ: пороги подобраны так, чтобы модель качественно
    воспроизводила известное соотношение из палеоданных — большую часть
    400 тыс. лет ледник проводит в (пере)ледниковых состояниях (g, G),
    а межледниковья (i) относительно кратки и наступают лишь при
    достаточно сильном и устойчивом положительном форсинге.
    """
    i1, i0, g1, g1_exit = 0.45, 0.15, -0.30, 0.0
    if state == "i" and I < i0:
        return "g"
    if state == "g":
        if I < g1:
            return "G"
        if I > i1:
            return "i"
    if state == "G" and I > g1_exit:
        return "g"
    return state


def run_paillard_model(t_ka_span, n_eval=2000):
    t_eval = np.linspace(*t_ka_span, n_eval)
    dt = t_eval[1] - t_eval[0]
    V = np.zeros(n_eval)
    states = []
    state = "G"
    V[0] = 1.0

    for k in range(1, n_eval):
        I = milankovitch_insolation_proxy(t_eval[k - 1])
        state = paillard_transition(state, V[k - 1], I)
        states.append(state)
        rate = PAILLARD_RATES[state]
        target = PAILLARD_TARGETS[state]
        V[k] = V[k - 1] + dt * rate * np.sign(target - V[k - 1])
        V[k] = np.clip(V[k], 0, 1.3)
    states.append(state)

    return t_eval, V, states


def plot_adhoc_climate_models():
    t_ka_span = (-400, 0)  # 400 тыс. лет назад -> настоящее время
    t_grid = np.linspace(*t_ka_span, 2000)
    I = milankovitch_insolation_proxy(t_grid)

    t_c, V_calder = run_calder_model(t_ka_span)
    t_im, V_imbrie = run_imbrie_model(t_ka_span)
    t_p, V_paillard, states = run_paillard_model(t_ka_span)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t_grid, I, color="#7f7f7f", lw=1)
    axes[0].set_ylabel("Инсоляция\n(усл. ед.)")
    axes[0].set_title("Синтетический прокси-форсинг Миланковича")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_c, V_calder, color="#1f77b4", lw=1.5)
    axes[1].set_ylabel("Объём льда\n(усл. ед.)")
    axes[1].set_title("Модель Колдера (Calder, 1974) — релейная модель")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_im, V_imbrie, color="#d62728", lw=1.5)
    axes[2].set_ylabel("Объём льда\n(усл. ед.)")
    axes[2].set_title("Модель Имбри (Imbrie, 1980) — релаксация с задержкой")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t_p, V_paillard, color="#2ca02c", lw=1.5)
    axes[3].set_xlabel("Время, тыс. лет назад")
    axes[3].set_ylabel("Объём льда\n(усл. ед.)")
    axes[3].set_title("Модель Пайара (Paillard, 1998, упрощённая) — 3 устойчивых состояния")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/adhoc_climate_models.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 5. ПЛЕЙСТОЦЕНОВЫЙ ЛЕДЯНОЙ СЛОЙ (PIL)
# =============================================================================
#
# Реликтовый придонный лёд (сформировавшийся в плейстоцене, при иной
# температуре и структуре кристаллов) деформируется существенно легче
# современного голоценового льда. Мощность этого слоя ограничена
# эмпирической эвристикой:
#
#   H_p = min(0.12 * H, 80 м)
#
# где H — полная толщина ледника (формула применяется для глубоких зон,
# H > 150 м).


def calculate_pil_thickness(H):
    """Толщина плейстоценового слоя (PIL), м."""
    H = np.asarray(H, dtype=float)
    return np.minimum(0.12 * H, 80.0)


def velocity_profile_with_pil(z, H, dS_dx, A=9.34e-26, n=3.0, E_pil=3.5):
    """
    Профиль скорости по глубине (SIA) с учётом мягкого плейстоценового
    слоя у ложа: коэффициент усиления текучести (enhancement factor)
    E_pil > 1 увеличивает эффективную мягкость A в пределах слоя PIL,
    что даёт непропорционально большую долю сдвиговой деформации у ложа.

    По умолчанию A соответствует T ≈ -25°C (уравнение Аррениуса,
    Модуль 2) — типичная температура внутренней части глубокого
    ледникового щита в районе PIL-слоя (в отличие от умеренного льда
    выводных ледников из Модуля 3). При использовании более "тёплого"
    A (как в Модуле 3) и типичных для щита пологих уклонов (~0.005)
    скорость деформации для H~3000-3500 м даёт нереалистично огромные
    значения (~10⁴ м/год) — это тот же урок калибровки по ITS_LIVE,
    что и в Модуле 3: параметры закона Глена должны соответствовать
    РЕАЛЬНОЙ температуре и геометрии места, а не браться произвольно.

    Используется послойное численное интегрирование профиля скорости
    (в отличие от чисто аналитической формулы Модуля 3, здесь E зависит
    от глубины, поэтому интеграл вычисляется численно).
    """
    H_p = calculate_pil_thickness(H)
    z = np.atleast_1d(z).astype(float)

    # Коэффициент усиления E(z): =1 в голоценовом льду (z > H_p от ложа),
    # =E_pil в пределах PIL (0 <= z <= H_p)
    E_of_z = np.where(z <= H_p, E_pil, 1.0)

    tau_b_like = RHO_ICE * G * H * np.abs(dS_dx)  # характерное базальное напряжение

    # Локальная скорость сдвиговой деформации на высоте z' над ложем:
    # d(u)/dz' = 2*A*E(z')*(rho*g*|dS/dx|*(H-z'))^n
    z_fine = np.linspace(0, H, 400)
    E_fine = np.where(z_fine <= H_p, E_pil, 1.0)
    dudz = 2 * A * E_fine * (RHO_ICE * G * np.abs(dS_dx) * np.maximum(H - z_fine, 0)) ** n

    # Численное интегрирование от ложа (z'=0, u=0) до текущей высоты z
    u_fine = np.concatenate([[0.0], np.cumsum(0.5 * (dudz[1:] + dudz[:-1]) * np.diff(z_fine))])
    u_at_z = np.interp(z, z_fine, u_fine)

    return u_at_z, H_p


def plot_pil_and_velocity_profile():
    H = 3500.0
    dS_dx = 0.005  # типичный пологий уклон внутренней части ледникового щита
    z = np.linspace(0, H, 300)

    u_standard, _ = velocity_profile_with_pil(z, H, dS_dx, E_pil=1.0)
    u_pil, H_p = velocity_profile_with_pil(z, H, dS_dx, E_pil=3.5)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.plot(u_standard * SEC_PER_YEAR, H - z, ls="--", color="gray", lw=2, label="Стандартный лёд (E=1.0)")
    ax.plot(u_pil * SEC_PER_YEAR, H - z, color="#17becf", lw=2.5, label=f"С учётом PIL (E={3.5}, H_p={H_p:.0f} м)")
    ax.axhspan(0, H_p, color="orange", alpha=0.15, label="Плейстоценовый слой (PIL)")

    ax.invert_yaxis()
    ax.set_xlabel("Скорость сдвиговой деформации, м/год")
    ax.set_ylabel("Глубина от поверхности, м")
    ax.set_title("Роль мягкого плейстоценового слоя (PIL) в профиле скорости")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/pil_velocity_profile.png", dpi=150)
    plt.close(fig)

    return H_p, u_pil[-1] * SEC_PER_YEAR, u_standard[-1] * SEC_PER_YEAR


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    print("Модуль 1: Анализ данных и простые климатические модели")
    print("=" * 60)

    print("1. Синтетический временной ряд массы льда (аналог ICESat-2)...")
    df_mass = generate_mass_balance_timeseries()
    trend = plot_mass_balance_timeseries(df_mass)
    print(f"   Оценённый линейный тренд: {trend:.1f} Гт/год")

    print("2. Очистка радарных данных ложа (аналог MCoRDS L2, pandas)...")
    df_radar = generate_synthetic_mcords_profile()
    df_clean, stats = clean_and_analyze_mcords(df_radar)
    plot_mcords_profile(df_clean)
    print(f"   Точек: {stats['n_points_total']}, пропусков: {stats['n_missing']} "
          f"({stats['missing_frac']*100:.1f}%)")
    print(f"   Средняя толщина льда: {stats['mean_thickness_m']:.0f} м, "
          f"корреляция surface-bottom: {stats['surface_bottom_corr']:.2f}")

    print("3. Модель PDD (положительные градусо-дни)...")
    pdd_result = plot_pdd_model()
    print(f"   Суммарное годовое таяние: {pdd_result['total_melt_mm']:.0f} мм в.э.")

    print("4. Феноменологические модели циклов (Колдер, Имбри, Пайар)...")
    plot_adhoc_climate_models()

    print("5. Плейстоценовый слой (PIL) и профиль скорости...")
    H_p, u_pil_surf, u_std_surf = plot_pil_and_velocity_profile()
    print(f"   Толщина PIL: {H_p:.0f} м")
    print(f"   Поверхностная скорость: стандартный лёд = {u_std_surf:.1f} м/год, "
          f"с учётом PIL = {u_pil_surf:.1f} м/год "
          f"(усиление в {u_pil_surf/max(u_std_surf,1e-9):.1f}x)")

    print("\nГотово. Графики сохранены в /mnt/user-data/outputs/")
