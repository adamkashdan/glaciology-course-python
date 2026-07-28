"""
Модуль 4: Механика скольжения и универсальный закон Зоэта-Иверсона
======================================================================

Курс по гляциологии — практическая реализация на Python.

Содержание:
1. Регуляризованный закон Кулона (Zoet & Iverson, 2020; форма как в Schoof, 2005):
   универсальный закон скольжения, объединяющий степенной режим (медленное
   скольжение) и кулоновский пластический режим (быстрое скольжение).
2. Параметризация переходной скорости u_t в зависимости от размера обломков
   (clasts) на ложе и эффективного давления N.
3. Пружинно-ползунковая (spring-slider) модель "прыжков" (stick-slip) —
   классический механизм генерации ледниковых микросейсм / stick-slip
   событий (ср. приливные stick-slip на Whillans Ice Stream).

Требуемые библиотеки: numpy, matplotlib, scipy
Установка: pip install numpy matplotlib scipy

Ключевые ссылки:
- Zoet, L.K. & Iverson, N.R. (2020). A slip law for glaciers on deformable
  beds. Science, 368(6486), 76-78.
- Schoof, C. (2005). The effect of cavitation on glacier sliding.
  Proc. R. Soc. A, 461, 609-627.
- Helanow et al. (2021). A slip law for hard-bedded glaciers derived from
  observed bed topography. Science Advances, 7(20).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

RHO_ICE = 917.0
G = 9.81
SEC_PER_YEAR = 365.25 * 24 * 3600

# =============================================================================
# 1. РЕГУЛЯРИЗОВАННЫЙ ЗАКОН КУЛОНА (ZOET & IVERSON, 2020)
# =============================================================================
#
#   tau_b = C * N * (u_s / (u_s + u_t))^(1/n)
#
# где:
#   tau_b — базальное касательное напряжение (драг), Па
#   N     — эффективное давление (давление льда минус давление воды), Па
#   C     — безразмерный коэффициент кулоновского трения (~0.4-0.8 для till)
#   u_s   — скорость скольжения, м/с
#   u_t   — переходная скорость: выше неё till/ложе "срывается" в
#           кулоновский пластический режим (независимость tau_b от u_s)
#   n     — показатель степени закона Глена (используется и здесь по
#           аналогии со степенным режимом при малых скоростях)
#
# При u_s << u_t:  tau_b ~ C*N*(u_s/u_t)^(1/n)   — "rate-strengthening",
#                  похоже на вязкое обтекание обломков (Weertman-режим).
# При u_s >> u_t:  tau_b -> C*N                   — кулоновское плато,
#                  till деформируется/обломки "пропахивают" ложе при
#                  постоянном пороговом напряжении, не зависящем от скорости.


def zoet_iverson_stress(u_s, N, C=0.5, u_t=50.0 / SEC_PER_YEAR, n=3.0):
    """
    Базальное напряжение по регуляризованному закону Кулона.

    Параметры
    ----------
    u_s : float или np.ndarray
        Скорость скольжения, м/с
    N : float
        Эффективное давление, Па
    C : float
        Коэффициент кулоновского трения (безразмерный)
    u_t : float
        Переходная скорость, м/с (типично 40-80 м/год по Zoet & Iverson, 2020)
    n : float
        Показатель степени (по аналогии с законом Глена)
    """
    u_s = np.abs(u_s)
    return C * N * (u_s / (u_s + u_t)) ** (1.0 / n)


def zoet_iverson_velocity(tau_b, N, C=0.5, u_t=50.0 / SEC_PER_YEAR, n=3.0):
    """
    Обратная функция: скорость скольжения по заданному базальному
    напряжению (нужно для моделей, где известен driving stress, а
    скорость — искомая величина).

        (tau_b / (C*N))^n = u_s / (u_s + u_t)
        => u_s = u_t * r / (1 - r),   r = (tau_b/(C*N))^n

    Напряжение tau_b физически не может превышать C*N (кулоновский
    предел) — при приближении к нему скорость скольжения расходится
    (till "течёт" сколь угодно быстро при постоянном напряжении).
    """
    tau_b = np.asarray(tau_b, dtype=float)
    r = np.clip(tau_b / (C * N), 0.0, 0.999999)  # регуляризация вблизи предела
    r_n = r ** n
    return u_t * r_n / (1.0 - r_n)


def plot_slip_law_curve():
    """
    Воспроизводит классический график (Zoet & Iverson, 2020, Fig. 1 style):
    нормированный драг tau_b/N в зависимости от скорости скольжения —
    для нескольких значений переходной скорости u_t.
    """
    u_s = np.logspace(-1, 4, 400) / SEC_PER_YEAR  # от 0.1 до 10000 м/год

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for u_t_yr, color in zip([20, 50, 100, 200], ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        tau_over_N = zoet_iverson_stress(u_s, N=1.0, C=0.5, u_t=u_t_yr / SEC_PER_YEAR) / 0.5
        ax.semilogx(u_s * SEC_PER_YEAR, tau_over_N, lw=2, color=color, label=f"u_t = {u_t_yr} м/год")

    ax.axhline(1.0, color="gray", ls="--", lw=1, label="Кулоновский предел (τ_b/N = C)")
    ax.set_xlabel("Скорость скольжения u_s, м/год")
    ax.set_ylabel(r"Нормированный драг $\tau_b / (C\,N)$")
    ax.set_title("Регуляризованный закон Кулона (Zoet & Iverson, 2020)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/zoet_iverson_slip_law.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 2. ПЕРЕХОДНАЯ СКОРОСТЬ u_t: ЗАВИСИМОСТЬ ОТ РАЗМЕРА ОБЛОМКОВ И N
# =============================================================================
#
# ВАЖНОЕ МЕТОДИЧЕСКОЕ ЗАМЕЧАНИЕ: физический смысл u_t — это скорость,
# при которой напряжение, создаваемое обтеканием льдом выступающих
# обломков (clasts) на ложе, достигает кулоновской прочности till. Из
# качественных выводов Zoet & Iverson (2020) и последующих работ
# (Helanow et al., 2021) следует, что u_t должна РАСТИ с эффективным
# давлением N (выше N -> выше прочность till -> нужна большая скорость,
# чтобы драг "дорос" до предела) и МЕНЯТЬСЯ с характерным размером
# обломков/шероховатостью ложа (крупные обломки создают большее локальное
# напряжение при том же u_s, поэтому порог достигается раньше -> меньшее
# u_t; наблюдаемый диапазон u_t в экспериментах и полевых оценках — от
# ~40 до ~80 м/год).
#
# Ниже приведена УПРОЩЁННАЯ иллюстративная параметризация (не дословная
# формула из оригинальной статьи, а педагогическая модель, откалиброванная
# на этот диапазон), качественно воспроизводящая эти зависимости.


def transition_velocity(N, clast_size, N_ref=5e5, d_ref=0.1, u_t_ref=50.0 / SEC_PER_YEAR):
    """
    Иллюстративная параметризация переходной скорости u_t(N, d_clast).

        u_t = u_t_ref * (N / N_ref) * (d_ref / d_clast)

    Параметры
    ----------
    N : float или np.ndarray
        Эффективное давление, Па
    clast_size : float или np.ndarray
        Характерный размер обломков на ложе, м
    N_ref, d_ref : float
        Референсные значения (при которых u_t = u_t_ref)
    u_t_ref : float
        Референсная переходная скорость, м/с (по умолчанию 50 м/год —
        типичное значение из диапазона Zoet & Iverson, 2020 / реализаций
        в моделях PISM, CISM)
    """
    return u_t_ref * (N / N_ref) * (d_ref / clast_size)


def plot_transition_velocity():
    """График u_t(N) для нескольких характерных размеров обломков."""
    N_range = np.linspace(1e5, 1.5e6, 200)  # эффективное давление, Па (0.1-1.5 МПа)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for d_clast, label, color in zip(
        [0.02, 0.05, 0.1, 0.3],
        ["мелкий гравий (2 см)", "средний гравий (5 см)", "крупная галька (10 см)", "валуны (30 см)"],
        ["#9467bd", "#1f77b4", "#2ca02c", "#d62728"],
    ):
        u_t = transition_velocity(N_range, d_clast) * SEC_PER_YEAR
        ax.plot(N_range / 1e3, u_t, lw=2, color=color, label=label)

    ax.axhspan(40, 80, color="gray", alpha=0.15, label="Наблюдаемый диапазон\n(Zoet & Iverson, 2020)")
    ax.set_xlabel("Эффективное давление N, кПа")
    ax.set_ylabel("Переходная скорость u_t, м/год")
    ax.set_title("Переходная скорость: роль размера обломков и эффективного давления")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/transition_velocity.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 3. ПРУЖИННО-ПОЛЗУНКОВАЯ МОДЕЛЬ STICK-SLIP
# =============================================================================
#
# Классическая концептуальная модель генерации "прыжков" ледникового
# потока (ср. приливные stick-slip события на Whillans Ice Stream,
# Bindschadler et al., 2003; Winberry et al., 2009; Lipovsky & Dunham,
# 2016): лёд выше точки наблюдения ведёт себя как упругая "пружина",
# нагружаемая постоянной скоростью течения вышележащего льда (V_load).
# Локальное базальное напряжение tau следует квазистатическому закону
# скольжения (здесь — регуляризованный закон Кулона), а скорость
# скольжения u_s(tau) определяется обращением этого закона.
#
#   d(tau)/dt = k * (V_load - u_s(tau, N))
#
# ВАЖНЫЙ МЕТОДИЧЕСКИЙ РЕЗУЛЬТАТ: поскольку регуляризованный закон Кулона
# всюду "rate-strengthening" (drag монотонно растёт со скоростью), эта
# система при ПОСТОЯННОМ N всегда УСТОЙЧИВО релаксирует к стационарному
# режиму скольжения (u_s = V_load) — самопроизвольных, незатухающих
# stick-slip колебаний она не порождает. Это соответствует известному
# результату теории трения: чисто скорость-упрочняющиеся законы
# скольжения устойчивы.
#
# Наблюдаемые в природе stick-slip события (например, на Whillans Ice
# Stream) возникают за счёт ВНЕШНЕГО периодического воздействия —
# приливной модуляции эффективного давления N. Когда N кратковременно
# падает, кулоновский предел C*N тоже падает и может опуститься ниже
# текущего накопленного напряжения tau — тогда скорость скольжения резко
# (импульсно) возрастает, напряжение сбрасывается, после чего цикл
# нагружения начинается заново. Ниже показаны оба режима: устойчивое
# стационарное скольжение (постоянное N) и импульсные "прыжки",
# синхронизированные с фазой приливной модуляции N.


def stick_slip_rhs(t, y, k, V_load, N_func, C, u_t, n):
    """Правая часть ОДУ для пружинно-ползунковой модели: d(tau)/dt."""
    tau = y[0]
    N = N_func(t)
    u_s = zoet_iverson_velocity(tau, N, C=C, u_t=u_t, n=n)
    dtau_dt = k * (V_load - u_s)
    return [dtau_dt]


def simulate_stick_slip(
    k=2e4,                      # "жёсткость" пружины, Па с/м (эффективная упругость столба льда)
    V_load=200.0 / SEC_PER_YEAR,  # скорость нагружения (фоновое течение льда), м/с
    N0=4e5,                      # базовое эффективное давление, Па
    N_amplitude=0.0,             # амплитуда периодической модуляции N (0 = без модуляции)
    N_period_hours=12.42,        # период модуляции (по умолчанию — полусуточный "приливной")
    C=0.5, u_t=50.0 / SEC_PER_YEAR, n=3.0,
    t_end_hours=72.0,
    tau0_frac=0.3,
):
    """
    Интегрирует пружинно-ползунковую ОДУ и возвращает временные ряды
    напряжения и скорости скольжения.

    N_amplitude > 0 позволяет промоделировать влияние периодических
    колебаний эффективного давления (например, приливной модуляции
    подлёдного давления воды, как на Whillans Ice Stream) на частоту и
    амплитуду stick-slip событий.
    """
    def N_func(t_sec):
        omega = 2 * np.pi / (N_period_hours * 3600)
        return N0 + N_amplitude * np.sin(omega * t_sec)

    tau0 = tau0_frac * C * N0
    t_span = (0, t_end_hours * 3600)
    t_eval = np.linspace(*t_span, 20000)

    sol = solve_ivp(
        stick_slip_rhs, t_span, [tau0], t_eval=t_eval,
        args=(k, V_load, N_func, C, u_t, n),
        method="RK45", max_step=30.0, rtol=1e-8, atol=1e-6,
    )

    tau = sol.y[0]
    N_t = N_func(sol.t)
    u_s = zoet_iverson_velocity(tau, N_t, C=C, u_t=u_t, n=n)

    return {
        "t_hours": sol.t / 3600,
        "tau": tau,
        "u_s_myr": u_s * SEC_PER_YEAR,
        "N": N_t,
    }


def plot_stick_slip(result, title_suffix=""):
    """Строит временные ряды напряжения и скорости + фазовый портрет."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(result["t_hours"], result["tau"] / 1e3, color="#1f77b4", lw=1)
    axes[0].set_xlabel("Время, ч")
    axes[0].set_ylabel("Базальное напряжение τ, кПа")
    axes[0].set_title("Накопление и сброс напряжения")
    axes[0].grid(alpha=0.3)

    axes[1].plot(result["t_hours"], result["u_s_myr"], color="#d62728", lw=1)
    axes[1].set_xlabel("Время, ч")
    axes[1].set_ylabel("Скорость скольжения, м/год")
    axes[1].set_title("\"Прыжки\" скорости скольжения (stick-slip)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(result["u_s_myr"], result["tau"] / 1e3, color="#2ca02c", lw=0.7)
    axes[2].set_xlabel("Скорость скольжения, м/год")
    axes[2].set_ylabel("Напряжение τ, кПа")
    axes[2].set_title("Фазовый портрет (предельный цикл)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Пружинно-ползунковая модель stick-slip {title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/stick_slip_simulation.png", dpi=150)
    plt.close(fig)


def plot_stick_slip_tidal_modulation():
    """
    Демонстрация влияния периодической (например, приливной) модуляции
    эффективного давления N на характер stick-slip событий — качественный
    аналог наблюдений на Whillans Ice Stream (Bindschadler et al., 2003;
    Winberry et al., 2009), где приливные колебания подлёдного давления
    воды модулируют время между "прыжками" ледникового потока.

    В отличие от стационарного случая (постоянное N), здесь снижение N
    на каждом приливном минимуме кратковременно опускает кулоновский
    предел C*N ниже накопленного напряжения — это даёт резкий импульсный
    скачок скорости скольжения (типичное stick-slip событие), после
    которого напряжение сбрасывается и цикл нагружения начинается заново.
    """
    result = simulate_stick_slip(
        N_amplitude=3.3e5, N_period_hours=12.42, t_end_hours=72.0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    axes[0].plot(result["t_hours"], result["N"] / 1e3, color="#9467bd", lw=1.5)
    axes[0].set_ylabel("Эффективное давление N, кПа")
    axes[0].set_title("Приливная модуляция эффективного давления")
    axes[0].grid(alpha=0.3)

    axes[1].plot(result["t_hours"], result["u_s_myr"], color="#d62728", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Время, ч")
    axes[1].set_ylabel("Скорость скольжения, м/год\n(лог. шкала)")
    axes[1].set_title("Импульсные stick-slip события синхронизированы с минимумами N")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/stick_slip_tidal_modulation.png", dpi=150)
    plt.close(fig)

    return result


# =============================================================================
# 4. МЁРЗЛАЯ КАЙМА (FROZEN FRINGE)
# =============================================================================
#
# Мёрзлая кайма — переходный слой на границе лёд-ложе, в котором лёд
# проникает в поры подстилающих отложений. Её механическая прочность
# определяется взаимодействием насыщения льдом (S) и порового давления
# воды (см. Rempel, 2008; Meyer & Minchew, 2018; Hansen et al., 2024,
# "Presence of Frozen Fringe Impacts Soft-Bedded Slip Relationship",
# Geophysical Research Letters):
#
#   1. ЗАКУПОРКА: рост насыщения льдом S снижает проницаемость осадка
#      (лёд забивает поровые каналы) -> вода не успевает оттекать при
#      нагружении -> избыточное поровое давление растёт с S.
#   2. ОСЛАБЛЕНИЕ: по механике грунтов прочность определяется эффективным
#      напряжением N_eff = N_far - ΔP(S) (вес льда минус давление воды).
#      Рост ΔP(S) снижает N_eff -> кайма механически СЛАБЕЕ, чем
#      свободный от льда till ниже неё, особенно на малых скоростях.
#   3. ЗАВИСИМОСТЬ ОТ S: сопротивление кулоновскому+скоростному закону
#      определяется тем же регуляризованным законом Кулона, но с N_eff(S)
#      вместо "дальнего" эффективного давления N_far. Rate-strengthening
#      сохраняется вплоть до нового (сниженного) кулоновского предела
#      C*N_eff(S).
#
# Ниже — упрощённая иллюстративная параметризация этих трёх эффектов
# (не дословные уравнения из статьи, а педагогическая модель, качественно
# воспроизводящая описанное поведение).


def pore_pressure_excess(S, dP_max, p=3.0):
    """
    Избыточное поровое давление в кайме за счёт "закупорки" (choking)
    поровых каналов льдом.

        ΔP(S) = dP_max * S^p

    При S -> 0 (поры свободны ото льда) ΔP -> 0 (давление как в глубоком
    till). При S -> 1 (поры полностью заполнены льдом) ΔP -> dP_max —
    вода почти не может оттекать, давление приближается к весу
    вышележащего льда.

    Параметры
    ----------
    S : float или np.ndarray
        Степень насыщения льдом порового пространства, [0, 1]
    dP_max : float
        Максимальное избыточное поровое давление при S=1, Па
    p : float
        Показатель нелинейности "закупорки" (p>1: давление растёт
        медленно при малых S и резко ускоряется при приближении к
        полному насыщению — качественно как в Kozeny-Carman-подобных
        моделях проницаемости)
    """
    S = np.clip(np.asarray(S, dtype=float), 0.0, 1.0)
    return dP_max * S ** p


def fringe_effective_pressure(S, N_far, dP_max, p=3.0, N_min_frac=0.02):
    """
    Эффективное давление внутри мёрзлой каймы как функция насыщения льдом.

        N_eff(S) = N_far - ΔP(S)

    N_min_frac задаёт нижнюю границу (малую долю от N_far), чтобы
    избежать нефизичного N_eff <= 0 при экстремальном насыщении.
    """
    dP = pore_pressure_excess(S, dP_max, p)
    return np.maximum(N_far - dP, N_min_frac * N_far)


def fringe_slip_stress(u_s, S, N_far, C=0.5, u_t=50.0 / SEC_PER_YEAR, n=3.0,
                        dP_max=None, p=3.0):
    """
    Базальное напряжение мёрзлой каймы: регуляризованный закон Кулона
    (та же функциональная форма, что и zoet_iverson_stress), но с
    эффективным давлением, ослабленным насыщением льдом.

    Если dP_max не задан, используется 0.9*N_far (при полном насыщении
    почти весь вес льда несёт вода, N_eff падает до N_min_frac*N_far).
    """
    if dP_max is None:
        dP_max = 0.9 * N_far
    N_eff = fringe_effective_pressure(S, N_far, dP_max, p)
    return zoet_iverson_stress(u_s, N_eff, C=C, u_t=u_t, n=n), N_eff


def plot_frozen_fringe():
    """
    Сравнивает кривые скольжения (drag vs скорость) для нескольких
    уровней насыщения льдом S и для "дальнего" (свободного ото льда)
    till как эталона. Показывает: (а) при любом S сохраняется
    rate-strengthening до кулоновского предела; (б) с ростом S предел
    и вся кривая опускаются ниже эталонной кривой till — кайма слабее,
    особенно заметно на малых скоростях.
    """
    u_s = np.logspace(-1, 4, 400) / SEC_PER_YEAR
    N_far = 5e5  # эффективное давление в глубоком till, Па

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Панель 1: кривые скольжения для разных S ---
    tau_till, _ = fringe_slip_stress(u_s, S=0.0, N_far=N_far)  # эталон: S=0 <=> till
    axes[0].semilogx(u_s * SEC_PER_YEAR, tau_till / 1e3, lw=2.5, color="black",
                      ls="--", label="Свободный ото льда till (эталон)")

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, 4))
    for S_val, color in zip([0.3, 0.6, 0.85, 0.97], colors):
        tau_fringe, N_eff = fringe_slip_stress(u_s, S=S_val, N_far=N_far)
        axes[0].semilogx(u_s * SEC_PER_YEAR, tau_fringe / 1e3, lw=2, color=color,
                          label=f"Мёрзлая кайма, S = {S_val:.2f} (N_eff = {N_eff/1e3:.0f} кПа)")

    axes[0].set_xlabel("Скорость скольжения u_s, м/год")
    axes[0].set_ylabel("Базальное напряжение τ, кПа")
    axes[0].set_title("Ослабление сопротивления с ростом насыщения льдом S")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    # --- Панель 2: N_eff(S) и избыточное поровое давление ---
    S_range = np.linspace(0, 1, 200)
    N_eff_range = fringe_effective_pressure(S_range, N_far, dP_max=0.9 * N_far)
    dP_range = pore_pressure_excess(S_range, dP_max=0.9 * N_far)

    axes[1].plot(S_range, N_eff_range / 1e3, lw=2, color="#1f77b4", label="N_eff(S) — эффективное давление")
    axes[1].plot(S_range, dP_range / 1e3, lw=2, color="#d62728", ls="--", label="ΔP(S) — избыточное поровое давление")
    axes[1].set_xlabel("Насыщение льдом S")
    axes[1].set_ylabel("Давление, кПа")
    axes[1].set_title("Закупорка пор льдом снижает эффективное давление")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Мёрзлая кайма: связь насыщения льдом, порового давления и прочности ложа "
                  "(ср. Hansen et al., 2024, GRL)", fontsize=11)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/frozen_fringe.png", dpi=150)
    plt.close(fig)

    print(f"   При S=0.97: N_eff = {fringe_effective_pressure(0.97, N_far, 0.9*N_far):.0f} Па "
          f"({fringe_effective_pressure(0.97, N_far, 0.9*N_far)/N_far*100:.1f}% от N_far={N_far:.0e} Па)")

if __name__ == "__main__":
    print("Модуль 4: Механика скольжения и закон Зоэта-Иверсона")
    print("=" * 60)

    print("1. Регуляризованный закон Кулона (кривая tau_b/N vs u_s)...")
    plot_slip_law_curve()

    print("2. Переходная скорость u_t(N, размер обломков)...")
    plot_transition_velocity()

    print("3. Пружинно-ползунковая модель: постоянное N (проверка устойчивости)...")
    result = simulate_stick_slip(N_amplitude=0.0, t_end_hours=72.0)
    plot_stick_slip(result, title_suffix="(постоянное N — устойчивая релаксация к V_load)")
    print(f"   u_s(t=0) = {result['u_s_myr'][0]:.2f} м/год -> "
          f"u_s(t=72ч) = {result['u_s_myr'][-1]:.2f} м/год "
          f"(стремится к V_load = {200.0:.0f} м/год, колебаний нет)")

    print("4. Stick-slip с приливной модуляцией эффективного давления...")
    result_tidal = plot_stick_slip_tidal_modulation()
    print(f"   Скорость скольжения: {result_tidal['u_s_myr'].min():.2f} - "
          f"{result_tidal['u_s_myr'].max():.0f} м/год (импульсные события на минимумах N)")

    print("5. Мёрзлая кайма: ослабление сопротивления при насыщении льдом S...")
    plot_frozen_fringe()

    print("\nГотово. Графики сохранены в /mnt/user-data/outputs/")
