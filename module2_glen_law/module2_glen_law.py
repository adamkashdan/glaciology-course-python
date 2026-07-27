"""
Модуль 2: Реология льда и закон Глена
========================================

Курс по гляциологии — практическая реализация на Python.

Содержание:
1. Закон Глена: связь напряжения и скорости деформации
2. Уравнение Аррениуса: зависимость коэффициента мягкости A от температуры
3. Расчёт эффективной вязкости льда (в т.ч. с поправкой на содержание пыли)
4. Модель отклика материала на циклическую нагрузку

Требуемые библиотеки: numpy, matplotlib, scipy
Установка: pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# 1. ЗАКОН ГЛЕНА
# =============================================================================
#
# Лёд ведёт себя как неньютоновская вязкая жидкость. Связь между скоростью
# деформации (эффективной скоростью сдвига) и девиаторным напряжением
# описывается степенным законом:
#
#       ε̇ = A * τ^n
#
# где:
#   ε̇  — эффективная скорость деформации, с^-1
#   τ  — эффективное девиаторное напряжение, Па
#   A  — коэффициент мягкости (softness parameter), Па^-n * с^-1
#   n  — показатель степени закона Глена (обычно n = 3)


def glen_strain_rate(tau, A, n=3.0):
    """
    Скорость деформации по закону Глена.

    Параметры
    ----------
    tau : float или np.ndarray
        Эффективное девиаторное напряжение, Па
    A : float
        Коэффициент мягкости, Па^-n с^-1
    n : float
        Показатель степени закона Глена (по умолчанию 3)

    Возвращает
    -------
    float или np.ndarray
        Скорость деформации, с^-1
    """
    return A * np.abs(tau) ** (n - 1) * tau


def glen_effective_viscosity(tau, A, n=3.0, tau_min=1e3):
    """
    Эффективная (кажущаяся) вязкость льда, вычисленная из закона Глена.

    Вязкость нелинейной жидкости определяется как:
        eta_eff = tau / (2 * epsilon_dot)  =  0.5 * A^(-1) * tau^(1-n)

    tau_min используется как нижняя граница напряжения, чтобы избежать
    деления на ноль (сингулярность вязкости при tau -> 0, характерная
    для степенных реологических законов).
    """
    tau_safe = np.maximum(np.abs(tau), tau_min)
    eps_dot = glen_strain_rate(tau_safe, A, n)
    return tau_safe / (2.0 * eps_dot)


# =============================================================================
# 2. УРАВНЕНИЕ АРРЕНИУСА: A(T)
# =============================================================================
#
# Коэффициент мягкости A существенно зависит от температуры льда.
# Стандартная параметризация (Cuffey & Paterson, 2010) использует
# уравнение Аррениуса с изломом при T* = -10°C (263.15 K), поскольку
# энергия активации меняется при переходе через эту температуру.

R_GAS = 8.314  # универсальная газовая постоянная, Дж/(моль*К)
T_STAR = 263.15  # температура излома, К (-10°C)

# Энергия активации, Дж/моль
Q_LOW = 60e3    # для T < T*
Q_HIGH = 139e3  # для T >= T* (учитывает межзёренное плавление)

# Предэкспоненциальный множитель, Па^-3 с^-1
A0_LOW = 3.985e-13
A0_HIGH = 1.916e3


def arrhenius_A(T_kelvin, n=3.0):
    """
    Коэффициент мягкости A(T) по уравнению Аррениуса с изломом при -10°C.

    A(T) = A0 * exp(-Q / (R * T))

    Параметры
    ----------
    T_kelvin : float или np.ndarray
        Температура льда, К
    n : float
        Показатель степени закона Глена (для справки, здесь не используется
        напрямую, но обычно параметризация A0/Q приводится именно для n=3)

    Возвращает
    -------
    float или np.ndarray
        Коэффициент мягкости A, Па^-n с^-1
    """
    T = np.asarray(T_kelvin, dtype=float)
    A = np.where(
        T < T_STAR,
        A0_LOW * np.exp(-Q_LOW / (R_GAS * T)),
        A0_HIGH * np.exp(-Q_HIGH / (R_GAS * T)),
    )
    return A


def dust_enhancement_factor(dust_content_ppm, sensitivity=0.02):
    """
    Упрощённый эмпирический множитель усиления деформации (enhancement
    factor) в зависимости от содержания пыли/примесей в льду.

    Это учебная параметризация: реальные лабораторные данные (например,
    по кернам с высоким содержанием пыли в ледниковые периоды) показывают
    увеличение скорости деформации на десятки процентов при повышенном
    содержании нерастворимых частиц, за счёт изменения микроструктуры льда.

    E(dust) = 1 + sensitivity * dust_content_ppm

    Параметры
    ----------
    dust_content_ppm : float или np.ndarray
        Содержание пыли, частей на миллион (по массе)
    sensitivity : float
        Эмпирический коэффициент чувствительности

    Возвращает
    -------
    float
        Безразмерный коэффициент усиления E >= 1
    """
    return 1.0 + sensitivity * np.asarray(dust_content_ppm, dtype=float)


def softness_with_dust(T_kelvin, dust_content_ppm=0.0):
    """Комбинированный коэффициент A с поправкой на содержание пыли."""
    A_temp = arrhenius_A(T_kelvin)
    E = dust_enhancement_factor(dust_content_ppm)
    return A_temp * E


# =============================================================================
# 3. ВИЗУАЛИЗАЦИЯ: A(T) и вязкость
# =============================================================================

def plot_arrhenius_curve():
    """График зависимости коэффициента мягкости A от температуры."""
    T_celsius = np.linspace(-50, 0, 200)
    T_kelvin = T_celsius + 273.15
    A_values = arrhenius_A(T_kelvin)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Линейная шкала
    axes[0].plot(T_celsius, A_values, color="#1f77b4", lw=2)
    axes[0].axvline(-10, color="gray", ls="--", lw=1, label="Излом T* = -10°C")
    axes[0].set_xlabel("Температура, °C")
    axes[0].set_ylabel(r"$A$, Па$^{-3}$ с$^{-1}$")
    axes[0].set_title("Коэффициент мягкости A(T) — линейная шкала")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Логарифмическая шкала (наглядно показывает экспоненциальный рост)
    axes[1].semilogy(T_celsius, A_values, color="#d62728", lw=2)
    axes[1].axvline(-10, color="gray", ls="--", lw=1, label="Излом T* = -10°C")
    axes[1].set_xlabel("Температура, °C")
    axes[1].set_ylabel(r"$A$, Па$^{-3}$ с$^{-1}$ (лог. шкала)")
    axes[1].set_title("Коэффициент мягкости A(T) — логарифмическая шкала")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Уравнение Аррениуса для коэффициента мягкости льда", fontsize=13)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/arrhenius_A_vs_T.png", dpi=150)
    plt.close(fig)


def plot_viscosity_vs_stress():
    """
    График эффективной вязкости льда как функции напряжения при
    нескольких температурах — демонстрирует неньютоновское
    (сдвиговое разжижение) поведение льда.
    """
    tau = np.logspace(3, 6, 200)  # напряжение от 1 кПа до 1 МПа
    temperatures_C = [-30, -20, -10, -2]

    fig, ax = plt.subplots(figsize=(7, 5))
    for T_c in temperatures_C:
        T_k = T_c + 273.15
        A = arrhenius_A(T_k)
        eta = glen_effective_viscosity(tau, A)
        ax.loglog(tau, eta, lw=2, label=f"T = {T_c}°C")

    ax.set_xlabel("Девиаторное напряжение τ, Па")
    ax.set_ylabel(r"Эффективная вязкость $\eta_{eff}$, Па·с")
    ax.set_title("Сдвиговое разжижение льда: вязкость vs напряжение")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/viscosity_vs_stress.png", dpi=150)
    plt.close(fig)


def plot_dust_effect():
    """Влияние содержания пыли на коэффициент мягкости при фиксированной T."""
    dust_range = np.linspace(0, 200, 100)  # ppm
    T_k = -20 + 273.15

    A_clean = arrhenius_A(T_k)
    A_dusty = softness_with_dust(T_k, dust_range)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dust_range, A_dusty / A_clean, color="#2ca02c", lw=2)
    ax.set_xlabel("Содержание пыли, ppm")
    ax.set_ylabel(r"Относительное усиление $A_{dusty} / A_{clean}$")
    ax.set_title("Влияние содержания пыли на мягкость льда при T = -20°C")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/dust_effect.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 4. МОДЕЛЬ ОТКЛИКА НА ЦИКЛИЧЕСКУЮ НАГРУЗКУ
# =============================================================================
#
# Простая феноменологическая модель: полная деформация складывается из
# упругой (мгновенной) и вязкой (по закону Глена) компонент — аналог
# модели Максвелла, но с нелинейной (степенной) вязкой ветвью.
#
#   ε_total = ε_elastic + ε_viscous
#   dε_viscous/dt = A * τ(t)^n            (закон Глена)
#   ε_elastic = τ(t) / E                  (мгновенный упругий отклик)
#
# Напряжение задаётся как циклическая (синусоидальная) нагрузка —
# это может имитировать, например, приливные циклы нагрузки на
# шельфовый ледник или сезонные колебания напряжений.


def cyclic_stress(t, tau_amplitude, period_s, tau_mean=0.0):
    """Синусоидальное циклическое напряжение."""
    omega = 2 * np.pi / period_s
    return tau_mean + tau_amplitude * np.sin(omega * t)


def viscous_strain_ode(eps_viscous, t, A, n, tau_amplitude, period_s, tau_mean):
    """Правая часть ОДУ: d(eps_viscous)/dt = A * tau(t)^n (закон Глена)."""
    tau_t = cyclic_stress(t, tau_amplitude, period_s, tau_mean)
    return glen_strain_rate(tau_t, A, n)


def simulate_cyclic_response(
    T_celsius=-10.0,
    n=3.0,
    E_elastic=9e9,      # модуль Юнга льда, Па (типичное значение ~9 ГПа)
    tau_amplitude=1e5,  # амплитуда напряжения, Па (100 кПа)
    tau_mean=0.0,
    period_s=12.42 * 3600,  # период ~ полусуточный прилив (12ч 25мин)
    n_periods=3,
    n_points=2000,
):
    """
    Моделирует отклик льда (упругая + вязкая деформация по Глену)
    на циклическую нагрузку.

    Возвращает словарь с массивами времени, напряжения, вязкой,
    упругой и суммарной деформации — удобно для построения графиков
    в интерактивном Jupyter Notebook (например, с ipywidgets.interact
    для варьирования T, tau_amplitude, period_s).
    """
    T_k = T_celsius + 273.15
    A = arrhenius_A(T_k)

    t_end = n_periods * period_s
    t = np.linspace(0, t_end, n_points)

    # Численное интегрирование вязкой деформации
    eps_viscous = odeint(
        viscous_strain_ode,
        y0=0.0,
        t=t,
        args=(A, n, tau_amplitude, period_s, tau_mean),
    ).flatten()

    tau_t = cyclic_stress(t, tau_amplitude, period_s, tau_mean)
    eps_elastic = tau_t / E_elastic
    eps_total = eps_elastic + eps_viscous

    return {
        "t": t,
        "t_hours": t / 3600,
        "tau": tau_t,
        "eps_elastic": eps_elastic,
        "eps_viscous": eps_viscous,
        "eps_total": eps_total,
        "A": A,
    }


def plot_cyclic_response(result, title_suffix=""):
    """Строит три панели: напряжение, компоненты деформации, петля гистерезиса."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(result["t_hours"], result["tau"] / 1e3, color="#9467bd")
    axes[0].set_xlabel("Время, ч")
    axes[0].set_ylabel("Напряжение τ, кПа")
    axes[0].set_title("Циклическая нагрузка")
    axes[0].grid(alpha=0.3)

    axes[1].plot(result["t_hours"], result["eps_elastic"], label="Упругая", lw=1.5)
    axes[1].plot(result["t_hours"], result["eps_viscous"], label="Вязкая (Глен)", lw=1.5)
    axes[1].plot(result["t_hours"], result["eps_total"], label="Суммарная", lw=2, color="black")
    axes[1].set_xlabel("Время, ч")
    axes[1].set_ylabel("Деформация ε")
    axes[1].set_title("Компоненты деформации")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Петля гистерезиса: напряжение vs суммарная деформация
    axes[2].plot(result["eps_total"], result["tau"] / 1e3, color="#e377c2")
    axes[2].set_xlabel("Деформация ε")
    axes[2].set_ylabel("Напряжение τ, кПа")
    axes[2].set_title("Петля гистерезиса (τ vs ε)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Отклик льда на циклическую нагрузку {title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/cyclic_response.png", dpi=150)
    plt.close(fig)


# =============================================================================
# ТОЧКА ВХОДА: запуск всех демонстраций модуля 2
# =============================================================================

if __name__ == "__main__":
    print("Модуль 2: Реология льда и закон Глена")
    print("=" * 50)

    # 1. Проверка закона Глена
    tau_test = 1e5  # 100 кПа
    A_test = arrhenius_A(263.15)  # -10°C
    eps_dot = glen_strain_rate(tau_test, A_test)
    print(f"A(-10°C) = {A_test:.3e} Па^-3 с^-1")
    print(f"При τ = {tau_test:.0f} Па -> ε̇ = {eps_dot:.3e} с^-1")

    # 2. Графики A(T) и вязкости
    plot_arrhenius_curve()
    plot_viscosity_vs_stress()
    plot_dust_effect()
    print("Графики сохранены: arrhenius_A_vs_T.png, viscosity_vs_stress.png, dust_effect.png")

    # 3. Моделирование циклической нагрузки (пример: приливной изгиб шельфа)
    result = simulate_cyclic_response(T_celsius=-10.0, tau_amplitude=1.5e5)
    plot_cyclic_response(result, title_suffix="(T = -10°C, полусуточный цикл)")
    print("График циклического отклика сохранён: cyclic_response.png")

    print("\nГотово. Все результаты в /mnt/user-data/outputs/")
