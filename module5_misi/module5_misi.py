"""
Модуль 5: Стабильность морских ледниковых щитов (MISI)
==========================================================

Курс по гляциологии — практическая реализация на Python.

Содержание:
1. Поток льда через линию налегания (grounding line flux) по теории
   пограничного слоя Шуфа (Schoof, 2007) — как функция толщины льда на
   линии налегания и параметров базального трения.
2. Условие флотации и связь между геометрией ложа и толщиной на линии
   налегания.
3. Линейный анализ устойчивости положения линии налегания: поиск
   устойчивых и неустойчивых стационарных состояний.
4. Демонстрация морской ледниковой нестабильности (MISI): на обратном
   (ретроградном) уклоне ложа устойчивых состояний нет — линия налегания
   "проваливается" в неконтролируемое отступание; гистерезис при
   медленном изменении климатического форсинга.

Требуемые библиотеки: numpy, matplotlib, scipy
Установка: pip install numpy matplotlib scipy

Ключевая ссылка:
- Schoof, C. (2007). Ice sheet grounding line dynamics: Steady states,
  stability, and hysteresis. J. Geophys. Res., 112, F03S28.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

RHO_ICE = 917.0
RHO_WATER = 1028.0
G = 9.81
SEC_PER_YEAR = 365.25 * 24 * 3600

# =============================================================================
# 1. ПОТОК ЧЕРЕЗ ЛИНИЮ НАЛЕГАНИЯ (SCHOOF, 2007)
# =============================================================================
#
# Теория пограничного слоя Шуфа даёт аналитическое выражение для потока
# массы льда через линию налегания как функцию ТОЛЬКО толщины льда на
# линии налегания H_g и параметров трения ложа (закон скольжения
# Веертмана: tau_b = C * u^(1/m)):
#
#   q_g = [ A (rho_i g)^(n+1) (1 - rho_i/rho_w)^n / (4^n C) ]^(1/(m+1))
#         * H_g^((m+n+3)/(m+1))
#
# Ключевой физический смысл: q_g РЕЗКО растёт с толщиной H_g (типичный
# показатель степени (m+n+3)/(m+1) при m=1/3, n=3 даёт показатель ~4.7)
# — именно эта сильная зависимость лежит в основе неустойчивости MISI.


def schoof_grounding_line_flux(H_g, C=7.6e6, A=2.4e-24, m=1.0 / 3.0, n=3.0):
    """
    Поток льда через линию налегания по Шуфу (2007), м^2/с (поток на
    единицу ширины).

    Параметры
    ----------
    H_g : float или np.ndarray
        Толщина льда на линии налегания, м
    C : float
        Коэффициент трения в законе скольжения Веертмана
        (tau_b = C * u^(1/m)), Па (м/с)^(-1/m)
    A : float
        Коэффициент мягкости льда (закон Глена), Па^-n с^-1
    m : float
        Показатель степени закона скольжения (m=1/3 — типичное значение
        Веертмана; m=1 соответствует линейному/вязкому скольжению)
    n : float
        Показатель степени закона Глена
    """
    exponent_prefactor = 1.0 / (m + 1.0)
    exponent_H = (m + n + 3.0) / (m + 1.0)

    prefactor = (A * (RHO_ICE * G) ** (n + 1) * (1 - RHO_ICE / RHO_WATER) ** n
                 / (4.0 ** n * C)) ** exponent_prefactor

    return prefactor * H_g ** exponent_H


def plot_schoof_flux_curve():
    """Демонстрация резкой (степенной) зависимости q_g от H_g."""
    H_g = np.linspace(200, 2500, 300)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for C, label, color in zip([3e6, 7.6e6, 2e7],
                                 ["мягкое ложе (C=3e6)", "среднее (C=7.6e6)", "жёсткое ложе (C=2e7)"],
                                 ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        q_g = schoof_grounding_line_flux(H_g, C=C) * SEC_PER_YEAR / 1e3  # тыс. м^2/год
        ax.plot(H_g, q_g, lw=2, color=color, label=label)

    ax.set_xlabel("Толщина льда на линии налегания H_g, м")
    ax.set_ylabel("Поток q_g, тыс. м²/год")
    ax.set_title("Поток через линию налегания (Schoof, 2007): резкая\nзависимость от толщины — основа неустойчивости MISI")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/schoof_flux_curve.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 2. ГЕОМЕТРИЯ ЛОЖА И УСЛОВИЕ ФЛОТАЦИИ
# =============================================================================
#
# На линии налегания лёд находится на грани флотации:
#   rho_i * H_g = rho_w * (-b(x_g))    =>    H_g(x_g) = -b(x_g) * rho_w/rho_i
# (при условии b(x_g) < 0, т.е. ложе ниже уровня моря). ВАЖНО: здесь
# координата x растёт ВГЛУБЬ МАТЕРИКА (от побережья/океана к внутренним
# районам ледникового щита) — этой конвенции придерживаются обе функции
# ложа ниже, а также вывод критерия устойчивости в разделе 3.


def bed_prograde(x):
    """
    "Нормальный" (прямой/prograde) профиль ложа: монотонно ПОДНИМАЕТСЯ
    (мелеет) по мере удаления от побережья вглубь материка — типичная
    стабилизирующая геометрия (бедрок ледникового щита обычно выше в
    его внутренних районах, вдали от прибрежных переуглублений).
    """
    return -600.0 + 0.003 * x  # x в метрах: глубоко у "побережья" (x=0), мелеет вглубь материка


def bed_retrograde_basin(x):
    """
    Волнообразное ложе с чередующимися порогами (риджелями) и
    переуглублёнными бассейнами возрастающей глубины — качественный
    аналог сложного рельефа ложа моря Амундсена (Западная Антарктида),
    где несколько последовательных порогов создают серию потенциальных
    точек "зацепления" линии налегания. Именно такая волнообразная
    структура (а не одиночный изолированный бассейн) необходима для
    демонстрации ПОДЛИННОЙ биустойчивости — одновременного сосуществования
    двух и более устойчивых положений линии налегания при одном и том же
    внешнем форсинге Q_in, что и порождает гистерезис при его изменении.
    """
    x_km = x / 1000.0
    return -300.0 - 0.001 * x + 350.0 * np.sin(2 * np.pi * x_km / 250.0 + 0.3)


def grounding_line_thickness(x, bed_func):
    """Толщина льда на линии налегания из условия флотации, м."""
    b = bed_func(x)
    return np.maximum(-b, 1.0) * RHO_WATER / RHO_ICE


# =============================================================================
# 3. ПОИСК СТАЦИОНАРНЫХ СОСТОЯНИЙ И ЛИНЕЙНЫЙ АНАЛИЗ УСТОЙЧИВОСТИ
# =============================================================================
#
# Упрощённая (box-model) динамика положения линии налегания x_g. Так как
# координата x растёт ВГЛУБЬ МАТЕРИКА (см. выше), баланс потоков
# записывается как:
#
#   dx_g/dt = k * (q_g(x_g) - Q_in)
#
# где Q_in — поток льда, поступающий из внутренних районов ледникового
# щита (заданный внешний форсинг, определяемый SIA/SSA динамикой выше по
# потоку), k>0 — константа, переводящая дисбаланс потоков в скорость
# смещения линии налегания. Логика знака: если поток НА ВЫХОДЕ (q_g)
# больше притока (Q_in), лёд "не успевает" компенсироваться -> линия
# налегания ОТСТУПАЕТ вглубь материка (x_g растёт).
#
# Стационарное состояние: q_g(x_g*) = Q_in.
# Устойчивость (линеаризация): состояние устойчиво, если
#   d/dx_g [q_g(x_g) - Q_in] < 0    <=>    dq_g/dx_g < 0 при x_g*
# (Физически: если линию налегания сместить чуть глубже вглубь материка
# (+x), поток на выходе должен УМЕНЬШИТЬСЯ, чтобы вернуть систему к
# равновесию — отрицательная обратная связь. Если же при смещении
# вглубь материка q_g, наоборот, РАСТЁТ — положительная обратная связь
# запускает неограниченное отступание: это и есть механизм MISI.)


def find_equilibria(bed_func, Q_in, x_range=(1000, 900000), n_scan=20000, **schoof_kwargs):
    """
    Находит все стационарные положения линии налегания (корни
    q_g(x) - Q_in = 0) методом сканирования знака с уточнением через
    brentq, и классифицирует их устойчивость (см. критерий выше:
    устойчиво <=> dq_g/dx < 0, поскольку x растёт вглубь материка).
    """
    x_scan = np.linspace(*x_range, n_scan)
    H_g_scan = grounding_line_thickness(x_scan, bed_func)
    q_g_scan = schoof_grounding_line_flux(H_g_scan, **schoof_kwargs)
    residual = q_g_scan - Q_in

    roots = []
    for i in range(len(x_scan) - 1):
        if residual[i] == 0:
            roots.append(x_scan[i])
        elif residual[i] * residual[i + 1] < 0:
            root = brentq(lambda x: schoof_grounding_line_flux(
                grounding_line_thickness(x, bed_func), **schoof_kwargs) - Q_in,
                x_scan[i], x_scan[i + 1])
            roots.append(root)

    equilibria = []
    dx = 1.0  # м, для численной производной
    for x_star in roots:
        H_plus = grounding_line_thickness(x_star + dx, bed_func)
        H_minus = grounding_line_thickness(x_star - dx, bed_func)
        q_plus = schoof_grounding_line_flux(H_plus, **schoof_kwargs)
        q_minus = schoof_grounding_line_flux(H_minus, **schoof_kwargs)
        dq_dx = (q_plus - q_minus) / (2 * dx)
        stable = dq_dx < 0  # x растёт вглубь материка (см. вывод критерия выше)
        equilibria.append({"x_g": x_star, "H_g": grounding_line_thickness(x_star, bed_func),
                             "dq_dx": dq_dx, "stable": stable})

    return equilibria


def plot_misi_stability(bed_func, Q_in, title, filename, x_range=(1000, 900000)):
    """
    Строит профиль ложа и поток q_g(x) в сравнении с заданным Q_in,
    отмечая устойчивые (закрашенные) и неустойчивые (полые) стационарные
    точки.
    """
    x = np.linspace(*x_range, 3000)
    b = bed_func(x)
    H_g = grounding_line_thickness(x, bed_func)
    q_g = schoof_grounding_line_flux(H_g) * SEC_PER_YEAR / 1e3

    equilibria = find_equilibria(bed_func, Q_in, x_range=x_range)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x / 1000, b, color="#8c510a", lw=2, label="Ложе b(x)")
    axes[0].axhline(0, color="#4393c3", ls="--", lw=1, label="Уровень моря")
    axes[0].fill_between(x / 1000, b, 0, where=(b < 0), color="#c6dbef", alpha=0.4)
    axes[0].set_ylabel("Высота ложа, м")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x / 1000, q_g, color="#1f77b4", lw=2, label="q_g(x) — поток по Шуфу")
    axes[1].axhline(Q_in * SEC_PER_YEAR / 1e3, color="#d62728", ls="--", lw=1.5,
                     label=f"Q_in = {Q_in*SEC_PER_YEAR/1e3:.1f} тыс. м²/год")

    for eq in equilibria:
        q_val = schoof_grounding_line_flux(eq["H_g"]) * SEC_PER_YEAR / 1e3
        marker_style = dict(color="#2ca02c", s=90, zorder=5) if eq["stable"] else dict(
            facecolors="white", edgecolors="#d62728", s=90, linewidths=2, zorder=5)
        axes[1].scatter([eq["x_g"] / 1000], [q_val], **marker_style)

    axes[1].set_xlabel("Расстояние от условного начала отсчёта, км")
    axes[1].set_ylabel("Поток, тыс. м²/год")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"/mnt/user-data/outputs/{filename}", dpi=150)
    plt.close(fig)

    return equilibria


# =============================================================================
# 4. ГИСТЕРЕЗИС: МЕДЛЕННОЕ ИЗМЕНЕНИЕ КЛИМАТИЧЕСКОГО ФОРСИНГА
# =============================================================================
#
# Постепенно увеличиваем, а затем уменьшаем Q_in (в этой упрощённой
# box-модели Q_in играет роль внешнего форсинга, зависящего, например,
# от поступления массы льда из внутренних районов) и отслеживаем
# устойчивую ветвь равновесий. При прохождении седло-узловой
# бифуркации (стационарные состояния исчезают) линия налегания
# "срывается" в быстрое отступание до следующей устойчивой ветви —
# классический гистерезис MISI (ср. Schoof, 2007, Fig. 7-9).


def compute_hysteresis_branch(bed_func, Q_in_range, x_range=(1000, 900000)):
    """
    Для последовательности значений Q_in находит устойчивое положение
    линии налегания, предполагая непрерывное отслеживание ближайшей
    устойчивой ветви (адиабатическое изменение форсинга).
    """
    x_g_track = []
    x_prev = None

    for Q_in in Q_in_range:
        equilibria = find_equilibria(bed_func, Q_in, x_range=x_range)
        stable_eqs = [eq for eq in equilibria if eq["stable"]]
        if not stable_eqs:
            x_g_track.append(np.nan)  # нет устойчивого состояния -> "обвал"
            continue
        if x_prev is None:
            chosen = stable_eqs[0]
        else:
            chosen = min(stable_eqs, key=lambda eq: abs(eq["x_g"] - x_prev))
        x_g_track.append(chosen["x_g"])
        x_prev = chosen["x_g"]

    return np.array(x_g_track)


def plot_hysteresis(bed_func):
    Q_in_up = np.linspace(0.3e6, 1.2e7, 150) / SEC_PER_YEAR
    Q_in_down = Q_in_up[::-1]

    x_up = compute_hysteresis_branch(bed_func, Q_in_up)
    x_down = compute_hysteresis_branch(bed_func, Q_in_down)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Q_in_up * SEC_PER_YEAR / 1e3, x_up / 1000, color="#d62728", lw=2, label="Рост Q_in (потепление)")
    ax.plot(Q_in_down * SEC_PER_YEAR / 1e3, x_down / 1000, color="#1f77b4", lw=2, ls="--",
            label="Снижение Q_in (похолодание)")

    ax.set_xlabel("Q_in, тыс. м²/год (форсинг притока льда)")
    ax.set_ylabel("Устойчивое положение линии налегания x_g, км")
    ax.set_title("Гистерезис MISI на ретроградном ложе (ср. Schoof, 2007)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/misi_hysteresis.png", dpi=150)
    plt.close(fig)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    print("Модуль 5: Стабильность морских ледниковых щитов (MISI)")
    print("=" * 60)

    print("1. Поток через линию налегания (Schoof, 2007)...")
    plot_schoof_flux_curve()

    Q_in_demo = 2.0e6 / SEC_PER_YEAR  # м^2/с (выбрано так, чтобы попасть в зону биустойчивости)

    print("2. Устойчивость на прямом (prograde) ложе...")
    eq_prograde = plot_misi_stability(
        bed_prograde, Q_in_demo,
        "Прямой (prograde) профиль ложа — единственное устойчивое состояние",
        "misi_prograde.png",
    )
    for eq in eq_prograde:
        print(f"   x_g = {eq['x_g']/1000:.1f} км, H_g = {eq['H_g']:.0f} м, "
              f"{'УСТОЙЧИВО' if eq['stable'] else 'неустойчиво'}")

    print("3. Устойчивость на ретроградном ложе (переуглублённый бассейн)...")
    eq_retro = plot_misi_stability(
        bed_retrograde_basin, Q_in_demo,
        "Ретроградный профиль ложа — множественные равновесия (MISI)",
        "misi_retrograde.png",
    )
    for eq in eq_retro:
        print(f"   x_g = {eq['x_g']/1000:.1f} км, H_g = {eq['H_g']:.0f} м, "
              f"{'УСТОЙЧИВО' if eq['stable'] else 'неустойчиво'}")

    print("4. Гистерезис при изменении климатического форсинга...")
    plot_hysteresis(bed_retrograde_basin)

    print("\nГотово. Графики сохранены в /mnt/user-data/outputs/")
