"""
Лабораторная работа №4 по интервальному анализу.

Интервальные описательные статистики (мода, медианы Крейновича и Пролубникова),
коэффициент Жаккара и оценивание констант a, t в моделях
    a + X = Y,      t * X = Y
по данным диагностики томсоновского рассеяния (файлы *_fast_data.bin).

Запуск:
    python lab4.py <файл X .bin> <файл Y .bin> [папка_для_результатов]
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Параметры
# ----------------------------------------------------------------------------
N = 14
RAD = 1.0 / 2 ** N          # rad x = rad y = 1/2^N
EPS = 1e-7                  # уровень точности eps при поиске argmax F
HEADER_SIZE = 256           # заголовок файла, байт
FRAME_SIZE = 16400          # размер кадра, байт
FRAME_HEADER = 16           # заголовок кадра, байт
N_POINTS, N_CH = 1024, 8    # 1024 точки по 8 каналов


# ----------------------------------------------------------------------------
# Чтение данных (формат из Save_to_BIN.pdf)
# ----------------------------------------------------------------------------
def load_bin(path):
    raw = Path(path).read_bytes()
    side, mode = raw[0], raw[1]
    frame_count = int(np.frombuffer(raw[2:4], dtype="<u2")[0])
    frames = np.frombuffer(raw[HEADER_SIZE:HEADER_SIZE + frame_count * FRAME_SIZE],
                           dtype=np.uint8).reshape(frame_count, FRAME_SIZE)
    stop_point = frames[:, 0:2].copy().view("<u2").ravel()
    codes = frames[:, FRAME_HEADER:].copy().view("<u2").reshape(frame_count, N_POINTS, N_CH)
    volts = codes.astype(np.float64) / 16384.0 - 0.5          # V = Code/16384 - 0.5
    info = dict(side=int(side), mode=int(mode), frame_count=frame_count,
                stop_point_first=stop_point[:5].tolist())
    return volts, info


def make_intervals(values, rad=RAD):
    """x_i = [v_i - rad, v_i + rad]; возвращает массивы нижних и верхних концов."""
    v = np.asarray(values, dtype=np.float64).ravel()
    return v - rad, v + rad


# ----------------------------------------------------------------------------
# Операции арифметики Каухера
# ----------------------------------------------------------------------------
def meet(a_lo, a_hi, b_lo, b_hi):
    """a ∧ b = [max(a_lo, b_lo), min(a_hi, b_hi)] (может быть неправильным)."""
    return np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)


def join(a_lo, a_hi, b_lo, b_hi):
    """a ∨ b = [min(a_lo, b_lo), max(a_hi, b_hi)]."""
    return np.minimum(a_lo, b_lo), np.maximum(a_hi, b_hi)


def jaccard(a_lo, a_hi, b_lo, b_hi):
    """Ji(a, b) = wid(a ∧ b) / wid(a ∨ b),  Ji ∈ [-1, 1]."""
    m_lo, m_hi = meet(a_lo, a_hi, b_lo, b_hi)
    j_lo, j_hi = join(a_lo, a_hi, b_lo, b_hi)
    return (m_hi - m_lo) / (j_hi - j_lo)


def shift(lo, hi, a):
    """a + x = [a + x_lo, a + x_hi]."""
    return lo + a, hi + a


def scale(lo, hi, t):
    """t * x: при t >= 0 -> [t x_lo, t x_hi], при t < 0 -> [t x_hi, t x_lo]."""
    if t >= 0:
        return t * lo, t * hi
    return t * hi, t * lo


# ----------------------------------------------------------------------------
# Интервальные статистики
# ----------------------------------------------------------------------------
def interval_mode(lo, hi):
    """
    Мода интервальной выборки.
    Все концы интервалов упорядочиваются, получаем элементарные интервалы z_j.
    Частота mu_j = число x_i, содержащих z_j. Мода — объединение z_j с max mu_j.
    Возвращает (mode_lo, mode_hi, mu_max, список компонент).
    """
    pts = np.unique(np.concatenate([lo, hi]))
    # число интервалов с lo <= z_lo  минус число интервалов с hi <= z_lo
    lo_s, hi_s = np.sort(lo), np.sort(hi)
    z_lo, z_hi = pts[:-1], pts[1:]
    mu = np.searchsorted(lo_s, z_lo, side="right") - np.searchsorted(hi_s, z_lo, side="right")
    mu_max = mu.max()
    idx = np.where(mu == mu_max)[0]
    # склеиваем соседние элементарные интервалы в компоненты
    comps = []
    start = idx[0]
    for k in range(1, len(idx) + 1):
        if k == len(idx) or idx[k] != idx[k - 1] + 1:
            comps.append((float(z_lo[start]), float(z_hi[idx[k - 1]])))
            if k < len(idx):
                start = idx[k]
    mode_lo = min(c[0] for c in comps)
    mode_hi = max(c[1] for c in comps)
    return mode_lo, mode_hi, int(mu_max), comps, (z_lo, z_hi, mu)


def median_kreinovich(lo, hi):
    """medK X = [med x_lo, med x_hi]."""
    return float(np.median(lo)), float(np.median(hi))


def median_prolubnikov(lo, hi):
    """
    medP X: интервалы упорядочиваются по середине (при равенстве — по радиусу),
    медиана — средний элемент упорядоченной выборки; при чётном n —
    полусумма двух средних элементов (x_(n/2) + x_(n/2+1)) / 2.
    """
    mid, rad = (lo + hi) / 2, (hi - lo) / 2
    order = np.lexsort((rad, mid))
    n = len(order)
    if n % 2 == 1:
        k = order[n // 2]
        return float(lo[k]), float(hi[k])
    k1, k2 = order[n // 2 - 1], order[n // 2]
    return float((lo[k1] + lo[k2]) / 2), float((hi[k1] + hi[k2]) / 2)


# ----------------------------------------------------------------------------
# Функционалы F(s)
# ----------------------------------------------------------------------------
def make_F(param, x_lo, x_hi, y_lo, y_hi, chunk=200_000):
    """
    F(s) = Ji(s, X, Y) — средний по парам (x_i, y_i) коэффициент Жаккара
    Ji(s*x_i или s+x_i, y_i). Для одноэлементных выборок (мода, медианы)
    это просто Ji двух интервалов.
    """
    op = shift if param == "a" else scale
    n = len(x_lo)

    def F(s):
        total = 0.0
        for i in range(0, n, chunk):
            lo, hi = op(x_lo[i:i + chunk], x_hi[i:i + chunk], s)
            total += jaccard(lo, hi, y_lo[i:i + chunk], y_hi[i:i + chunk]).sum()
        return total / n

    return F


def argmax_F(F, s_min, s_max, eps=EPS, n_grid=401):
    """
    s_hat = argmax F(s): грубый поиск на сетке, затем поиск на мелкой сетке
    в окрестности максимума и уточнение методом золотого сечения до точности eps.
    """
    for _ in range(2):
        grid = np.linspace(s_min, s_max, n_grid)
        vals = np.array([F(s) for s in grid])
        k = int(np.argmax(vals))
        h = grid[1] - grid[0]
        s_min, s_max = grid[k] - 5 * h, grid[k] + 5 * h
    a, b = grid[k] - h, grid[k] + h
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = F(c), F(d)
    while b - a > eps:
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = F(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = F(d)
    s_hat = (a + b) / 2
    return s_hat, F(s_hat)


def level_set(F, s_hat, level, s_min, s_max, eps=EPS):
    """
    Интервальная оценка: связная компонента множества {s : F(s) >= level},
    содержащая s_hat. Концы находятся бисекцией с точностью eps.
    """
    if F(s_hat) < level:
        return None

    def bisect(inside, outside):
        while abs(outside - inside) > eps:
            m = (inside + outside) / 2
            if F(m) >= level:
                inside = m
            else:
                outside = m
        return inside

    left = s_min if F(s_min) >= level else bisect(s_hat, s_min)
    right = s_max if F(s_max) >= level else bisect(s_hat, s_max)
    return left, right


# ----------------------------------------------------------------------------
# Основной расчёт
# ----------------------------------------------------------------------------
def main(file_x, file_y, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vx, info_x = load_bin(file_x)
    vy, info_y = load_bin(file_y)
    x_lo, x_hi = make_intervals(vx)
    y_lo, y_hi = make_intervals(vy)
    n = len(x_lo)
    print(f"n = {n}, rad = 2^-{N} = {RAD:.6e} В")

    res = dict(N=N, rad=RAD, eps=EPS, n=n, info_x=info_x, info_y=info_y)
    res["X_stats"] = dict(min=float(vx.min()), max=float(vx.max()),
                          mean=float(vx.mean()), median=float(np.median(vx)), std=float(vx.std()))
    res["Y_stats"] = dict(min=float(vy.min()), max=float(vy.max()),
                          mean=float(vy.mean()), median=float(np.median(vy)), std=float(vy.std()))

    # --- Статистики ---------------------------------------------------------
    mx = interval_mode(x_lo, x_hi)
    my = interval_mode(y_lo, y_hi)
    stats = {
        "mode": ((mx[0], mx[1]), (my[0], my[1])),
        "medK": (median_kreinovich(x_lo, x_hi), median_kreinovich(y_lo, y_hi)),
        "medP": (median_prolubnikov(x_lo, x_hi), median_prolubnikov(y_lo, y_hi)),
    }
    res["mode_X"] = dict(interval=[mx[0], mx[1]], mu=mx[2], components=mx[3])
    res["mode_Y"] = dict(interval=[my[0], my[1]], mu=my[2], components=my[3])
    for key in ("medK", "medP"):
        res[key + "_X"] = list(stats[key][0])
        res[key + "_Y"] = list(stats[key][1])
    for k, v in stats.items():
        print(f"{k}: X = [{v[0][0]:.8f}, {v[0][1]:.8f}],  Y = [{v[1][0]:.8f}, {v[1][1]:.8f}]")

    plot_data(vx, vy, mx, my, stats, out)

    # --- Оценивание ---------------------------------------------------------
    ranges = {"a": (0.30, 0.39), "t": (-1.15, -0.88)}
    cases = {
        "B1": ("Ji(s, X, Y)", (x_lo, x_hi, y_lo, y_hi)),
        "B2": ("Ji(s, mode X, mode Y)", tuple(np.array([c]) for c in (*stats["mode"][0], *stats["mode"][1]))),
        "B3": ("Ji(s, medK X, medK Y)", tuple(np.array([c]) for c in (*stats["medK"][0], *stats["medK"][1]))),
        "B4": ("Ji(s, medP X, medP Y)", tuple(np.array([c]) for c in (*stats["medP"][0], *stats["medP"][1]))),
    }
    res["estimates"] = {}
    for case, (title, data) in cases.items():
        for param in ("a", "t"):
            F = make_F(param, *data)
            s_min, s_max = ranges[param]
            s_hat, F_hat = argmax_F(F, s_min, s_max)
            # Интервальная оценка — множество {s : F(s) >= level}.
            # Для двух интервалов (B2-B4) level = 0: F >= 0 <=> интервалы пересекаются.
            # Для B1 max F < 0, поэтому берётся уровень половины высоты пика
            # над минимально возможным значением Ji = -1: level = (F_max - 1) / 2
            # (при F_max = 1 он совпадает с уровнем 0).
            level = 0.0 if case != "B1" else (F_hat - 1) / 2
            est = level_set(F, s_hat, level, s_min, s_max)
            r = dict(F=title, s_hat=s_hat, F_max=F_hat, level=level,
                     interval=[float(est[0]), float(est[1])] if est else None)
            res["estimates"][f"{case}_{param}"] = r
            print(f"{case} {param}: s_hat = {s_hat:.7f}, F_max = {F_hat:.6f}, "
                  f"level = {level:.4f}, interval = {r['interval']}")
            plot_F(F, case, param, title, s_hat, F_hat, est, level, out)

    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return res


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def plot_data(vx, vy, mx, my, stats, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, v, m, name, (k, c) in ((axes[0], vx, mx, "X", (0, "C0")), (axes[1], vy, my, "Y", (1, "C1"))):
        med = np.median(v)
        step = 4.0 / 16384                       # ширина столбца — 4 кванта АЦП
        bins = np.arange(med - 0.04, med + 0.04 + step, step)
        ax.hist(v.ravel(), bins=bins, color=c, alpha=0.75)
        ax.set_xlim(med - 0.04, med + 0.04)
        ax.axvspan(m[0], m[1], color="red", alpha=0.6, label="mode")
        ax.axvline(np.mean(stats["medK"][k]), color="k", ls="--", lw=1, label="medK = medP (mid)")
        ax.set_title(f"Выборка {name}: гистограмма значений, В")
        ax.set_xlabel("V, В")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hist_XY.png", dpi=150)
    plt.close(fig)

    # частота mu(z) около моды
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, name, c in ((axes[0], mx, "X", "C0"), (axes[1], my, "Y", "C1")):
        z_lo, z_hi, mu = m[4]
        sel = (z_lo > m[0] - 0.02) & (z_hi < m[1] + 0.02)
        ax.step(z_lo[sel], mu[sel], where="post", color=c)
        ax.axvspan(m[0], m[1], color="red", alpha=0.4, label=f"mode, μ = {m[2]}")
        ax.set_title(f"Частота μ(z) для {name}")
        ax.set_xlabel("z, В")
        ax.set_ylabel("μ")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mu_XY.png", dpi=150)
    plt.close(fig)


def plot_F(F, case, param, title, s_hat, F_hat, est0, level, out):
    if case == "B1":
        half = 1.3 * max(s_hat - est0[0], est0[1] - s_hat)
        n_pts = 601
    else:
        width = (est0[1] - est0[0]) if est0 else 1e-3
        half = max(1.5 * width, 2e-4)
        n_pts = 2001
    s = np.linspace(s_hat - half, s_hat + half, n_pts)
    f = np.array([F(v) for v in s])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, f, lw=1.5)
    if case != "B1":
        ax.axhline(0, color="gray", lw=0.8)
    ax.plot([s_hat], [F_hat], "ro", label=f"${param}_{{max}}$ = {s_hat:.6f}, F = {F_hat:.4f}")
    if est0:
        ax.axvspan(est0[0], est0[1], color="green", alpha=0.15,
                   label=f"F({param}) ≥ {level:.3f}: [{est0[0]:.6f}, {est0[1]:.6f}]")
        if level != 0:
            ax.axhline(level, color="green", lw=0.8, ls="--")
    ax.set_xlabel(param)
    ax.set_ylabel("F")
    ax.set_title(f"{case}: F({param}) = {title.replace('s', param, 1)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"F_{case}_{param}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "results")
