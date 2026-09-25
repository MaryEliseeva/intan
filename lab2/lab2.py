# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Interval:
    def __init__(self, low, high=None):
        if high is None:
            high = low
        self.low = float(low)
        self.high = float(high)

    @property
    def mid(self):
        return 0.5 * (self.low + self.high)

    @property
    def rad(self):
        return 0.5 * (self.high - self.low)

    @property
    def wid(self):
        return self.high - self.low

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Interval(self.low + other, self.high + other)
        return Interval(self.low + other.low, self.high + other.high)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(self.low - other, self.high - other)
        return Interval(self.low - other.high, self.high - other.low)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(other - self.high, other - self.low)
        return Interval(other.low - self.high, other.high - self.low)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            p = [self.low * other, self.high * other]
            return Interval(min(p), max(p))
        p = [self.low * other.low, self.low * other.high,
             self.high * other.low, self.high * other.high]
        return Interval(min(p), max(p))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        if not isinstance(power, int):
            raise ValueError("Поддерживаются только целые степени")
        if power == 0:
            return Interval(1.0, 1.0)
        if power % 2 == 1:
            return Interval(self.low ** power, self.high ** power)
        else:
            if self.low <= 0 <= self.high:
                return Interval(0.0, max(abs(self.low), abs(self.high)) ** power)
            else:
                vals = [self.low ** power, self.high ** power]
                return Interval(min(vals), max(vals))

    def sin(self):
        grid = np.linspace(self.low, self.high, 1000)
        sin_grid = np.sin(grid)
        return Interval(np.min(sin_grid), np.max(sin_grid))

    def intersect(self, other):
        if other is None:
            return self
        l = max(self.low, other.low)
        h = min(self.high, other.high)
        if l > h:
            return None
        return Interval(l, h)

    def __repr__(self):
        return f"[{self.low:.6f}, {self.high:.6f}]"


def hausdorff_dist(A, B):
    return max(abs(A.low - B.low), abs(A.high - B.high))


def cut(val, bounds=(-1.0, 1.0)):
    return max(bounds[0], min(bounds[1], val))


# ==========================================
# 1. ФУНКЦИЯ f1(x) = x^3 - 3x^2 + 2
# ==========================================

def solve_f1():
    print("=" * 70)
    print("ЧАСТЬ 1.1: Функция f1(x) = x^3 - 3x^2 + 2 на X1 = [0, 3]")
    print("=" * 70)

    X1 = Interval(0.0, 3.0)
    ran_f1 = Interval(-2.0, 2.0)

    def f1_val(x):
        return x ** 3 - 3 * x ** 2 + 2

    f1_eir = X1 ** 3 - 3 * (X1 ** 2) + 2
    f1_horner = (X1 ** 2) * (X1 - 3.0) + 2
    Y1 = Interval(-1.0, 2.0)
    f1_shift = Y1 ** 3 - 3 * Y1

    df1_X1 = 3 * X1 * (X1 - 2.0)  # [-18, 9]

    # Исследование F_mv в различных точках
    centers_f1 = [0.0, 0.75, 1.0, 1.5, 2.0, 2.25, 3.0]
    mv_results = []
    print("\n--- Дифференциальная центрированная форма F_mv(X1, c) ---")
    for c in centers_f1:
        res = f1_val(c) + df1_X1 * (X1 - c)
        mv_results.append(res)
        print(f"c = {c:4.2f}: F_mv = {res}, wid = {res.wid:6.3f}, dist = {hausdorff_dist(res, ran_f1):6.3f}")

    inter_mv = mv_results[0]
    for r in mv_results[1:]:
        inter_mv = inter_mv.intersect(r)
    print(f"ПЕРЕСЕЧЕНИЕ всех F_mv: {inter_mv}, dist = {hausdorff_dist(inter_mv, ran_f1):.4f}")

    # Наклонная форма
    def slope_f1(X, c):
        return X ** 2 + (c - 3.0) * X + (c ** 2 - 3.0 * c)

    print("\n--- Наклонная центрированная форма F_sl(X1, c) ---")
    for c in [0.0, 1.5, 3.0]:
        sl_int = slope_f1(X1, c)
        f_sl = f1_val(c) + sl_int * (X1 - c)
        print(f"c = {c:4.2f}: slope = {sl_int}, F_sl = {f_sl}, dist = {hausdorff_dist(f_sl, ran_f1):.4f}")

    # Бауманн
    p1 = cut(df1_X1.mid / df1_X1.rad)  # -1/3
    c_star_low = X1.mid - p1 * X1.rad  # 2.0
    c_star_high = X1.mid + p1 * X1.rad  # 1.0
    f1_mv_low = f1_val(c_star_low) + df1_X1 * (X1 - c_star_low)
    f1_mv_high = f1_val(c_star_high) + df1_X1 * (X1 - c_star_high)
    f1_bic = f1_mv_low.intersect(f1_mv_high)

    print(f"\nБауманн F_bic(X1) = F_mv(c_*) ∩ F_mv(c^*): {f1_bic}, dist = {hausdorff_dist(f1_bic, ran_f1):.4f}")

    # График
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = np.linspace(0, 3, 500)
    y_vals = f1_val(x_vals)

    ax.plot(x_vals, y_vals, 'b-', lw=2.5, label=r'$f_1(x) = x^3 - 3x^2 + 2$')
    ax.axhline(ran_f1.low, color='green', linestyle='--', label=r'Точная область $[-2, 2]$')
    ax.axhline(ran_f1.high, color='green', linestyle='--')
    ax.axhspan(f1_eir.low, f1_eir.high, alpha=0.1, color='red', label=r'ЕИР $[-25, 29]$')
    ax.axhspan(f1_bic.low, f1_bic.high, alpha=0.2, color='orange', label=r'Бауманн $[-20, 18]$')

    ax.set_title(r'График $f_1(x)$ и интервальные оценки на $[0, 3]$', fontsize=12)
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fig_f1.png', dpi=150)
    plt.close()


# ==========================================
# 2. ФУНКЦИЯ f2(x) = x^4 - 2x^2 + sin(x)
# ==========================================

def solve_f2():
    print("\n" + "=" * 70)
    print("ЧАСТЬ 1.2: Функция f2(x) = x^4 - 2x^2 + sin(x) на X2 = [-1.5, 1.5]")
    print("=" * 70)

    X2 = Interval(-1.5, 1.5)

    def f2_val(x):
        return x ** 4 - 2 * x ** 2 + np.sin(x)

    x_grid = np.linspace(-1.5, 1.5, 100000)
    y_grid = f2_val(x_grid)
    ran_f2 = Interval(np.min(y_grid), np.max(y_grid))

    f2_eir = X2 ** 4 - 2 * (X2 ** 2) + X2.sin()
    f2_horner = (X2 ** 2 - 2.0) * (X2 ** 2) + X2.sin()
    f2_sq = (X2 ** 2 - 1.0) ** 2 - 1.0 + X2.sin()

    df2_X2 = Interval(-19.429263, 20.500000)

    # F_mv при разных c
    centers_f2 = [-1.5, -0.75, -0.0402, 0.0, 0.0402, 0.75, 1.5]
    mv_results_f2 = []
    print("\n--- Дифференциальная центрированная форма F_mv(X2, c) ---")
    for c in centers_f2:
        res = f2_val(c) + df2_X2 * (X2 - c)
        mv_results_f2.append(res)
        print(f"c = {c:7.4f}: F_mv = {res}, wid = {res.wid:6.3f}, dist = {hausdorff_dist(res, ran_f2):6.3f}")

    inter_mv_f2 = mv_results_f2[0]
    for r in mv_results_f2[1:]:
        inter_mv_f2 = inter_mv_f2.intersect(r)
    print(f"ПЕРЕСЕЧЕНИЕ всех F_mv: {inter_mv_f2}, dist = {hausdorff_dist(inter_mv_f2, ran_f2):.4f}")

    # Наклонная форма F_sl для f2
    cos_X2 = Interval(np.cos(1.5), 1.0)

    def slope_f2(X, c):
        return X ** 3 + c * (X ** 2) + (c ** 2) * X + (c ** 3) - 2 * (X + c) + cos_X2

    print("\n--- Наклонная центрированная форма F_sl(X2, c) ---")
    for c in [-1.5, -0.75, 0.0, 0.75, 1.5]:
        sl_int = slope_f2(X2, c)
        f_sl = f2_val(c) + sl_int * (X2 - c)
        print(
            f"c = {c:5.2f}: slope = {sl_int}, F_sl = {f_sl}, wid = {f_sl.wid:6.3f}, dist = {hausdorff_dist(f_sl, ran_f2):.4f}")

    p2 = cut(df2_X2.mid / df2_X2.rad)
    c_star_low = X2.mid - p2 * X2.rad  # -0.0402
    c_star_high = X2.mid + p2 * X2.rad  # 0.0402
    f2_mv_low = f2_val(c_star_low) + df2_X2 * (X2 - c_star_low)
    f2_mv_high = f2_val(c_star_high) + df2_X2 * (X2 - c_star_high)
    f2_bic = f2_mv_low.intersect(f2_mv_high)

    print(f"\nБауманн F_bic(X2) = {f2_bic}, dist = {hausdorff_dist(f2_bic, ran_f2):.4f}")

    # График
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, y_grid, 'm-', lw=2.5, label=r'$f_2(x) = x^4 - 2x^2 + \sin(x)$')
    ax.axhline(ran_f2.low, color='green', linestyle='--', label=f'Точный мин: {ran_f2.low:.3f}')
    ax.axhline(ran_f2.high, color='green', linestyle='--', label=f'Точный макс: {ran_f2.high:.3f}')
    ax.axhspan(f2_sq.low, f2_sq.high, alpha=0.25, color='cyan', label=r'Полный квадрат $[-2.00, 1.56]$')

    ax.set_title(r'График $f_2(x)$ и точнейшая оценка через полный квадрат', fontsize=12)
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fig_f2.png', dpi=150)
    plt.close()


# ==========================================
# 3. ФУНКЦИЯ f3(x, y) = x^2 + 2y^2 - 2xy + x
# ==========================================

def solve_f3():
    print("\n" + "=" * 70)
    print("ЧАСТЬ 2: Функция f3(x, y) = x^2 + 2y^2 - 2xy + x на X = [-2, 1] x [-1, 2]")
    print("=" * 70)

    X1 = Interval(-2.0, 1.0)
    X2 = Interval(-1.0, 2.0)

    def f3_val(x, y):
        return x ** 2 + 2 * y ** 2 - 2 * x * y + x

    ran_f3 = Interval(-0.5, 18.0)

    f3_eir = X1 ** 2 + 2 * (X2 ** 2) - 2 * X1 * X2 + X1
    f3_eq1 = (X1 - X2) ** 2 + X2 ** 2 + X1
    f3_eq2 = (X1 - X2 + 0.5) ** 2 + X2 ** 2 - 0.25

    df3_dx = Interval(-7.0, 5.0)
    df3_dy = Interval(-6.0, 12.0)

    centers_f3 = [
        (-0.5, 0.5),  # mid
        (-2.0, -1.0),  # LL
        (1.0, 2.0),  # UR
        (-0.25, 0.0),  # c_*
        (-0.75, 1.0),  # c^*
        (-1.0, -0.5),  # exact min
        (-2.0, 2.0)  # exact max
    ]
    mv_results_f3 = []
    print("\n--- Дифференциальная центрированная форма F_mv(X3, c) ---")
    for (cx, cy) in centers_f3:
        res = f3_val(cx, cy) + df3_dx * (X1 - cx) + df3_dy * (X2 - cy)
        mv_results_f3.append(res)
        print(
            f"c = ({cx:5.2f}, {cy:5.2f}): F_mv = {res}, wid = {res.wid:6.3f}, dist = {hausdorff_dist(res, ran_f3):6.3f}")

    inter_mv_f3 = mv_results_f3[0]
    for r in mv_results_f3[1:]:
        inter_mv_f3 = inter_mv_f3.intersect(r)
    print(f"ПЕРЕСЕЧЕНИЕ всех F_mv: {inter_mv_f3}, dist = {hausdorff_dist(inter_mv_f3, ran_f3):.4f}")

    # Бауманн
    px = cut(df3_dx.mid / df3_dx.rad)  # -1/6
    py = cut(df3_dy.mid / df3_dy.rad)  # 1/3
    c_star_low = (X1.mid - px * X1.rad, X2.mid - py * X2.rad)  # (-0.25, 0.0)
    c_star_high = (X1.mid + px * X1.rad, X2.mid + py * X2.rad)  # (-0.75, 1.0)
    f3_mv_low = f3_val(*c_star_low) + df3_dx * (X1 - c_star_low[0]) + df3_dy * (X2 - c_star_low[1])
    f3_mv_high = f3_val(*c_star_high) + df3_dx * (X1 - c_star_high[0]) + df3_dy * (X2 - c_star_high[1])
    f3_bic = f3_mv_low.intersect(f3_mv_high)

    print(f"\nБауманн F_bic(X3) = {f3_bic}, dist = {hausdorff_dist(f3_bic, ran_f3):.4f}")

    # График
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(-2.5, 1.5, 300)
    y = np.linspace(-1.5, 2.5, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z_grid = f3_val(X_grid, Y_grid)

    contours = ax.contour(X_grid, Y_grid, Z_grid, levels=15, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)

    rect = plt.Rectangle((-2, -1), 3, 3, linewidth=2, edgecolor='red', facecolor='none', label='Брус X')
    ax.add_patch(rect)

    ax.plot(-1.0, -0.5, 'ro', markersize=8, label=r'Минимум $(-1, -0.5), f = -0.5$')
    ax.plot(-2.0, 2.0, 'go', markersize=8, label=r'Максимум $(-2, 2), f = 18.0$')

    ax.set_title(r'Линии уровня $f_3(x, y) = x^2 + 2y^2 - 2xy + x$ и область $X$', fontsize=12)
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('fig_f3.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    solve_f1()
    solve_f2()
    solve_f3()
