# -*- coding: utf-8 -*-

"""
Лабораторная работа №1
По интервальному анализу

Исследование вырожденности интервальных матриц.

Рассматриваются:
1. Прямоугольная матрица A1:
   - линейная регрессия (радиус только в первом столбце);
   - малоракурсная томография (радиус в обоих столбцах).
2. Квадратная матрица A2.

Для каждого случая:
- находится минимальное значение delta;
- строится точечная матрица A';
- проверяется A' принадлежит интервальной матрице;
- проверяется вырожденность;
- исследуется поведение при delta меньше/равно/больше критического.
"""

import itertools
import numpy as np


EPS = 1e-12


# ============================================================
# Общие функции
# ============================================================

def interval_contains(matrix, center, radius, eps=EPS):
    """Проверяет принадлежность точечной матрицы интервальной."""
    lower = center - radius
    upper = center + radius

    return np.all(
        (matrix >= lower - eps) &
        (matrix <= upper + eps)
    )


def matrix_rank(matrix, tol=1e-10):
    """Численный ранг матрицы."""
    return np.linalg.matrix_rank(matrix, tol=tol)


def is_singular(matrix, tol=1e-10):
    """Проверка вырожденности квадратной матрицы."""
    return abs(np.linalg.det(matrix)) <= tol


def print_matrix(matrix, name):
    print(f"\n{name}:")
    print(np.array2string(
        matrix,
        precision=9,
        suppress_small=False
    ))


# ============================================================
# A1. Линейная регрессия
# ============================================================

def analyze_a1_regression():
    """
    A1 из задания:

        [0.95  1]
        [1.05  1]
        [1.10  1]

    rad(A1) = delta *
        [1 0]
        [1 0]
        [1 0]

    Неопределённость присутствует только в первом столбце.
    Второй столбец фиксирован: [1, 1, 1]^T.
    """

    print("=" * 70)
    print("A1 — ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 70)

    center = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    direction = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])

    print_matrix(center, "mid(A1)")
    print_matrix(direction, "Матрица радиусов")

    # --------------------------------------------------------
    # Теоретическое условие
    # --------------------------------------------------------
    #
    # Ранг матрицы 3×2 падает с 2 до 1, если столбцы становятся
    # линейно зависимыми: первый столбец = z × второй.
    #
    # Второй столбец фиксирован: [1, 1, 1]^T.
    # Значит первый столбец должен стать константой [z, z, z]^T.
    #
    # Каждый элемент первого столбца может принимать значение
    # из интервала [m_i - delta, m_i + delta], где
    # m_i ∈ {0.95, 1.05, 1.10}.
    #
    # Нужно найти такое z, которое принадлежит всем трём
    # интервалам одновременно — то есть их пересечение непусто.
    #
    # Пересечение существует, если левая граница самого правого
    # интервала не превышает правую границу самого левого:
    #
    #   min(m_i) + delta >= max(m_i) - delta
    #
    #   2 * delta >= max(m_i) - min(m_i)
    #
    #   delta* = (max - min) / 2
    #
    # При delta = delta* пересечение состоит из одной точки:
    #
    #   z = (min(m_i) + max(m_i)) / 2
    # --------------------------------------------------------

    m_min = np.min(center[:, 0])  # 0.95
    m_max = np.max(center[:, 0])  # 1.10

    minimum_delta = (m_max - m_min) / 2.0
    # = (1.10 - 0.95) / 2 = 0.075

    # Значение z, при котором первый столбец становится константой.
    # Это середина отрезка [m_min, m_max] — единственная точка
    # пересечения всех интервалов при delta = delta*.
    z = (m_min + m_max) / 2.0
    # = (0.95 + 1.10) / 2 = 1.025

    print("\nПоиск минимального delta:")
    print(f"m_min (первый столбец) = {m_min:.6f}")
    print(f"m_max (первый столбец) = {m_max:.6f}")
    print(f"Разброс = {m_max - m_min:.6f}")
    print(f"Минимальное delta = {minimum_delta:.6f}")
    print(f"Значение z = {z:.6f}")

    # Построение вырожденной матрицы:
    # первый столбец — константа z, второй — единицы.
    a1_prime = np.column_stack([
        np.full(3, z),
        np.ones(3)
    ])

    radius = minimum_delta * direction

    print_matrix(a1_prime, "Построенная матрица A1'")

    # Проверки
    belongs = interval_contains(a1_prime, center, radius)
    rank = matrix_rank(a1_prime)

    print("\nПроверка результата:")
    print(f"A1' принадлежит A1: {belongs}")
    print(f"rank(A1') = {rank}")
    print(f"Требуется rank(A1') < 2: {rank < 2}")

    return minimum_delta, a1_prime


# ============================================================
# A1. Малоракурсная томография
# ============================================================

def analyze_a1_tomography():
    """
    A1:

        [0.95  1]
        [1.05  1]
        [1.10  1]

    rad(A1) = delta *
        [1 1]
        [1 1]
        [1 1]

    Неопределённы оба столбца.
    Второй столбец теперь также «плавает» в интервале
    [1 - delta, 1 + delta].
    """

    print("\n" + "=" * 70)
    print("A1 — МАЛОРАКУРСНАЯ ТОМОГРАФИЯ")
    print("=" * 70)

    center = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    direction = np.ones((3, 2))

    print_matrix(center, "mid(A1)")
    print_matrix(direction, "Матрица радиусов")

    # --------------------------------------------------------
    # Условие вырожденности: столбцы линейно зависимы.
    #
    #   a_i = z * b_i  для каждой строки i
    #
    # где:
    #   a_i ∈ [m_i - delta, m_i + delta]  — первый столбец
    #   b_i ∈ [1 - delta, 1 + delta]      — второй столбец
    #
    # Отсюда z должен удовлетворять:
    #
    #   (m_i - delta) / (1 + delta) <= z <= (m_i + delta) / (1 - delta)
    #
    # Самая узкая нижняя граница z задаётся строкой с
    # максимальным m_i (т.к. числитель больше → дробь больше):
    #
    #   z >= (m_max - delta) / (1 + delta)
    #
    # Самая узкая верхняя граница z задаётся строкой с
    # минимальным m_i:
    #
    #   z <= (m_min + delta) / (1 - delta)
    #
    # При критическом delta эти границы совпадают:
    #
    #   (m_max - delta) / (1 + delta) = (m_min + delta) / (1 - delta)
    #
    # Раскрываем крест-накрест:
    #
    #   (m_max - delta)(1 - delta) = (m_min + delta)(1 + delta)
    #
    #   m_max - m_max*delta - delta + delta^2 = m_min + m_min*delta + delta + delta^2
    #
    # delta^2 сокращается:
    #
    #   m_max - m_min = (m_max + 1 + m_min + 1) * delta
    #
    # Здесь коэффициент при delta — это:
    #   m_max + m_min + 2
    #
    # Двойка возникает потому, что центр второго столбца c = 1,
    # и он входит в формулу дважды (по одному разу для каждой
    # границы z). В общем случае при произвольном центре c:
    #
    #   delta* = (m_max - m_min) / (m_max + m_min + 2*c)
    #
    # При c = 1:
    #   delta* = (1.10 - 0.95) / (1.10 + 0.95 + 2*1)
    #          = 0.15 / 4.05 = 1/27 ≈ 0.037037...
    # --------------------------------------------------------

    m_min = np.min(center[:, 0])  # 0.95
    m_max = np.max(center[:, 0])  # 1.10
    c = center[0, 1]              # 1.00 — центр второго столбца

    spread = m_max - m_min  # 0.15

    # 2.0 = 2 * c — удвоенный центр второго столбца.
    # В формуле он появляется дважды: один раз из левой
    # границы z (знаменатель 1 + delta → коэффициент +1 при delta),
    # второй раз из правой границы (знаменатель 1 - delta →
    # ещё +1). Итого +2 при c = 1.
    minimum_delta = spread / (m_max + m_min + 2.0 * c)
    # = 0.15 / (1.10 + 0.95 + 2.0) = 0.15 / 4.05 = 1/27

    # z вычисляется из условия совпадения границ.
    # Используем верхнюю границу (можно и нижнюю — совпадают):
    z = (m_min + minimum_delta) / (c - minimum_delta)
    # = (0.95 + 1/27) / (1 - 1/27) ≈ 1.01887

    print("\nПоиск минимального delta:")
    print(f"m_min = {m_min:.6f}")
    print(f"m_max = {m_max:.6f}")
    print(f"c (центр 2-го столбца) = {c:.6f}")
    print(f"Разброс = {spread:.6f}")
    print(f"Минимальное delta = {minimum_delta:.12f}")
    print(f"Точно: delta = 1/27 = {1.0/27:.12f}")
    print(f"Коэффициент пропорциональности z = {z:.12f}")

    # --------------------------------------------------------
    # Построение A1'
    #
    # Нужно выбрать конкретные значения a_i и b_i внутри
    # интервалов так, чтобы a_i = z * b_i.
    #
    # На границе вырождения (delta = delta*) есть ровно одно
    # решение. Выбираем b_i на краях интервалов:
    #
    #   Строка 0 (m = 0.95, минимальный): b = c - delta
    #     → a = z * b должно попасть в [0.95 - delta, 0.95 + delta]
    #
    #   Строка 2 (m = 1.10, максимальный): b = c + delta
    #     → a = z * b должно попасть в [1.10 - delta, 1.10 + delta]
    #
    #   Среднюю строку (m = 1.05) можно взять b = c (центр),
    #     тогда a = z * c должно попасть в [1.05 - delta, 1.05 + delta]
    # --------------------------------------------------------

    b = np.array([
        c - minimum_delta,  # нижняя граница для строки 0
        c,                  # центр для строки 1
        c + minimum_delta   # верхняя граница для строки 2
    ])

    a = z * b

    a1_prime = np.column_stack([a, b])
    radius = minimum_delta * direction

    print_matrix(a1_prime, "Построенная матрица A1'")

    # Проверки
    belongs = interval_contains(a1_prime, center, radius)
    rank = matrix_rank(a1_prime)

    print("\nПроверка результата:")
    print(f"A1' принадлежит A1: {belongs}")
    print(f"rank(A1') = {rank}")
    print(f"Требуется rank(A1') < 2: {rank < 2}")

    print("\nПроверка линейной зависимости:")
    print(f"max|A1'[:,0] - z*A1'[:,1]| = "
          f"{np.max(np.abs(a - z * b)):.3e}")

    return minimum_delta, a1_prime


# ============================================================
# A2. Квадратная матрица
# ============================================================

def find_singular_matrix_a2(center):
    """
    Поиск сингулярной матрицы внутри интервальной матрицы A2.

    Используется теорема о радиусе невырожденности:

    Для матрицы A с радиусом неопределённости delta во всех
    элементах, минимальное delta, при котором в интервальной
    матрице появляется сингулярная точечная матрица, равно:

        delta* = 1 / max_{y, z} |z^T A^{-1} y|

    где y, z — всевозможные знаковые векторы из {-1, +1}^n.

    Идея: возмущение E = delta * y * z^T (матрица ранга 1)
    задаёт «наихудшее» направление. При delta = delta*
    определитель det(A + E) обращается в нуль.

    Формула определителя при возмущении ранга 1:

        det(A + delta * y * z^T) = det(A) * (1 + delta * z^T A^{-1} y)

    det = 0, когда:
        1 + delta * z^T A^{-1} y = 0
        delta = -1 / (z^T A^{-1} y)

    Минимальное положительное delta достигается при
    максимальном |z^T A^{-1} y|.
    """

    inverse = np.linalg.inv(center)
    n = center.shape[0]

    best_value = -np.inf
    best_raw = 0.0       # z^T A^{-1} y с сохранением знака
    best_y = None
    best_z = None

    # Перебираем все 2^n × 2^n комбинации знаковых векторов.
    # Для n = 3 это 8 × 8 = 64 варианта — мгновенно.
    for y in itertools.product([-1.0, 1.0], repeat=n):
        y = np.array(y)

        for z in itertools.product([-1.0, 1.0], repeat=n):
            z = np.array(z)

            raw_value = z @ inverse @ y  # с сохранением знака!
            value = abs(raw_value)

            if value > best_value:
                best_value = value
                best_raw = raw_value
                best_y = y.copy()
                best_z = z.copy()

    # delta* = 1 / max|z^T A^{-1} y|
    minimum_delta = 1.0 / best_value

    # --------------------------------------------------------
    # Построение сингулярного возмущения.
    #
    # det(A + E) = det(A) * (1 + delta * z^T A^{-1} y)
    #
    # Нужно, чтобы 1 + delta * z^T A^{-1} y = 0, то есть
    # delta * z^T A^{-1} y = -1.
    #
    # Если z^T A^{-1} y < 0 (как в нашем случае = -10),
    # то возмущение E = +delta * y * z^T даёт:
    #   delta * z^T A^{-1} y = delta * (-10) = -1  ✓
    #
    # Если z^T A^{-1} y > 0, нужен минус:
    #   E = -delta * y * z^T
    #
    # Поэтому знак перед возмущением должен совпадать
    # со знаком raw_value, чтобы произведение
    # delta * sign * (z^T A^{-1} y) было равно -1.
    # --------------------------------------------------------

    sign = np.sign(best_raw)  # +1 или -1

    # Возмущение: E = -delta * sign(yz) * y * z^T
    # Знак минус нужен, потому что delta* = 1/|raw|,
    # а нам нужно delta * raw * sign_coeff = -1.
    # При raw < 0: sign = -1, E = +delta * y * z^T  → delta * raw = -1 ✓
    # При raw > 0: sign = +1, E = -delta * y * z^T  → delta * raw = -1 ✓
    perturbation = -minimum_delta * sign * np.outer(best_y, best_z)

    singular_matrix = center + perturbation

    return (
        minimum_delta,
        singular_matrix,
        perturbation,
        best_value,
        best_raw,
        best_y,
        best_z
    )


def analyze_a2():
    """
    A2 из задания:

        [1.10  0.90  1.10]
        [1.40  1.00  0.80]
        [0.80  1.40  1.20]

    rad(A2) = delta * ones(3,3)
    """

    print("\n" + "=" * 70)
    print("A2 — КВАДРАТНАЯ МАТРИЦА")
    print("=" * 70)

    center = np.array([
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20]
    ])

    direction = np.ones((3, 3))

    print_matrix(center, "mid(A2)")
    print_matrix(direction, "Матрица радиусов")

    # Проверяем, что исходная матрица невырождена.
    det_center = np.linalg.det(center)
    rank_center = matrix_rank(center)

    print("\nИсходная матрица:")
    print(f"det(mid(A2)) = {det_center:.12f}")
    print(f"rank(mid(A2)) = {rank_center}")

    # Поиск минимального delta и построение сингулярной матрицы.
    (
        minimum_delta,
        a2_prime,
        perturbation,
        best_value,
        best_raw,
        y,
        z
    ) = find_singular_matrix_a2(center)

    print("\nПоиск минимального delta:")
    print(f"max |z^T A^(-1) y| = {best_value:.12f}")
    print(f"z^T A^(-1) y (с знаком) = {best_raw:.12f}")
    print(f"Минимальное delta = {minimum_delta:.12f}")

    print("\nЗнаковые векторы, определяющие критическое "
          "возмущение:")
    print(f"y = {y.astype(int)}")
    print(f"z = {z.astype(int)}")

    print_matrix(perturbation, "Критическое возмущение E")
    print_matrix(a2_prime, "Построенная матрица A2'")

    # Проверки
    radius = minimum_delta * direction

    belongs = interval_contains(a2_prime, center, radius)
    determinant = np.linalg.det(a2_prime)
    rank = matrix_rank(a2_prime)

    print("\nПроверка результата:")
    print(f"A2' принадлежит A2: {belongs}")
    print(f"det(A2') = {determinant:.12e}")
    print(f"rank(A2') = {rank}")
    print(f"A2' вырожденная: {is_singular(a2_prime)}")

    return minimum_delta, a2_prime


# ============================================================
# Исследование зависимости от delta
# ============================================================

def investigate_delta_a1_regression():
    """
    Показывает, что для delta < delta* вырожденной матрицы
    нет, а при delta >= delta* она появляется.

    В случае регрессии неопределённость только в первом
    столбце. Вырожденность наступает, когда все интервалы
    первого столбца пересекаются (можно выбрать общее z).
    """

    center = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    # delta* = (max - min) / 2 = 0.075
    delta_star = (
        np.max(center[:, 0]) -
        np.min(center[:, 0])
    ) / 2.0

    print("\n" + "-" * 70)
    print("ИССЛЕДОВАНИЕ A1 — РЕГРЕССИЯ")
    print("-" * 70)

    for delta in [
        0.0,
        delta_star / 2,
        delta_star - 0.001,
        delta_star,
        delta_star + 0.001,
        delta_star * 2
    ]:
        # Пересение интервалов первого столбца:
        #   left  = max(m_i - delta)  — самая правая левая граница
        #   right = min(m_i + delta)  — самая левая правая граница
        intersection_left = np.max(center[:, 0] - delta)
        intersection_right = np.min(center[:, 0] + delta)

        # Пересечение непусто, если left <= right
        exists = intersection_left <= intersection_right + EPS

        print(
            f"delta = {delta:.6f} | "
            f"пересечение: "
            f"[{intersection_left:.6f}, "
            f"{intersection_right:.6f}] | "
            f"вырожденная матрица существует: {exists}"
        )

def investigate_delta_a2():
    center = np.array([
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20]
    ])

    direction = np.ones((3, 3))

    delta_star, _, _, _, _, _, _ = find_singular_matrix_a2(center)

    print("\n" + "-" * 70)
    print("ИССЛЕДОВАНИЕ A2")
    print("-" * 70)

    print(f"Критическое значение delta* = {delta_star:.12f}")

    # Значения delta для численного исследования
    delta_values = [
        0,
        delta_star / 2,
        delta_star - 0.001,
        delta_star,
        delta_star + 0.001,
        2 * delta_star
    ]

    print("\nЧисленное исследование различных значений delta:")

    for delta in delta_values:

        radius = delta * direction

        lower = center - radius
        upper = center + radius

        print("\n" + "-" * 70)
        print(f"delta = {delta:.6f}")

        print("Интервальная матрица A2:")

        for i in range(3):
            print(
                f"[{lower[i, 0]:.4f}; {upper[i, 0]:.4f}]  "
                f"[{lower[i, 1]:.4f}; {upper[i, 1]:.4f}]  "
                f"[{lower[i, 2]:.4f}; {upper[i, 2]:.4f}]"
            )

        # Для квадратной матрицы проверяем,
        # содержит ли интервал сингулярную точечную матрицу.
        #
        # При delta < delta_star сингулярной матрицы нет.
        # При delta >= delta_star она существует.

        if delta < delta_star - 1e-12:
            print("Сингулярная точечная матрица: НЕ существует")
        else:
            print("Сингулярная точечная матрица: существует")

    print("\n" + "-" * 70)
    print("ТЕОРЕТИЧЕСКИЙ ВЫВОД")
    print("-" * 70)

    print(
        "При delta < delta* интервальная матрица "
        "не содержит сингулярных точечных матриц."
    )

    print(
        "При delta = delta* появляется первая "
        "сингулярная точечная матрица."
    )

    print(
        "При delta > delta* интервальная матрица "
        "также содержит сингулярные точечные матрицы."
    )
# ============================================================
# Сводный вывод
# ============================================================

def print_conclusion(
    delta_regression,
    delta_tomography,
    delta_square
):
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    print(
        f"\nA1 — линейная регрессия:"
        f"\n  delta* = {delta_regression:.12f}"
        f"\n  delta* = 0.075"
    )

    print(
        f"\nA1 — малоракурсная томография:"
        f"\n  delta* = {delta_tomography:.12f}"
        f"\n  delta* = 1/27 ≈ {1.0/27:.12f}"
    )

    print(
        f"\nA2 — квадратная система:"
        f"\n  delta* = {delta_square:.12f}"
    )

    print("\nИнтерпретация:")

    print(
        "\n1. В случае линейной регрессии неопределённость "
        "есть только в первом столбце. Для появления "
        "линейной зависимости столбцов первый столбец "
        "должен стать постоянным. Это требует delta* = 0.075 "
        "(половина разброса центров первого столбца)."
    )

    print(
        "\n2. В случае малоракурсной томографии неопределённы "
        "оба столбца A1. Поэтому можно одновременно "
        "изменять оба столбца и получить линейную "
        "зависимость уже при меньшем delta* = 1/27 ≈ 0.037. "
        "Двойка в знаменателе формулы (m_max + m_min + 2c) "
        "возникает потому, что центр второго столбца c = 1 "
        "входит в выражение дважды — по одному разу из "
        "каждой границы пропорции z."
    )

    print(
        "\n3. Для квадратной матрицы A2 исходная матрица "
        "невырождена, однако при критическом радиусе "
        "delta в интервальной матрице появляется "
        "сингулярная точечная матрица. Поиск ведётся "
        "перебором знаковых векторов y, z ∈ {-1,+1}^3 "
        "с максимизацией |z^T A^{-1} y|."
    )


# ============================================================
# Запуск исследования
# ============================================================

if __name__ == "__main__":

    # A1 — линейная регрессия
    delta_regression, _ = analyze_a1_regression()

    # A1 — малоракурсная томография
    delta_tomography, _ = analyze_a1_tomography()

    # A2 — квадратная матрица
    delta_square, _ = analyze_a2()

    # Дополнительное исследование
    investigate_delta_a1_regression()
    investigate_delta_a2()

    # Итог
    print_conclusion(
        delta_regression,
        delta_tomography,
        delta_square
    )
