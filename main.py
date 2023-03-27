from more_itertools import flatten
from rich import print
import sympy
import numpy as np
from manimlib import *

MASS = 0.050
GRAVITY = 9.81
SPRINGINESS = 0.3
REST = 1
DELTA = 1 / 30

POINTS = [
    # x0 y0 vx0 vy0
    ((0, 0), (0, 0)),
    ((0, 1), (0, 0)),
    ((0, 2), (0, 0)),
]
SPRINGS = [
    # a b
    (0, 1),
    (1, 2),
]

t = sympy.symbols('t')


class Point:
    def __init__(self, n):
        self.x = sympy.Function(f'x_{n}')(t)
        self.y = sympy.Function(f'y_{n}')(t)
        self.vx = sympy.diff(self.x, t)
        self.vy = sympy.diff(self.y, t)
        self.ax = sympy.diff(self.vx, t)
        self.ay = sympy.diff(self.vy, t)

    def k(self):
        return 0.5 * MASS * (self.vx ** 2 + self.vy ** 2)

    def u(self):
        return 0
        # return MASS * GRAVITY * self.y


class Spring:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.l = sympy.sqrt(self.x ** 2 + self.y ** 2)

    def u(self):
        return 0.5 * SPRINGINESS * (self.l - REST) ** 2


points = [Point(i) for i in range(len(POINTS))]
springs = [Spring(points[a], points[b]) for a, b in SPRINGS]

kinetic_energy = sum(p.k() for p in points)
potential_energy = sum(p.u() for p in points) + sum(s.u() for s in springs)

lagrangian = kinetic_energy - potential_energy
hamiltonian = kinetic_energy + potential_energy

eq_lagrangian = [
    sympy.Matrix([sympy.diff(sympy.diff(lagrangian, p.vx), t) - sympy.diff(lagrangian, p.x),
                  sympy.diff(sympy.diff(lagrangian, p.vy), t) - sympy.diff(lagrangian, p.y)])
    for p in points
]

solutions = [
    sympy.solve(eq, [p.ax, p.ay])
    for eq, p in zip(eq_lagrangian, points)
]

q = list(flatten([p.x, p.y] for p in points))
qd = list(flatten([p.vx, p.vy] for p in points))

lambdas = [
    (sympy.lambdify(q + qd, s[p.ax], 'numpy'),
     sympy.lambdify(q + qd, s[p.ay], 'numpy'))
    for s, p in zip(solutions, points)
]


def a(q, qd):
    return np.array(list(flatten([(f(*q, *qd), g(*q, *qd)) for (f, g) in lambdas])), dtype=np.float64)


hamiltonian = sympy.lambdify(q + qd, hamiltonian, 'numpy')


def range_kutta(q, qd, a):
    k1 = qd
    k1d = a(q, qd)
    k2 = qd + 0.5 * k1d * DELTA
    k2d = a(q + 0.5 * k1 * DELTA, qd + 0.5 * k1d * DELTA)
    k3 = qd + 0.5 * k2d * DELTA
    k3d = a(q + 0.5 * k2 * DELTA, qd + 0.5 * k2d * DELTA)
    k4 = qd + k3d * DELTA
    k4d = a(q + k3 * DELTA, qd + k3d * DELTA)
    q += (k1 + 2 * k2 + 2 * k3 + k4) * DELTA / 6
    qd += (k1d + 2 * k2d + 2 * k3d + k4d) * DELTA / 6
    return q, qd


class Sim(Scene):
    def construct(self):
        axes = Axes(
            # x-axis ranges from -1 to 10, with a default step size of 1
            x_range=(-5, 5, 0.5),
            # y-axis ranges from -2 to 2 with a step size of 0.5
            y_range=(-5, 5, 0.5),
            # The axes will be stretched so as to match the specified
            # height and wiDELTAh
            height=10,
            wiDELTAh=10,
            # Axes is made of two NumberLine mobjects.  You can specify
            # their configuration with axis_config
            axis_config={
                "stroke_color": GREY_A,
                "stroke_wiDELTAh": 2,
            },
        )

        axes.add_coordinate_labels(font_size=20, num_decimal_places=1)
        self.add(axes)

        q = np.array(list(flatten([p[0] for p in POINTS])), dtype=np.float64)
        qd = np.array(list(flatten([p[1] for p in POINTS])), dtype=np.float64)
        print(q)
        print(qd)

        dots = [Dot(color=RED) for _ in POINTS]
        lines = [Line(color=RED_A) for _ in SPRINGS]

        for i, spring in enumerate(lines):
            f_always(spring.put_start_and_end_on, dots[SPRINGS[i][0]].get_center, dots[SPRINGS[i][1]].get_center)

        # one by one dots
        for i, dot in enumerate(dots):
            dot.move_to(axes.c2p(q[i * 2], q[i * 2 + 1]))
            self.play(FadeIn(dot, scale=0.5))

        # one by one lines
        for i, line in enumerate(lines):
            self.play(FadeIn(line, scale=0.5))

        try:
            while True:
                q, qd = range_kutta(q, qd, a)
                # h0 = hamiltonian(x0, y0, vx0, vy0)

                self.play(
                    *[dot.animate.move_to(axes.c2p(q[i * 2], q[i * 2 + 1])) for i, dot in enumerate(dots)],
                    run_time=DELTA
                )

                # self.play(
                #     dot.animate.move_to(axes.c2p(x0, y0)),
                #     hd.animate.move_to(axes.c2p(h0, 0)),
                #     run_time=DELTA
                # )
        except KeyboardInterrupt:
            pass
