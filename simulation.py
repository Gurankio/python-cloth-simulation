import sympy
from manimlib import *
from more_itertools import flatten
from tqdm.auto import tqdm

t = sympy.symbols('t')


class FixedPoint:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def k(self):
        return 0

    def u(self):
        return 0


class Point:
    __slots__ = ("mass", "gravity", "x", "y", "vx", "vy", "ax", "ay")

    def __init__(self, n, mass, gravity):
        self.mass = mass
        self.gravity = gravity
        self.x = sympy.Function(f'x_{n}')(t)
        self.y = sympy.Function(f'y_{n}')(t)
        self.vx = sympy.diff(self.x, t)
        self.vy = sympy.diff(self.y, t)
        self.ax = sympy.diff(self.vx, t)
        self.ay = sympy.diff(self.vy, t)

    def k(self):
        return 0.5 * self.mass * (self.vx ** 2 + self.vy ** 2)

    def u(self):
        return self.mass * self.gravity * self.y


class Spring:
    __slots__ = ("springiness", "rest", "a", "b", "x", "y", "l")

    def __init__(self, a, b, springiness, rest):
        self.springiness = springiness
        self.rest = rest
        self.a = a
        self.b = b
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.l = sympy.sqrt(self.x ** 2 + self.y ** 2)

    def u(self):
        return 0.5 * self.springiness * (self.l - self.rest) ** 2


T = TypeVar('T')


class Simulation(Scene):
    # required
    SPRINGS = None
    POINTS = None
    FIXED_POINTS = None

    # optional
    MASS = 0.05
    GRAVITY = 9.81
    SPRINGINESS = 0.5
    DAMPING = 0.01
    REST = 0.25

    # time
    DELTA = 1 / 30

    # graphics
    FADE = 1

    AXES = Axes(
        x_range=(0, 10, 0.5),
        y_range=(0, 10, 0.5),
    )
    AXES.add_coordinate_labels(
        font_size=20,
        num_decimal_places=1
    )

    __slots__ = (
        'points',
        'fixed_points',
        'springs',
        'lagrangian',
        'hamiltonian',
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fixed_points = None
        self.points = None
        self.springs = None
        self.lagrangian = None
        self.hamiltonian = None

    @staticmethod
    def resolve_point(index: int, points: [T], fixed_points: [T]) -> T:
        return points[index] if index >= 0 else fixed_points[-index - 1]

    def setup(self):
        self.fixed_points = [FixedPoint(x, y) for x, y in self.FIXED_POINTS]
        self.points = [Point(i, self.MASS, self.GRAVITY) for i in range(len(self.POINTS))]
        self.springs = [Spring(self.resolve_point(a, self.points, self.fixed_points),
                               self.resolve_point(b, self.points, self.fixed_points),
                               self.SPRINGINESS, self.REST)
                        for a, b in self.SPRINGS]

        kinetic_energy = sum(p.k() for p in self.points)
        potential_energy = sum(p.u() for p in self.points) + sum(s.u() for s in self.springs)

        lagrangian = kinetic_energy - potential_energy
        hamiltonian = kinetic_energy + potential_energy

        eq_lagrangian = [
            sympy.solve(sympy.Matrix([sympy.diff(sympy.diff(lagrangian, p.vx), t) - sympy.diff(lagrangian, p.x),
                                      sympy.diff(sympy.diff(lagrangian, p.vy), t) - sympy.diff(lagrangian, p.y)]),
                        [p.ax, p.ay])
            for p in tqdm(self.points, desc='Solving Euler-Lagrange equations')
        ]

        q = list(flatten([p.x, p.y] for p in self.points))
        qd = list(flatten([p.vx, p.vy] for p in self.points))

        lambdas = list(flatten([
            (sympy.lambdify(q + qd, e[p.ax], 'numpy', cse=True),
             sympy.lambdify(q + qd, e[p.ay], 'numpy', cse=True))
            for e, p in tqdm(zip(eq_lagrangian, self.points), desc='Compiling Euler-Lagrange equations')
        ]))

        self.lagrangian = lambda q, qd: np.array([lam(*q, *qd) for lam in lambdas], dtype=np.float64)
        self.hamiltonian = sympy.lambdify(q + qd, hamiltonian, 'numpy')

    def range_kutta(self, q, qd):
        k1 = qd
        k1d = self.lagrangian(q, qd)
        k2 = qd + 0.5 * k1d * self.DELTA
        k2d = self.lagrangian(q + 0.5 * k1 * self.DELTA, qd + 0.5 * k1d * self.DELTA)
        k3 = qd + 0.5 * k2d * self.DELTA
        k3d = self.lagrangian(q + 0.5 * k2 * self.DELTA, qd + 0.5 * k2d * self.DELTA)
        k4 = qd + k3d * self.DELTA
        k4d = self.lagrangian(q + k3 * self.DELTA, qd + k3d * self.DELTA)
        q += (k1 + 2 * k2 + 2 * k3 + k4) * self.DELTA / 6
        qd += (k1d + 2 * k2d + 2 * k3d + k4d) * self.DELTA / 6
        return q, qd

    def construct(self):
        self.AXES.fix_in_frame()
        self.add(self.AXES)

        q = np.array(list(flatten([p[0] for p in self.POINTS])), dtype=np.float64)
        qd = np.array(list(flatten([p[1] for p in self.POINTS])), dtype=np.float64)

        hamilton_dot = Dot(self.AXES.c2p(0, 0), color=GREEN)

        fixed_dots = [Dot(self.AXES.c2p(x, y), color=BLUE) for x, y in self.FIXED_POINTS]
        for dot in fixed_dots:
            self.play(FadeIn(dot, scale=0.5), run_time=self.FADE / len(fixed_dots))

        dots = [Dot(color=RED) for _ in self.POINTS]
        for i, dot in enumerate(dots):
            dot.move_to(self.AXES.c2p(q[i * 2], q[i * 2 + 1]))
            self.play(FadeIn(dot, scale=0.5), run_time=self.FADE / len(dots))

        lines = [Line(color=RED_C) for _ in self.SPRINGS]
        for i, line in enumerate(lines):
            p1, p2 = self.SPRINGS[i]
            f_always(
                line.put_start_and_end_on,
                (dots[p1] if p1 >= 0 else fixed_dots[-p1 - 1]).get_center,
                (dots[p2] if p2 >= 0 else fixed_dots[-p2 - 1]).get_center
            )
            self.play(FadeIn(line, scale=0.5), run_time=self.FADE / len(lines))

        try:
            while True:
                q, qd = self.range_kutta(q, qd)
                qd *= 1 - self.DAMPING
                h = self.hamiltonian(*q, *qd)

                self.play(
                    hamilton_dot.animate.move_to(self.AXES.c2p(h, 0)),
                    *[dot.animate.move_to(self.AXES.c2p(q[i * 2], q[i * 2 + 1])) for i, dot in enumerate(dots)],
                    run_time=self.DELTA
                )
        except KeyboardInterrupt:
            pass


class Rope(Simulation):
    FIXED_POINTS = [
        (5, 9),
    ]
    POINTS = [
        ((5, 8), (1, 0)),
        ((5, 7), (2, 1)),
        ((5, 6), (4, 2)),
    ]
    SPRINGS = [
        (-1, 0),
        (0, 1),
        (1, 2),
    ]


class DoubleRope(Simulation):
    MASS = 0.0005
    SPRINGINESS = 0.5
    REST = 0.2

    FIXED_POINTS = [(2, 8), (8, 8)]
    POINTS = [((2 + 0.2 * i, 8), (0, 0)) for i in range(1, 30)]
    SPRINGS = [(-1, 0), (28, -2)] + [(i, i + 1) for i in range(28)]
