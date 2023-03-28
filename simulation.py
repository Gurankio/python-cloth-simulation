import importlib
import pathlib
import pickle

import appdirs
import numpy
import sympy
from manimlib import *
from more_itertools import flatten
from tqdm.auto import tqdm

t = sympy.symbols('t')

# caching
CACHE = pathlib.Path(appdirs.user_cache_dir('python-lagrangian-simulator'))
CACHE.mkdir(parents=True, exist_ok=True)


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
        'range_kutta',
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
        data = pickle.dumps((self.FIXED_POINTS,
                             self.POINTS, self.MASS, self.GRAVITY,
                             self.SPRINGS, self.SPRINGINESS, self.REST))
        data_hash = hashlib.blake2b(data, person="simulator-v1".encode('utf-8'))
        cache_file = CACHE / f'generated_{data_hash.hexdigest()}.py'

        if not cache_file.exists():
            print(f'Generating lagrangian ({data_hash.hexdigest()})')

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
                sympy.solve([sympy.diff(sympy.diff(lagrangian, p.vx), t) - sympy.diff(lagrangian, p.x),
                             sympy.diff(sympy.diff(lagrangian, p.vy), t) - sympy.diff(lagrangian, p.y)],
                            [p.ax, p.ay])
                for p in tqdm(self.points, desc='Solving Euler-Lagrange equations')
            ]

            q = list(flatten([p.x, p.y] for p in self.points))
            qd = list(flatten([p.vx, p.vy] for p in self.points))

            sources = list(flatten([
                (inspect.getsource(sympy.lambdify(q + qd, e[p.ax], 'numpy')),
                 inspect.getsource(sympy.lambdify(q + qd, e[p.ay], 'numpy')))
                for e, p in tqdm(zip(eq_lagrangian, self.points),
                                 desc='Generating source', total=len(self.points))
            ]))
            hamiltonian_source = inspect.getsource(sympy.lambdify(q + qd, hamiltonian, 'numpy'))

            parameters_map = [
                *flatten([f'q[{i * 2}]', f'q[{i * 2 + 1}]']
                         for i in range(len(self.points))),
                *flatten([f'qd[{len(self.points) + i * 2}]', f'qd[{len(self.points) + i * 2 + 1}]']
                         for i in range(len(self.points))),
            ]

            definition = re.compile(r'def .+\((.+)\):\s+return (.+)')

            def vectorize(source):
                parameters, expr = definition.match(source).groups()
                parameters = parameters.split(', ')
                for j, parameter in enumerate(parameters):
                    expr = expr.replace(parameter, parameters_map[j])
                return f'        {expr},\n'

            vectorized = ''.join([*tqdm((vectorize(source) for source in sources),
                                        desc='Refactoring', total=len(sources))
                                  ])
            hamiltonian_vectorized = vectorize(hamiltonian_source)

            source = f'from numpy import *\n' \
                     f'from numba import njit\n' \
                     f'\n' \
                     f'@njit(fastmath=True, cache=True)\n' \
                     f'def hamiltonian(q, qd):\n' \
                     f'    return {hamiltonian_vectorized.removesuffix(",")}' \
                     f'\n' \
                     '@njit(fastmath=True, cache=True)\n' \
                     'def range_kutta(q, qd, delta):\n' \
                     '    k1 = qd\n' \
                     '    k1d = f_lagrangian(q, qd)\n' \
                     '    k2 = qd + 0.5 * k1d * delta\n' \
                     '    k2d = f_lagrangian(q + 0.5 * k1 * delta, qd + 0.5 * k1d * delta)\n' \
                     '    k3 = qd + 0.5 * k2d * delta\n' \
                     '    k3d = f_lagrangian(q + 0.5 * k2 * delta, qd + 0.5 * k2d * delta)\n' \
                     '    k4 = qd + k3d * delta\n' \
                     '    k4d = f_lagrangian(q + k3 * delta, qd + k3d * delta)\n' \
                     '    q += (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6\n' \
                     '    qd += (k1d + 2 * k2d + 2 * k3d + k4d) * delta / 6\n' \
                     '    return q, qd\n' \
                     f'\n' \
                     f'@njit(fastmath=True, cache=True)\n' \
                     f'def f_lagrangian(q, qd):\n' \
                     f'    return array([\n' \
                     f'{vectorized}' \
                     f'    ])' \
                     f'\n'

            try:
                cache_file.touch(exist_ok=False)
                cache_file.write_text(source)
            except:
                print('Failed to save lagrangian')
                exit(1)

        print(f'Loading lagrangian, this might take a while... ({data_hash.hexdigest()})')
        now = time.time()
        spec = importlib.util.spec_from_file_location(cache_file.name, str(cache_file))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[cache_file.name] = mod
        spec.loader.exec_module(mod)
        self.range_kutta = mod.range_kutta
        self.hamiltonian = mod.hamiltonian
        elapsed = time.time() - now
        print(f'Lagrangian loaded successfully in {elapsed:.2f}s')

    def construct(self):
        self.AXES.fix_in_frame()
        self.add(self.AXES)

        q = numpy.array(list(flatten([p[0] for p in self.POINTS])), dtype=numpy.float64)
        qd = numpy.array(list(flatten([p[1] for p in self.POINTS])), dtype=numpy.float64)

        hamilton_dot = Dot(self.AXES.c2p(0, 0), color=GREEN)
        self.add(hamilton_dot)

        fixed_dots = [Dot(self.AXES.c2p(x, y), color=BLUE) for x, y in self.FIXED_POINTS]
        self.play(*(FadeIn(dot, scale=0.5) for dot in fixed_dots), run_time=self.FADE)

        dots = [Dot(color=RED) for _ in self.POINTS]
        for i, dot in enumerate(dots):
            dot.move_to(self.AXES.c2p(q[i * 2], q[i * 2 + 1]))
        self.play(*(FadeIn(dot, scale=0.5) for dot in dots), run_time=self.FADE)

        lines = [Line(color=RED_C) for _ in self.SPRINGS]
        for line, (p1, p2) in zip(lines, self.SPRINGS):
            f_always(
                line.put_start_and_end_on,
                (dots[p1] if p1 >= 0 else fixed_dots[-p1 - 1]).get_center,
                (dots[p2] if p2 >= 0 else fixed_dots[-p2 - 1]).get_center
            )
            self.bring_to_back(line)
        self.play(*(FadeIn(line, scale=0.5) for line in lines), run_time=self.FADE)

        try:
            while True:
                q, qd = self.range_kutta(q, qd, self.DELTA)
                qd *= 1 - self.DAMPING
                h = self.hamiltonian(q, qd)

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


class Cloth(Simulation):
    MASS = 0.001
    SPRINGINESS = 0.2
    REST = 1

    FIXED_POINTS = [
        (2, 8),
        (8, 8),
    ]
    POINTS = [
        ((2 + i, 8 - j), (0, 0))
        for i in range(0, 7)
        for j in range(0, 6)
        if (i != 0 and i != 6) or j != 0
    ]
    SPRINGS = [
                  (-1, 0),
                  (-1, 5),
                  (-2, 29),
                  (-2, 35),
              ] + [
                  (i, i + 1)
                  for i in range(39)
                  if i % 6 != 4
              ] + [
                  (i, i + 5 + (i < 30))
                  for i in range(35)
                  if i != 29
              ]


class Benchmark(Simulation):
    MASS = 0.0005
    SPRINGINESS = 0.5
    REST = 0.2

    DELTA = 1 / 10

    FIXED_POINTS = [(1 + 8 / 10 * x, 10) for x in range(10)]
    POINTS = [((1 + 8 / 10 * x, 10 - 5 / 10 * y), (0, 0))
              for x in range(10)
              for y in range(9)]
    SPRINGS = [(-1 - x, x) for x in range(10)] + \
              [(10 * y + x, 10 * (y + 1) + x) for x in range(10) for y in range(8)] + \
              [(10 * y + x - 1, 10 * y + x) for x in range(1, 10) for y in range(8)] + \
              [(10 * y + x, 10 * y + x + 1) for x in range(0, 9) for y in range(8)]
