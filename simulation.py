import abc
import math
import hashlib
import importlib
import inspect
import multiprocessing
import pathlib
import pickle
import re
import sys
import time
from typing import TypeVar

import appdirs
import numpy
import numpy as np
import pqdm.threads
import sympy
from more_itertools import flatten
from tqdm.auto import tqdm
from pqdm.processes import pqdm

import pygame

t = sympy.symbols('t')
T = TypeVar('T')

# caching
CACHE = pathlib.Path(appdirs.user_cache_dir('python-lagrangian-simulator'))
CACHE.mkdir(parents=True, exist_ok=True)

# simulation options
OPTIMIZED = False


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


def do_solve(eq):
    import sympy
    eq, t = sympy.sympify(eq)
    return repr(sympy.solve(eq, t))


class Simulation(abc.ABC):
    # required
    SPRINGS = None
    POINTS = None
    FIXED_POINTS = None

    # physics
    GRAVITY = 9.81
    DAMPING = 0.01

    # graphics
    STEPS = 10
    FPS = 60
    WIDTH = 1920 / 2
    HEIGHT = 1080 / 2

    @classmethod
    @abc.abstractmethod
    def fixed_points(cls) -> [(float | sympy.Function, float | sympy.Function)]:
        pass

    @classmethod
    @abc.abstractmethod
    def points(cls) -> [(float, float, float)]:
        pass

    @classmethod
    @abc.abstractmethod
    def springs(cls) -> [(float, float, float, float)]:
        pass

    def __init__(self):
        if self.FIXED_POINTS is None:
            self.FIXED_POINTS = self.fixed_points()

        if self.POINTS is None:
            self.POINTS = self.points()

        if self.SPRINGS is None:
            self.SPRINGS = self.springs()

        data = pickle.dumps((self.FIXED_POINTS, self.POINTS, self.SPRINGS))
        data_hash = hashlib.blake2b(data, person="simulator-v1".encode('utf-8'))
        cache_file = CACHE / f'generated_{data_hash.hexdigest()}.py'

        if not cache_file.exists():
            print(f'Generating lagrangian ({data_hash.hexdigest()})')
            source = self.generate()

            try:
                cache_file.touch(exist_ok=False)
                cache_file.write_text(source)
            except:
                print('Failed to save lagrangian, file exists?')
                exit(1)

        print(f'Loading lagrangian, this might take a while... ({data_hash.hexdigest()})')
        now = time.time()
        spec = importlib.util.spec_from_file_location(cache_file.name, str(cache_file))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[cache_file.name] = mod
        spec.loader.exec_module(mod)
        if OPTIMIZED:
            length = 2 * len(self.POINTS)
            self.range_kutta = mod.fast_range_kutta
            self.hamiltonian = mod.fast_hamiltonian

            try:
                self.range_kutta(np.ones(length, dtype=np.float64), np.ones(length, dtype=np.float64), 1.0)
            except:
                pass

            try:
                self.hamiltonian(np.ones(length, dtype=np.float64), np.ones(length, dtype=np.float64))
            except:
                pass
        else:
            self.range_kutta = mod.range_kutta
            self.hamiltonian = mod.hamiltonian
        print(f'Loaded lagrangian in {time.time() - now:.2f}s')

    def generate(self):
        def resolve_point(index: int, points: [T], fixed_points: [T]) -> T:
            return points[index] if index >= 0 else fixed_points[-index - 1]

        self.fixed_points = [FixedPoint(x, y) for x, y in self.FIXED_POINTS]
        self.points = [Point(i, mass, self.GRAVITY) for i, (_, _, mass) in enumerate(self.POINTS)]
        self.springs = [Spring(resolve_point(a, self.points, self.fixed_points),
                               resolve_point(b, self.points, self.fixed_points),
                               springiness, rest)
                        for a, b, springiness, rest in self.SPRINGS]
        kinetic_energy = sum(p.k() for p in self.points)
        potential_energy = sum(p.u() for p in self.points) + sum(s.u() for s in self.springs)
        lagrangian = kinetic_energy - potential_energy
        hamiltonian = kinetic_energy + potential_energy

        eq_lagrangian_data = [
            repr(([sympy.diff(sympy.diff(lagrangian, p.vx), t) - sympy.diff(lagrangian, p.x),
                   sympy.diff(sympy.diff(lagrangian, p.vy), t) - sympy.diff(lagrangian, p.y)],
                  [p.ax, p.ay]))
            for p in tqdm(self.points, desc='Preparing Euler-Lagrange equations')
        ]

        eq_lagrangian = pqdm(eq_lagrangian_data, do_solve, multiprocessing.cpu_count(),
                             desc='Solving Euler-Lagrange equations', total=len(self.points))
        eq_lagrangian = [*tqdm((sympy.sympify(sol) for sol in eq_lagrangian),
                               desc='Parsing Euler-Lagrange equations', total=len(self.points))]

        q = list(flatten([p.x, p.y] for p in self.points))
        qd = list(flatten([p.vx, p.vy] for p in self.points))

        sources = list(flatten([
            (inspect.getsource(sympy.lambdify([t] + q + qd, e[p.ax], 'numpy')),
             inspect.getsource(sympy.lambdify([t] + q + qd, e[p.ay], 'numpy')))
            for e, p in tqdm(zip(eq_lagrangian, self.points),
                             desc='Generating source', total=len(self.points))
        ]))
        hamiltonian_source = inspect.getsource(sympy.lambdify([t] + q + qd, hamiltonian, 'numpy'))
        parameters_map = [
            *flatten([f'q[{i * 2}]', f'q[{i * 2 + 1}]']
                     for i in range(len(self.points))),
            *flatten([f'qd[{i * 2}]', f'qd[{i * 2 + 1}]']
                     for i in range(len(self.points))),
        ]
        definition = re.compile(r'def .+\((.+)\):\s+return (.+)')

        def vectorize(source):
            parameters, expr = definition.match(source).groups()
            t, *parameters = parameters.split(', ')
            expr = expr.replace(t, 't')
            for j, parameter in enumerate(parameters):
                expr = expr.replace(parameter, parameters_map[j])
            return f'        {expr},\n'

        vectorized = ''.join([vectorize(source) for source in sources])
        hamiltonian_vectorized = vectorize(hamiltonian_source)
        return f'from numpy import *\n' \
               f'from numba import njit\n' \
               f'\n' \
               f'@njit(fastmath=True, parallel=False, cache=True)\n' \
               f'def fast_hamiltonian(t, q, qd):\n' \
               f'    return {hamiltonian_vectorized.strip().removesuffix(",")}' \
               f'\n' \
               '@njit(fastmath=True, parallel=False, cache=True)\n' \
               'def fast_range_kutta(t, q, qd, delta):\n' \
               '    k1 = qd\n' \
               '    k1d = _fast_lagrangian(t, q, qd)\n' \
               '    k2 = qd + 0.5 * k1d * delta\n' \
               '    k2d = _fast_lagrangian(t, q + 0.5 * k1 * delta, qd + 0.5 * k1d * delta)\n' \
               '    k3 = qd + 0.5 * k2d * delta\n' \
               '    k3d = _fast_lagrangian(t, q + 0.5 * k2 * delta, qd + 0.5 * k2d * delta)\n' \
               '    k4 = qd + k3d * delta\n' \
               '    k4d = _fast_lagrangian(t, q + k3 * delta, qd + k3d * delta)\n' \
               '    q += (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6\n' \
               '    qd += (k1d + 2 * k2d + 2 * k3d + k4d) * delta / 6\n' \
               '    return q, qd\n' \
               f'\n' \
               f'@njit(fastmath=True, parallel=False, cache=True)\n' \
               f'def _fast_lagrangian(t, q, qd):\n' \
               f'    return array([\n' \
               f'{vectorized}' \
               f'    ])\n' \
               f'\n' \
               f'def hamiltonian(t, q, qd):\n' \
               f'    return {hamiltonian_vectorized.strip().removesuffix(",")}' \
               f'\n' \
               'def range_kutta(t, q, qd, delta):\n' \
               '    k1 = qd\n' \
               '    k1d = _lagrangian(t, q, qd)\n' \
               '    k2 = qd + 0.5 * k1d * delta\n' \
               '    k2d = _lagrangian(t, q + 0.5 * k1 * delta, qd + 0.5 * k1d * delta)\n' \
               '    k3 = qd + 0.5 * k2d * delta\n' \
               '    k3d = _lagrangian(t, q + 0.5 * k2 * delta, qd + 0.5 * k2d * delta)\n' \
               '    k4 = qd + k3d * delta\n' \
               '    k4d = _lagrangian(t, q + k3 * delta, qd + k3d * delta)\n' \
               '    q += (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6\n' \
               '    qd += (k1d + 2 * k2d + 2 * k3d + k4d) * delta / 6\n' \
               '    return q, qd\n' \
               f'\n' \
               f'def _lagrangian(t, q, qd):\n' \
               f'    return array([\n' \
               f'{vectorized}' \
               f'    ])\n' \
               f'\n'

    def run(self):
        pygame.init()
        # screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.WIDTH = screen.get_width()
        self.HEIGHT = screen.get_height()
        clock = pygame.time.Clock()
        running = True

        q = numpy.array(list(flatten([p[0] for p in self.POINTS])), dtype=numpy.float64)
        qd = numpy.array(list(flatten([p[1] for p in self.POINTS])), dtype=numpy.float64)
        dt = 1 / self.FPS
        tv = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for _ in range(self.STEPS):
                q, qd = self.range_kutta(tv, q, qd, 1 / self.FPS / self.STEPS)
                tv += 1 / self.FPS / self.STEPS
            qd *= 1 - self.DAMPING
            h = self.hamiltonian(tv, q, qd)

            # fill the screen with a color to wipe away anything from last frame
            screen.fill(pygame.color.Color(27, 27, 27))

            from pygame.draw import line, circle

            line(screen, pygame.color.Color(255, 255, 255), self.transform(0, 0), self.transform(10, 0), 2)
            circle(screen, pygame.color.Color(93, 194, 57), self.transform((int(h > 0) - int(h < 0)) * math.log(abs(h)), 0), 5)

            def resolve_fixed(tv, x, y):
                return float((0 * t + x).evalf(subs={t: tv})), float((0 * t + y).evalf(subs={t: tv}))

            for i, (a, b, _, _) in enumerate(self.SPRINGS):
                p1 = (q[a * 2], q[a * 2 + 1]) if a >= 0 else resolve_fixed(tv, *self.FIXED_POINTS[-a - 1])
                p2 = (q[b * 2], q[b * 2 + 1]) if b >= 0 else resolve_fixed(tv, *self.FIXED_POINTS[-b - 1])
                line(screen, pygame.color.Color(225, 82, 82), self.transform(*p1), self.transform(*p2), 2)

            for pos in self.FIXED_POINTS:
                circle(screen, pygame.color.Color(236, 140, 40), self.transform(*resolve_fixed(tv, *pos)), 5)

            for i in range(len(self.POINTS)):
                circle(screen, pygame.color.Color(225, 82, 82), self.transform(q[i * 2], q[i * 2 + 1]), 5)

            pygame.display.flip()
            dt = clock.tick(self.FPS) / 1000

    def transform(self, x, y):
        scale = min(self.WIDTH, self.HEIGHT) / 12
        return scale * x + self.WIDTH / 2 - 5 * scale, -scale * y + self.HEIGHT / 2 + 5 * scale


class Rope(Simulation):
    MASS = 0.025
    SPRINGINESS = 0.25
    REST = 0.5

    @classmethod
    def fixed_points(cls):
        return [
            (5 + sympy.sin(t * 2), 9)
        ]

    @classmethod
    def points(cls):
        return [
            ((5, 8), (1, 0), cls.MASS),
            ((5, 7), (2, 1), cls.MASS),
            ((5, 6), (4, 2), cls.MASS),
        ]

    @classmethod
    def springs(cls):
        return [
            (-1, 0, cls.SPRINGINESS, cls.REST),
            (0, 1, cls.SPRINGINESS, cls.REST),
            (1, 2, cls.SPRINGINESS, cls.REST),
        ]


class DoubleRope(Simulation):
    MASS = 0.0010
    SPRINGINESS = 0.5
    REST = 0.2

    @classmethod
    def fixed_points(cls):
        return [(2, 8), (8, 8)]

    @classmethod
    def points(cls):
        return [((2 + 0.2 * i, 8), (0, 0), cls.MASS) for i in range(1, 30)]

    @classmethod
    def springs(cls):
        return [(-1, 0, cls.SPRINGINESS, cls.REST), (28, -2, cls.SPRINGINESS, cls.REST)] + \
            [(i, i + 1, cls.SPRINGINESS, cls.REST) for i in range(28)]


class Cloth(Simulation):
    MASS = 0.001
    SPRINGINESS = 0.2
    REST = 1

    @classmethod
    def fixed_points(cls):
        return [
            (2, 8),
            (8, 8),
        ]

    @classmethod
    def points(cls):
        return [
            ((2 + i, 8 - j), (0, 0), cls.MASS)
            for i in range(0, 7)
            for j in range(0, 6)
            if (i != 0 and i != 6) or j != 0
        ]

    @classmethod
    def springs(cls):
        return [
            (-1, 0, cls.SPRINGINESS, cls.REST),
            (-1, 5, cls.SPRINGINESS, cls.REST),
            (-2, 29, cls.SPRINGINESS, cls.REST),
            (-2, 35, cls.SPRINGINESS, cls.REST),
        ] + [
            (i, i + 1, cls.SPRINGINESS, cls.REST)
            for i in range(39)
            if i % 6 != 4
        ] + [
            (i, i + 5 + (i < 30), cls.SPRINGINESS, cls.REST)
            for i in range(35)
            if i != 29
        ]


class Benchmark(Simulation):
    MASS = 0.0005
    SPRINGINESS = 0.5
    REST = 0.2

    @classmethod
    def fixed_points(cls):
        return [(1 + 8 / 10 * x, 10) for x in range(10)]

    @classmethod
    def points(cls):
        return [((1 + 8 / 10 * x, 10 - 5 / 10 * y, cls.MASS), (0, 0, cls.MASS))
                for x in range(10)
                for y in range(9)]

    @classmethod
    def springs(cls):
        return [(-1 - x, x, cls.SPRINGINESS, cls.REST) for x in range(10)] + \
            [(10 * y + x, 10 * (y + 1) + x, cls.SPRINGINESS, cls.REST) for x in range(10) for y in range(8)] + \
            [(10 * y + x - 1, 10 * y + x, cls.SPRINGINESS, cls.REST) for x in range(1, 10) for y in range(8)] + \
            [(10 * y + x, 10 * y + x + 1, cls.SPRINGINESS, cls.REST) for x in range(0, 9) for y in range(8)]


class Building(Simulation):
    SPRINGINESS = 100
    MASS = 0.05

    H = 8

    @classmethod
    def fixed_points(cls):
        return [(i, 0) for i in range(11)]

    @classmethod
    def points(cls):
        return [
            ((i, j), (0, 0), cls.MASS / j)
            for i in range(3, 8)
            for j in range(1, cls.H)
        ]

    @classmethod
    def springs(cls):
        points = cls.points()
        return [
            (-4 - i, (cls.H - 1) * i, cls.SPRINGINESS, 1)
            for i in range(5)
        ] + [
            (-4 - i, (cls.H - 1) * (i + 1), cls.SPRINGINESS, 2 ** .5)
            for i in range(4)
        ] + [
            (-5 - i, (cls.H - 1) * i, cls.SPRINGINESS, 2 ** .5)
            for i in range(4)
        ] + [
            (i, j, cls.SPRINGINESS, d)
            for i, (a, _, _) in enumerate(points)
            for j, (b, _, _) in enumerate(points)
            if i < j and (d := ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) < 1.5
        ]


if __name__ == '__main__':
    sims = {
        'rope': Rope,
        'double-rope': DoubleRope,
        'cloth': Cloth,
        'benchmark': Benchmark,
        'building': Building,
    }[sys.argv[1]]().run()
