import hashlib
import importlib
import inspect
import pathlib
import pickle
import re
import sys
import time
from typing import TypeVar

import appdirs
import numpy
import numpy as np
import sympy
from more_itertools import flatten
from tqdm.auto import tqdm

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


class Simulation:
    # required
    SPRINGS = None
    POINTS = None
    FIXED_POINTS = None

    # optional
    GRAVITY = 9.81
    MASS = 0.05
    SPRINGINESS = 0.5
    DAMPING = 0.003
    REST = 0.25

    # time
    STEPS = 10
    FPS = 60
    WIDTH = 1920 / 2
    HEIGHT = 1080 / 2

    def __init__(self):
        self.fixed_points = None
        self.points = None
        self.springs = None
        self.range_kutta = None
        self.hamiltonian = None

        data = pickle.dumps((self.FIXED_POINTS,
                             self.POINTS, self.MASS, self.GRAVITY,
                             self.SPRINGS, self.SPRINGINESS, self.REST))
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

    @staticmethod
    def resolve_point(index: int, points: [T], fixed_points: [T]) -> T:
        return points[index] if index >= 0 else fixed_points[-index - 1]

    def generate(self):
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
            *flatten([f'qd[{i * 2}]', f'qd[{i * 2 + 1}]']
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
        return f'from numpy import *\n' \
               f'from numba import njit\n' \
               f'\n' \
               f'@njit(fastmath=True, parallel=False, cache=True)\n' \
               f'def fast_hamiltonian(q, qd):\n' \
               f'    return {hamiltonian_vectorized.strip().removesuffix(",")}' \
               f'\n' \
               '@njit(fastmath=True, parallel=False, cache=True)\n' \
               'def fast_range_kutta(q, qd, delta):\n' \
               '    k1 = qd\n' \
               '    k1d = _fast_lagrangian(q, qd)\n' \
               '    k2 = qd + 0.5 * k1d * delta\n' \
               '    k2d = _fast_lagrangian(q + 0.5 * k1 * delta, qd + 0.5 * k1d * delta)\n' \
               '    k3 = qd + 0.5 * k2d * delta\n' \
               '    k3d = _fast_lagrangian(q + 0.5 * k2 * delta, qd + 0.5 * k2d * delta)\n' \
               '    k4 = qd + k3d * delta\n' \
               '    k4d = _fast_lagrangian(q + k3 * delta, qd + k3d * delta)\n' \
               '    q += (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6\n' \
               '    qd += (k1d + 2 * k2d + 2 * k3d + k4d) * delta / 6\n' \
               '    return q, qd\n' \
               f'\n' \
               f'@njit(fastmath=True, parallel=False, cache=True)\n' \
               f'def _fast_lagrangian(q, qd):\n' \
               f'    return array([\n' \
               f'{vectorized}' \
               f'    ])\n' \
               f'\n' \
               f'def hamiltonian(q, qd):\n' \
               f'    return {hamiltonian_vectorized.strip().removesuffix(",")}' \
               f'\n' \
               'def range_kutta(q, qd, delta):\n' \
               '    k1 = qd\n' \
               '    k1d = _lagrangian(q, qd)\n' \
               '    k2 = qd + 0.5 * k1d * delta\n' \
               '    k2d = _lagrangian(q + 0.5 * k1 * delta, qd + 0.5 * k1d * delta)\n' \
               '    k3 = qd + 0.5 * k2d * delta\n' \
               '    k3d = _lagrangian(q + 0.5 * k2 * delta, qd + 0.5 * k2d * delta)\n' \
               '    k4 = qd + k3d * delta\n' \
               '    k4d = _lagrangian(q + k3 * delta, qd + k3d * delta)\n' \
               '    q += (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6\n' \
               '    qd += (k1d + 2 * k2d + 2 * k3d + k4d) * delta / 6\n' \
               '    return q, qd\n' \
               f'\n' \
               f'def _lagrangian(q, qd):\n' \
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

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for _ in range(self.STEPS):
                q, qd = self.range_kutta(q, qd, dt / self.STEPS)
            qd *= 1 - self.DAMPING
            h = self.hamiltonian(q, qd)

            # fill the screen with a color to wipe away anything from last frame
            screen.fill(pygame.color.Color(51, 51, 51))

            pygame.draw.line(screen, pygame.color.Color(255, 255, 255), self.transform(0, 0), self.transform(10, 0), 2)
            pygame.draw.circle(screen, pygame.color.Color(93, 194, 57), self.transform(h, 0), 5)

            for i, (a, b) in enumerate(self.SPRINGS):
                p1 = (q[a * 2], q[a * 2 + 1]) if a >= 0 else self.FIXED_POINTS[-a - 1]
                p2 = (q[b * 2], q[b * 2 + 1]) if b >= 0 else self.FIXED_POINTS[-b - 1]
                pygame.draw.line(screen, pygame.color.Color(225, 82, 82), self.transform(*p1), self.transform(*p2), 2)

            for pos in self.FIXED_POINTS:
                pygame.draw.circle(screen, pygame.color.Color(236, 140, 40), self.transform(*pos), 5)

            for i in range(len(self.POINTS)):
                pygame.draw.circle(screen, pygame.color.Color(225, 82, 82), self.transform(q[i * 2], q[i * 2 + 1]), 5)

            pygame.display.flip()
            dt = clock.tick(self.FPS) / 1000

    def transform(self, x, y):
        scale = min(self.WIDTH, self.HEIGHT) / 12
        return scale * x + self.WIDTH / 2 - 5 * scale, \
               -scale * y + self.HEIGHT / 2 + 5 * scale


class Rope(Simulation):
    MASS = 0.025
    SPRINGINESS = 0.25
    DAMPING = 0.005

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
    MASS = 0.0010
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

    FIXED_POINTS = [(1 + 8 / 10 * x, 10) for x in range(10)]
    POINTS = [((1 + 8 / 10 * x, 10 - 5 / 10 * y), (0, 0))
              for x in range(10)
              for y in range(9)]
    SPRINGS = [(-1 - x, x) for x in range(10)] + \
              [(10 * y + x, 10 * (y + 1) + x) for x in range(10) for y in range(8)] + \
              [(10 * y + x - 1, 10 * y + x) for x in range(1, 10) for y in range(8)] + \
              [(10 * y + x, 10 * y + x + 1) for x in range(0, 9) for y in range(8)]


if __name__ == '__main__':
    sims = {
        'rope': Rope,
        'double-rope': DoubleRope,
        'cloth': Cloth,
        'benchmark': Benchmark,
    }[sys.argv[1]]().run()
