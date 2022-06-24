
import taichi as ti

from boundary_condition import create_boundary_condition1

from solver import FdtdSolver


@ti.func
def visualize_norm(vec):
    c = vec.norm()
    return ti.Vector([c, c, c])


@ti.func
def visualize_pressure(val):
    return ti.Vector([max(val, 0.0), 0.0, max(-val, 0.0)])


@ti.data_oriented
class Simulator:
    def __init__(self, solver):
        self._solver = solver
        self.rgb_buf = ti.Vector.field(
            3, float, shape=solver._resolution)  # image buffer
        self._wall_color = ti.Vector([0.5, 0.7, 0.5])  # 壁の色

    def step(self):
        self._solver.update()

    def get_norm_field(self):
        self._to_norm(self.rgb_buf, *self._solver.get_fields()[:2])
        return self.rgb_buf

    def get_pressure_field(self):
        self._to_pressure(self.rgb_buf, self._solver.get_fields()[1])
        return self.rgb_buf

    @ti.kernel
    def _to_norm(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.025 * visualize_norm(vc[i, j])
            rgb_buf[i, j] += 0.0006 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_pressure(self, rgb_buf: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.004 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(resolution, dt, density, velocity):
        boundary_condition = create_boundary_condition1(resolution)

        solver = FdtdSolver(boundary_condition, dt, density, velocity)

        return Simulator(solver)
