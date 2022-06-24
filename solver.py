from abc import ABCMeta, abstractmethod

import taichi as ti

from differentiation import diff2_x, diff2_y, diff_x, diff_y, fdiff_x, fdiff_y, sample


class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(float, shape=resolution)
            self.next = ti.field(float, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, float, shape=resolution)
            self.next = ti.Vector.field(n_channel, float, shape=resolution)

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)


@ti.data_oriented
class Solver(metaclass=ABCMeta):
    def __init__(self, boundary_condition):
        self._bc = boundary_condition
        self._resolution = boundary_condition.get_resolution()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def get_fields(self):
        pass

    @ti.func
    def is_wall(self, i, j):
        return self._bc.is_wall(i, j)

    @ti.kernel
    def _clamp_field(self, field: ti.template(), low: float, high: float):
        for i, j in field:
            field[i, j] = ti.max(ti.min(field[i, j], high), low)


# これは音響では必要ない
@ti.func
def advect_upwind(vc, phi, i, j):
    """Upwind differencing
    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#tac8e468
    """
    k = i if vc[i, j].x < 0.0 else i - 1
    a = vc[i, j].x * fdiff_x(phi, k, j)

    k = j if vc[i, j].y < 0.0 else j - 1
    b = vc[i, j].y * fdiff_y(phi, i, k)

    return a + b


@ti.data_oriented
class FdtdSolver(Solver):
    """FDTD method"""

    def __init__(self, boundary_condition, dt, density, velocity):
        super().__init__(boundary_condition)

        self.dt = dt
        self.density = density
        self.velocity = velocity
        self.v = DoubleBuffers(self._resolution, 2)  # velocity
        self.p = DoubleBuffers(self._resolution, 1)  # pressure

        # initial condition
        self.v.current.fill(ti.Vector([0.0, 0.0]))

    def update(self):
        self._bc.set_boundary_condition(self.v.current, self.p.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self._clamp_field(self.v.next, -40.0, 40.0)
        self.v.swap()

        self._bc.set_boundary_condition(self.v.current, self.p.current)
        p_iter = 2
        for _ in range(p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

    def get_fields(self):
        return self.v.current, self.p.current

    @ti.kernel
    def _update_velocities(self, vn: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in vn:
            if not self._bc.is_wall(i, j):
                dx = fdiff_x(pc, i, j)
                dy = fdiff_y(pc, i, j)
                vn[i, j][0] = vc[i, j][0] - self.dt/(self.density*1) * dx
                vn[i, j][1] = vc[i, j][1] - self.dt/(self.density*1) * dy

    @ti.kernel
    def _update_pressures(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):
        for i, j in pn:
            if not self._bc.is_wall(i, j):
                dx = fdiff_x(vc[:,:][0], i, j)
                dy = fdiff_y(vc[:,:][1], i, j)
                pn[i, j] = pc[i, j] - ((self.density*self.velocity^2)/1)*(dx+dy)
