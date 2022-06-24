import numpy as np
import taichi as ti


@ti.data_oriented
class BoundaryCondition:
    def __init__(self, bc_mask):
        self._bc_mask = BoundaryCondition._to_field(bc_mask)

    @ti.kernel
    def set_boundary_condition(self, vc: ti.template(), pc: ti.template()):
        bc_mask = ti.static(self._bc_mask)
        for i, j in vc:
            self._set_wall_bc(vc, bc_mask, i, j)

    @ti.func
    def is_wall(self, i, j):
        return self._bc_mask[i, j] == 1

    def get_resolution(self):
        return self._bc_mask.shape[:2]

    @ti.func
    def _set_wall_bc(self, vc, bc_mask, i, j):
        if bc_mask[i, j] == 1:
            pass

    @staticmethod
    def _to_field(bc_mask):
        bc_mask_field = ti.field(ti.u8, shape=bc_mask.shape[:2])
        bc_mask_field.from_numpy(bc_mask)

        return bc_mask_field

    @staticmethod
    def _set_circle(bc_mask, i, j, radius):
        p = np.array([i, j])
        l_ = np.round(np.maximum(p - radius, 0)).astype(np.int32)
        u0 = round(min(p[0] + radius, bc_mask.shape[0]))
        u1 = round(min(p[1] + radius, bc_mask.shape[1]))
        for i in range(l_[0], u0):
            for j in range(l_[1], u1):
                x = np.array([i, j]) + 0.5
                if np.linalg.norm(x - p) < radius:
                    bc_mask[i, j] = 1


def create_boundary_condition1(resolution):
    # 1: 壁
    bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)
    # 壁の設定
    bc_mask[:2, :] = 1
    bc_mask[-1, :] = 1
    bc_mask[:, :2] = 1
    bc_mask[:, -2:] = 1

    # 円柱の設定
    r = resolution // 18
    BoundaryCondition._set_circle(
        bc_mask, resolution // 2 - r, resolution // 2, r)

    boundary_condition = BoundaryCondition(bc_mask)

    return boundary_condition
