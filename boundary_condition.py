import numpy as np
import taichi as ti


@ti.data_oriented
class BoundaryCondition:
    def __init__(self, bc_const, bc_mask):
        self._bc_const, self._bc_mask = BoundaryCondition._to_field(bc_const, bc_mask)

    @ti.kernel
    def set_boundary_condition(self, vc: ti.template(), pc: ti.template()):
        bc_const, bc_mask = ti.static(self._bc_const, self._bc_mask)
        for i, j in vc:
            self._set_wall_bc(vc, bc_const, bc_mask, i, j)
            self._set_inflow_bc(vc, bc_const, bc_mask, i, j)
            self._set_outflow_bc(vc, pc, bc_mask, i, j)

    @ti.func
    def is_wall(self, i, j):
        return self._bc_mask[i, j] == 1

    def get_resolution(self):
        return self._bc_const.shape[:2]

    @ti.func
    def _set_wall_bc(self, vc, bc_const, bc_mask, i, j):
        if bc_mask[i, j] == 1:
            vc[i, j] = bc_const[i, j]

    @ti.func
    def _set_inflow_bc(self, vc, bc_const, bc_mask, i, j):
        if bc_mask[i, j] == 2:
            vc[i, j] = bc_const[i, j]

    @ti.func
    def _set_outflow_bc(self, vc, pc, bc_mask, i, j):
        if bc_mask[i, j] == 3:
            vc[i, j].x = min(max(vc[i - 1, j].x, 0.0), 10.0)  # 逆流しないようにする
            vc[i, j].y = min(max(vc[i - 1, j].y, -10.0), 10.0)
            pc[i, j] = 0.0

    @staticmethod
    def _to_field(bc, bc_mask):
        bc_field = ti.Vector.field(2, ti.f32, shape=bc.shape[:2])
        bc_field.from_numpy(bc)
        bc_mask_field = ti.field(ti.u8, shape=bc_mask.shape[:2])
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_mask_field

    @staticmethod
    def _set_circle(bc, bc_mask, i, j, radius):
        p = np.array([i, j])
        l_ = np.round(np.maximum(p - radius, 0)).astype(np.int32)
        u0 = round(min(p[0] + radius, bc.shape[0]))
        u1 = round(min(p[1] + radius, bc.shape[1]))
        for i in range(l_[0], u0):
            for j in range(l_[1], u1):
                x = np.array([i, j]) + 0.5
                if np.linalg.norm(x - p) < radius:
                    bc[i, j] = np.array([0.0, 0.0])
                    bc_mask[i, j] = 1



def create_boundary_condition1(resolution):
    # 1: 壁, 2: 流入部, 3: 流出部
    bc = np.zeros((2 * resolution, resolution, 2), dtype=np.float32)
    bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)

    # 流入部の設定
    bc[:2, :] = np.array([20.0, 0.0])
    bc_mask[:2, :] = 2

    # 流出部の設定
    bc[-1, :] = np.array([20.0, 0.0])
    bc_mask[-1, :] = 3

    # 壁の設定
    bc[:, :2] = np.array([0.0, 0.0])
    bc_mask[:, :2] = 1
    bc[:, -2:] = np.array([0.0, 0.0])
    bc_mask[:, -2:] = 1

    # 円柱の設定
    r = resolution // 18
    BoundaryCondition._set_circle(bc, bc_mask, resolution // 2 - r, resolution // 2, r)

    boundary_condition = BoundaryCondition(bc, bc_mask)

    return boundary_condition

