import numpy as np


# Constant finite element matrices

s = np.sqrt(2)/2
mat_G = np.zeros((6, 9))
mat_G[0, 0] = 1
mat_G[1, 4] = 1
mat_G[2, 8] = 1
mat_G[3, 2] = s
mat_G[3, 6] = s
mat_G[4, 1] = s
mat_G[4, 3] = s
mat_G[5, 5] = s
mat_G[5, 7] = s

mat_P = np.zeros((9, 9))
mat_P[0, 0] = 1
mat_P[1, 3] = 1
mat_P[2, 6] = 1
mat_P[3, 1] = 1
mat_P[4, 4] = 1
mat_P[5, 7] = 1
mat_P[6, 2] = 1
mat_P[7, 5] = 1
mat_P[8, 8] = 1

# Reference coordinates of the nodes

nodes_reference_coords = np.array([[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])

# Shape functions coefficients

x0 = nodes_reference_coords[0, 0]
y0 = nodes_reference_coords[0, 1]
z0 = nodes_reference_coords[0, 2]
x1 = nodes_reference_coords[1, 0]
y1 = nodes_reference_coords[1, 1]
z1 = nodes_reference_coords[1, 2]
x2 = nodes_reference_coords[2, 0]
y2 = nodes_reference_coords[2, 1]
z2 = nodes_reference_coords[2, 2]
x3 = nodes_reference_coords[3, 0]
y3 = nodes_reference_coords[3, 1]
z3 = nodes_reference_coords[3, 2]

mat_A = np.array([[1, x0, y0, z0],
                  [1, x1, y1, z1],
                  [1, x2, y2, z2],
                  [1, x3, y3, z3]])

shapefun_coeffs = np.linalg.solve(mat_A, np.eye(4))

# Gauss points and weights

n_gauss = 4
gauss = [(np.array([0.5854101966, 0.1381966011, 0.1381966011]), 0.0416666667),
         (np.array([0.1381966011, 0.5854101966, 0.1381966011]), 0.0416666667),
         (np.array([0.1381966011, 0.1381966011, 0.5854101966]), 0.0416666667),
         (np.array([0.1381966011, 0.1381966011, 0.1381966011]), 0.0416666667)]

####
# Functions used for calculating element matrices at gauss points

def shapefun_value(node_idx, reference_coords):
    # N_i(x, y, z) = a + b*x + c*y + d*z

    x = reference_coords[0]
    y = reference_coords[1]
    z = reference_coords[2]

    value = shapefun_coeffs[0, node_idx] \
            + shapefun_coeffs[1, node_idx] * x \
            + shapefun_coeffs[2, node_idx] * y \
            + shapefun_coeffs[3, node_idx] * z

    return value


def derivative_shapefun_value(shapefun_idx, derivative_coord_idx):
    # dNdx_i(x, y, z) = b
    # dNdy_i(x, y, z) = c
    # dNdz_i(x, y, z) = d

    # derivative_coord_idx: 1 -> derivative with respect to x
    # derivative_coord_idx: 2 -> derivative with respect to y
    # derivative_coord_idx: 3 -> derivative with respect to z

    value = shapefun_coeffs[derivative_coord_idx, shapefun_idx]

    return value


def compute_mat_Ee(reference_coords):
    mat_I = np.eye(3)
    mat_E0 = shapefun_value(0, reference_coords) * mat_I
    mat_E1 = shapefun_value(1, reference_coords) * mat_I
    mat_E2 = shapefun_value(2, reference_coords) * mat_I
    mat_E3 = shapefun_value(3, reference_coords) * mat_I

    mat_Ee = np.concatenate((mat_E0, mat_E1, mat_E2, mat_E3), axis=1)

    return mat_Ee


def compute_mat_De():
    mat_I = np.eye(3)

    mat_D0dx = derivative_shapefun_value(0, 1) * mat_I
    mat_D1dx = derivative_shapefun_value(1, 1) * mat_I
    mat_D2dx = derivative_shapefun_value(2, 1) * mat_I
    mat_D3dx = derivative_shapefun_value(3, 1) * mat_I

    mat_Ddx = np.concatenate((mat_D0dx, mat_D1dx, mat_D2dx, mat_D3dx), axis=1)

    mat_D0dy = derivative_shapefun_value(0, 2) * mat_I
    mat_D1dy = derivative_shapefun_value(1, 2) * mat_I
    mat_D2dy = derivative_shapefun_value(2, 2) * mat_I
    mat_D3dy = derivative_shapefun_value(3, 2) * mat_I

    mat_Ddy = np.concatenate((mat_D0dy, mat_D1dy, mat_D2dy, mat_D3dy), axis=1)

    mat_D0dz = derivative_shapefun_value(0, 3) * mat_I
    mat_D1dz = derivative_shapefun_value(1, 3) * mat_I
    mat_D2dz = derivative_shapefun_value(2, 3) * mat_I
    mat_D3dz = derivative_shapefun_value(3, 3) * mat_I

    mat_Ddz = np.concatenate((mat_D0dz, mat_D1dz, mat_D2dz, mat_D3dz), axis=1)

    mat_De = np.concatenate((mat_Ddx, mat_Ddy, mat_Ddz), axis=0)

    return mat_De


class Element:
    """
    Class for 4-node tetrahedal elements
    """
    def __init__(self, number, material, nodes_nums, nodes_coords):
        self.number = number
        self.material = material
        self.nodes_nums = nodes_nums
        self.nodes_coords = nodes_coords

        self.dofs_nums_u = []
        for node_num in nodes_nums:
            self.dofs_nums_u.extend([node_num * 4, node_num * 4 + 1, node_num * 4 +2])
        self.dofs_nums_t = []
        for node_num in nodes_nums:
            self.dofs_nums_t.append(node_num * 4 + 3)

        self.det_J = None
        self.mat_invJJJ = None

        self.mat_Muu_e = None
        self.mat_Mtu_e = None
        self.mat_Mtt_e = None

        self.mat_Dtu_e = None
        self.mat_Dtt_e = None

        self.mat_Kuu_e = None
        self.mat_Kut_e = None
        self.mat_Ktt_e = None

