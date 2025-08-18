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

