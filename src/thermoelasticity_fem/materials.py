import numpy as np


class LinearThermoElastic:
    def __init__(self, rho, Y, nu, k, c, alpha, T0):
        self.rho = rho
        self.Y = Y
        self.nu = nu
        self.k = k
        self.c = c
        self.alpha = alpha
        self.T0 = T0
        self.beta = None
        self.compute_beta()
        self.mat_C = None
        self.compute_mat_C()

    def compute_beta(self):
        lame1 = (self.Y * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        lame2 = self.Y / (2 * (1 + self.nu))
        self.beta = (3 * lame1 + 2 * lame2) * self.alpha

    def compute_mat_C(self):
        lame1 = (self.Y * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        lame2 = self.Y / (2 * (1 + self.nu))

        repmat_lame1 = np.tile(lame1, (3, 3))
        self.mat_C = 2 * lame2 * np.eye(6)
        self.mat_C[:3, :3] += repmat_lame1


# based on glass SG773 in [2]
glass_SG773 = LinearThermoElastic(rho=2300, Y=64e9, nu=0.1, k=1., c=750., alpha=4e-6, T0=20.)
