import numpy as np
from scipy.sparse.linalg import spsolve


class LinearThermoStatics:
    def __init__(self, model):
        self.model = model

        self.X = None
        self.displacement = None
        self.temperature = None

    def solve(self):
        self.model.create_free_dofs_lists()
        self.model.assemble_K()
        self.model.assemble_F()
        self.model.apply_dirichlet()

        vec_X_f = spsolve(self.model.mat_K_f_f, self.model.vec_F_f)
        self.X = np.zeros((self.model.mesh.n_dofs, ))
        self.X[self.model.free_dofs] = vec_X_f
        for tag, vec_u in self.model.dict_dirichlet_U.items():
            dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
            dirichlet_nodes_U = list(set(dirichlet_nodes_U))
            for node in dirichlet_nodes_U:
                self.X[[node * 4, node * 4 + 1, node * 4 + 2]] += vec_u
        for tag, T in self.model.dict_dirichlet_T.items():
            dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
            dirichlet_nodes_T = list(set(dirichlet_nodes_T))
            for node in dirichlet_nodes_T:
                self.X[node * 4 + 3] += T

        self.displacement = np.zeros((self.model.mesh.n_nodes * 3, ))
        self.displacement[::3] = self.X[::4]
        self.displacement[1::3] = self.X[1::4]
        self.displacement[2::3] = self.X[2::4]
        self.temperature = self.X[3::4]
