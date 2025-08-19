import numpy as np
from scipy.sparse import csc_array


class Model:
    def __init__(self, mesh,
                 dict_dirichlet_U=None, dict_dirichlet_T=None,
                 dict_nodal_forces=None, dict_surface_forces=None, dict_volume_forces=None,
                 dict_heat_flux=None):
                 # alpha_M=None, alpha_K=None):
        self.mesh = mesh

        self.dict_dirichlet_U = dict_dirichlet_U
        self.dict_nodal_forces= dict_nodal_forces
        self.dict_surface_forces = dict_surface_forces
        self.dict_volume_forces = dict_volume_forces

        self.dict_dirichlet_T = dict_dirichlet_T
        self.dict_heat_flux = dict_heat_flux

        # self.alpha_M = alpha_M
        # self.alpha_K = alpha_K

        self.free_dofs = None

        self.mat_M = None
        self.mat_D = None
        self.mat_K = None
        self.vec_F = None

        self.mat_M_f_f = None
        self.mat_D_f_f = None
        self.mat_K_f_f = None
        self.vec_F_f = None

    def create_free_dofs_lists(self):
        dirichlet_nodes_U = []
        if self.dict_dirichlet_U is not None:
            for k in self.dict_dirichlet_U.keys():
                dirichlet_nodes_U.extend(list(self.mesh.dict_tri_groups[k].flatten()))
            dirichlet_nodes_U = list(set(dirichlet_nodes_U))
        dirichlet_dofs_U = []
        for node in dirichlet_nodes_U:
            dirichlet_dofs_U.extend([node * 4, node * 4 + 1, node * 4 + 2])
        dirichlet_nodes_T = []
        if self.dict_dirichlet_T is not None:
            for k in self.dict_dirichlet_T.keys():
                dirichlet_nodes_T.extend(list(self.mesh.dict_tri_groups[k].flatten()))
            dirichlet_nodes_T = list(set(dirichlet_nodes_T))
        dirichlet_dofs_T = []
        for node in dirichlet_nodes_T:
            dirichlet_dofs_T.append(node * 4 + 3)
        all_dirichlet_dofs = dirichlet_dofs_U + dirichlet_dofs_T
        self.free_dofs = [dof for dof in range(self.mesh.n_dofs) if dof not in all_dirichlet_dofs]

    def assemble_M(self):
        self.mat_M = np.zeros((self.mesh.n_dofs, self.mesh.n_dofs))
        for element in self.mesh.elements:
            mat_Muu_e = element.compute_mat_Muu_e()
            self.mat_M[element.dofs_nums_u, element.dofs_nums_u] += mat_Muu_e

    def assemble_K(self):
        self.mat_K = np.zeros((self.mesh.n_dofs, self.mesh.n_dofs))
        for element in self.mesh.elements:
            mat_Kuu_e = element.compute_mat_Kuu_e()
            self.mat_K[element.dofs_nums_u, element.dofs_nums_u] += mat_Kuu_e
            mat_Kut_e = element.compute_mat_Kut_e()
            self.mat_K[element.dofs_nums_u, element.dofs_nums_t] += mat_Kut_e
            mat_Ktt_e = element.compute_mat_Ktt_e()
            self.mat_K[element.dofs_nums_t, element.dofs_nums_t] += mat_Ktt_e

    def assemble_D(self):
        self.mat_D = np.zeros((self.mesh.n_dofs, self.mesh.n_dofs))
        for element in self.mesh.elements:
            mat_Dtu_e = element.compute_mat_Dtu_e()
            self.mat_D[element.dofs_nums_t, element.dofs_nums_u] += mat_Dtu_e
            mat_Dtt_e = element.compute_mat_Dtt_e()
            self.mat_D[element.dofs_nums_t, element.dofs_nums_t] += mat_Dtt_e

    def clear_full_matvec(self):
        self.mat_M = None
        self.mat_D = None
        self.mat_K = None
        self.vec_F = None
