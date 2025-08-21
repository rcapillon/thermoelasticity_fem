import numpy as np
from scipy.sparse import csc_array


class Model:
    def __init__(self, mesh,
                 dict_dirichlet_U=None, dict_dirichlet_T=None,
                 dict_nodal_forces=None, dict_surface_forces=None, dict_volume_forces=None,
                 dict_heat_flux=None, dict_heat_source=None):
        self.mesh = mesh

        self.dict_dirichlet_U = dict_dirichlet_U
        self.dict_nodal_forces = dict_nodal_forces
        self.dict_surface_forces = dict_surface_forces
        self.dict_volume_forces = dict_volume_forces

        self.dict_dirichlet_T = dict_dirichlet_T
        self.dict_heat_flux = dict_heat_flux
        self.dict_heat_source = dict_heat_source

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
            self.mat_M[np.ix_(element.dofs_nums_u, element.dofs_nums_u)] += mat_Muu_e
        self.mat_M = csc_array(self.mat_M)

    def assemble_K(self):
        self.mat_K = np.zeros((self.mesh.n_dofs, self.mesh.n_dofs))
        for element in self.mesh.elements:
            mat_Kuu_e = element.compute_mat_Kuu_e()
            self.mat_K[np.ix_(element.dofs_nums_u, element.dofs_nums_u)] += mat_Kuu_e
            mat_Kut_e = element.compute_mat_Kut_e()
            self.mat_K[np.ix_(element.dofs_nums_u, element.dofs_nums_t)] += mat_Kut_e
            mat_Ktt_e = element.compute_mat_Ktt_e()
            self.mat_K[np.ix_(element.dofs_nums_t, element.dofs_nums_t)] += mat_Ktt_e
        self.mat_K = csc_array(self.mat_K)

    def assemble_D(self):
        self.mat_D = np.zeros((self.mesh.n_dofs, self.mesh.n_dofs))
        for element in self.mesh.elements:
            mat_Dtu_e = element.compute_mat_Dtu_e()
            self.mat_D[np.ix_(element.dofs_nums_t, element.dofs_nums_u)] += mat_Dtu_e
            mat_Dtt_e = element.compute_mat_Dtt_e()
            self.mat_D[np.ix_(element.dofs_nums_t, element.dofs_nums_t)] += mat_Dtt_e
        self.mat_D = csc_array(self.mat_D)

    def assemble_F(self):
        self.vec_F = np.zeros((self.mesh.n_dofs, ))

        # Nodal forces (in N)
        if self.dict_nodal_forces is not None:
            for tag, vec_f in self.dict_nodal_forces.items():
                nodes = self.mesh.dict_nodes_groups[tag]
                for node in nodes:
                    self.vec_F[(node * 4):(node * 4 + 3)] += vec_f
        # Surface forces (in N/m^2)
        if self.dict_surface_forces is not None:
            for tag, vec_f in self.dict_surface_forces.items():
                table_tri = self.mesh.dict_tri_groups[tag]
                for i in range(table_tri.shape[0]):
                    nodes = table_tri[i, :]
                    X1 = self.mesh.table_nodes[nodes[0], :]
                    X2 = self.mesh.table_nodes[nodes[1], :]
                    X3 = self.mesh.table_nodes[nodes[2], :]
                    X12 = X2 - X1
                    X13 = X3 - X1
                    area = 0.5 * np.abs(np.dot(X12, X13))
                    for node in nodes:
                        self.vec_F[(node * 4):(node * 4 + 3)] += area * vec_f / 3
        # Volume forces (in N/m^3)
        if self.dict_volume_forces is not None:
            for tag, vec_f in self.dict_volume_forces.items():
                table_tet = self.mesh.dict_tet_groups[tag]
                for i in range(table_tet.shape[0]):
                    nodes = table_tet[i, :]
                    X1 = self.mesh.table_nodes[nodes[0], :]
                    X2 = self.mesh.table_nodes[nodes[1], :]
                    X3 = self.mesh.table_nodes[nodes[2], :]
                    X4 = self.mesh.table_nodes[nodes[3], :]
                    X12 = X2 - X1
                    X13 = X3 - X1
                    X14 = X4 - X1
                    volume = np.abs(np.dot(X14, np.cross(X13, X12))) / 6
                    for node in nodes:
                        self.vec_F[(node * 4):(node * 4 + 3)] += volume * vec_f / 4
        # Heat flux (in W/m^2)
        if self.dict_heat_flux is not None:
            for tag, q in self.dict_heat_flux.items():
                table_tri = self.mesh.dict_tri_groups[tag]
                for i in range(table_tri.shape[0]):
                    nodes = table_tri[i, :]
                    X1 = self.mesh.table_nodes[nodes[0], :]
                    X2 = self.mesh.table_nodes[nodes[1], :]
                    X3 = self.mesh.table_nodes[nodes[2], :]
                    X12 = X2 - X1
                    X13 = X3 - X1
                    area = 0.5 * np.abs(np.dot(X12, X13))
                    for node in nodes:
                        self.vec_F[node * 4 + 3] -= area * q / 3
        # Heat source (in W/m^3)
        if self.dict_heat_source is not None:
            for tag, R in self.dict_heat_source.items():
                table_tet = self.mesh.dict_tet_groups[tag]
                for i in range(table_tet.shape[0]):
                    nodes = table_tet[i, :]
                    X1 = self.mesh.table_nodes[nodes[0], :]
                    X2 = self.mesh.table_nodes[nodes[1], :]
                    X3 = self.mesh.table_nodes[nodes[2], :]
                    X4 = self.mesh.table_nodes[nodes[3], :]
                    X12 = X2 - X1
                    X13 = X3 - X1
                    X14 = X4 - X1
                    volume = np.abs(np.dot(X14, np.cross(X13, X12))) / 6
                    for node in nodes:
                        self.vec_F[node * 4 + 3] += volume * R / 4

    def clear_full_matvec(self):
        self.mat_M = None
        self.mat_D = None
        self.mat_K = None
        self.vec_F = None

    def apply_dirichlet(self):
        if self.mat_M is not None:
            self.mat_M_f_f = self.mat_M[self.free_dofs, :][:, self.free_dofs]
        if self.mat_D is not None:
            self.mat_D_f_f = self.mat_D[self.free_dofs, :][:, self.free_dofs]
        if self.mat_K is not None:
            self.mat_K_f_f = self.mat_K[self.free_dofs, :][:, self.free_dofs]
        if self.vec_F is not None:
            self.vec_F_f = self.vec_F[self.free_dofs]
            if self.dict_dirichlet_U is not None and self.mat_K is not None:
                for tag, vec_u in self.dict_dirichlet_U.items():
                    dirichlet_nodes_U = self.mesh.dict_tri_groups[tag].flatten()
                    dirichlet_nodes_U = list(set(dirichlet_nodes_U))
                    vec_U = np.zeros((self.mesh.n_dofs, ))
                    dirichlet_dofs_U = []
                    for node in dirichlet_nodes_U:
                        dirichlet_dofs_U.extend([node * 4, node * 4 + 1, node * 4 + 2])
                        vec_U[[node * 4, node * 4 + 1, node * 4 + 2]] = vec_u
                    vec_U_d = vec_U[dirichlet_dofs_U]
                    mat_K_f_dU = self.mat_K[self.free_dofs, :][:, dirichlet_dofs_U]
                    self.vec_F_f -= mat_K_f_dU @ vec_U_d
            if self.dict_dirichlet_T is not None and self.mat_K is not None:
                for tag, T in self.dict_dirichlet_T.items():
                    dirichlet_nodes_T = self.mesh.dict_tri_groups[tag].flatten()
                    dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                    vec_T = np.zeros((self.mesh.n_dofs,))
                    dirichlet_dofs_T = []
                    for node in dirichlet_nodes_T:
                        dirichlet_dofs_T.append(node * 4 + 3)
                        vec_T[node * 4 + 3] = T
                    vec_T_d = vec_T[dirichlet_dofs_T]
                    mat_K_f_dT = self.mat_K[self.free_dofs, :][:, dirichlet_dofs_T]
                    self.vec_F_f -= mat_K_f_dT @ vec_T_d
        self.clear_full_matvec()
