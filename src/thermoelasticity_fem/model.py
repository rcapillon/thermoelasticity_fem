import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import eigs


class Model:
    def __init__(self, mesh,
                 dict_dirichlet_U=None, dict_dirichlet_theta=None,
                 dict_nodal_forces=None, dict_surface_forces=None, dict_volume_forces=None,
                 dict_heat_flux=None, dict_heat_source=None,
                 alpha_M=0., alpha_K=0.):
        self.mesh = mesh

        self.dict_dirichlet_U = dict_dirichlet_U
        self.dict_nodal_forces = dict_nodal_forces
        self.dict_surface_forces = dict_surface_forces
        self.dict_volume_forces = dict_volume_forces

        self.dict_dirichlet_theta = dict_dirichlet_theta
        self.dict_heat_flux = dict_heat_flux
        self.dict_heat_source = dict_heat_source

        self.alpha_M = alpha_M
        self.alpha_K = alpha_K

        self.free_dofs = None
        self.free_dofs_U = None
        self.free_dofs_theta = None
        self.dirichlet_dofs_U = None
        self.dirichlet_dofs_theta = None

        self.mat_M = None
        self.mat_D = None
        self.mat_K = None
        self.vec_F = None

        self.mat_M_f_f = None
        self.mat_D_f_f = None
        self.mat_K_f_f = None
        self.vec_F_f = None

        self.n_q_u = None
        self.n_q_t = None
        self.mat_phi_u = None
        self.mat_phi_t = None
        self.mat_Mrom = None
        self.mat_Drom = None
        self.mat_Krom = None
        self.vec_From = None

    def create_free_dofs_lists(self):
        self.dirichlet_dofs_U = []
        if self.dict_dirichlet_U is not None:
            for tag, list_dir_u in self.dict_dirichlet_U.items():
                nodes = list(self.mesh.dict_tri_groups[tag].flatten())
                for str_dof, _ in list_dir_u:
                    if str_dof == 'x':
                        modif = 0
                    elif str_dof == 'y':
                        modif = 1
                    elif str_dof == 'z':
                        modif = 2
                    else:
                        raise ValueError('Unrecognized DOF.')
                    self.dirichlet_dofs_U.extend([node * 4 + modif for node in nodes])
            self.dirichlet_dofs_U = list(set(self.dirichlet_dofs_U))
        dirichlet_nodes_T = []
        if self.dict_dirichlet_theta is not None:
            for k in self.dict_dirichlet_theta.keys():
                dirichlet_nodes_T.extend(list(self.mesh.dict_tri_groups[k].flatten()))
            dirichlet_nodes_T = list(set(dirichlet_nodes_T))
        self.dirichlet_dofs_theta = []
        for node in dirichlet_nodes_T:
            self.dirichlet_dofs_theta.append(node * 4 + 3)
        all_dirichlet_dofs = self.dirichlet_dofs_U + self.dirichlet_dofs_theta
        self.free_dofs = [dof for dof in range(self.mesh.n_dofs) if dof not in all_dirichlet_dofs]
        self.free_dofs_U = []
        self.free_dofs_theta = []
        self.free_dofs_U.extend([node * 4 for node in range(self.mesh.n_nodes) if node * 4 not in all_dirichlet_dofs])
        self.free_dofs_U.extend([node * 4 + 1 for node in range(self.mesh.n_nodes)
                                 if node * 4 + 1 not in all_dirichlet_dofs])
        self.free_dofs_U.extend([node * 4 + 2 for node in range(self.mesh.n_nodes)
                                 if node * 4 + 2 not in all_dirichlet_dofs])
        self.free_dofs_theta.extend([node * 4 + 3 for node in range(self.mesh.n_nodes)
                                     if node * 4 + 3 not in all_dirichlet_dofs])

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
            if self.alpha_M != 0.:
                mat_Muu_e = element.compute_mat_Muu_e()
                self.mat_D[np.ix_(element.dofs_nums_u, element.dofs_nums_u)] += self.alpha_M * mat_Muu_e
            if self.alpha_K != 0.:
                mat_Kuu_e = element.compute_mat_Kuu_e()
                self.mat_D[np.ix_(element.dofs_nums_u, element.dofs_nums_u)] += self.alpha_K * mat_Kuu_e
        self.mat_D = csc_array(self.mat_D)

    def assemble_F(self, idx_t=None):
        self.vec_F = np.zeros((self.mesh.n_dofs, ))

        # Nodal forces (in N)
        if self.dict_nodal_forces is not None:
            if idx_t is None:
                for tag, vec_f in self.dict_nodal_forces.items():
                    nodes = self.mesh.dict_nodes_groups[tag]
                    for node in nodes:
                        self.vec_F[(node * 4):(node * 4 + 3)] += vec_f
            else:
                for tag, arr_f in self.dict_nodal_forces.items():
                    nodes = self.mesh.dict_nodes_groups[tag]
                    if arr_f.ndim == 2:
                        for node in nodes:
                            self.vec_F[(node * 4):(node * 4 + 3)] += arr_f[:, idx_t]
                    elif arr_f.ndim == 1:
                        for node in nodes:
                            self.vec_F[(node * 4):(node * 4 + 3)] += arr_f
                    else:
                        raise ValueError('Prescribed nodal force array should have 1 or 2 dimensions.')
        # Surface forces (in N/m^2)
        if self.dict_surface_forces is not None:
            if idx_t is None:
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
            else:
                for tag, arr_f in self.dict_surface_forces.items():
                    table_tri = self.mesh.dict_tri_groups[tag]
                    for i in range(table_tri.shape[0]):
                        nodes = table_tri[i, :]
                        X1 = self.mesh.table_nodes[nodes[0], :]
                        X2 = self.mesh.table_nodes[nodes[1], :]
                        X3 = self.mesh.table_nodes[nodes[2], :]
                        X12 = X2 - X1
                        X13 = X3 - X1
                        area = 0.5 * np.abs(np.dot(X12, X13))
                        if arr_f.ndim == 2:
                            for node in nodes:
                                self.vec_F[(node * 4):(node * 4 + 3)] += area * arr_f[:, idx_t] / 3
                        elif arr_f.ndim == 1:
                            for node in nodes:
                                self.vec_F[(node * 4):(node * 4 + 3)] += area * arr_f / 3
                        else:
                            raise ValueError('Prescribed surface force array should have 1 or 2 dimensions.')
        # Volume forces (in N/m^3)
        if self.dict_volume_forces is not None:
            if idx_t is None:
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
            else:
                for tag, arr_f in self.dict_volume_forces.items():
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
                        if arr_f.ndim == 2:
                            for node in nodes:
                                self.vec_F[(node * 4):(node * 4 + 3)] += volume * arr_f[:, idx_t] / 4
                        elif arr_f.ndim == 1:
                            for node in nodes:
                                self.vec_F[(node * 4):(node * 4 + 3)] += volume * arr_f / 4
                        else:
                            raise ValueError('Prescribed volume force array should have 1 or 2 dimensions.')
        # Heat flux (in W/m^2)
        if self.dict_heat_flux is not None:
            if idx_t is None:
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
            else:
                for tag, arr_q in self.dict_heat_flux.items():
                    table_tri = self.mesh.dict_tri_groups[tag]
                    for i in range(table_tri.shape[0]):
                        nodes = table_tri[i, :]
                        X1 = self.mesh.table_nodes[nodes[0], :]
                        X2 = self.mesh.table_nodes[nodes[1], :]
                        X3 = self.mesh.table_nodes[nodes[2], :]
                        X12 = X2 - X1
                        X13 = X3 - X1
                        area = 0.5 * np.abs(np.dot(X12, X13))
                        if not isinstance(arr_q, float):
                            if arr_q.ndim == 1:
                                for node in nodes:
                                    self.vec_F[node * 4 + 3] -= area * arr_q[idx_t] / 3
                            else:
                                raise ValueError('Prescribed flux array should be a float or have 1 dimension.')
                        else:
                            for node in nodes:
                                self.vec_F[node * 4 + 3] -= area * arr_q / 3
        # Heat source (in W/m^3)
        if self.dict_heat_source is not None:
            if idx_t is None:
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
            else:
                for tag, arr_R in self.dict_heat_source.items():
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
                        if not isinstance(arr_R, float):
                            if arr_R.ndim == 1:
                                for node in nodes:
                                    self.vec_F[node * 4 + 3] += volume * arr_R[idx_t] / 4
                            else:
                                raise ValueError('Prescribed heat source array should be a float or have 1 dimension.')
                        else:
                            for node in nodes:
                                self.vec_F[node * 4 + 3] += volume * arr_R / 4

    def clear_full_matrices(self):
        self.mat_M = None
        self.mat_D = None
        self.mat_K = None

    def clear_full_F(self):
        self.vec_F = None

    def apply_dirichlet_matrices(self):
        if self.mat_M is not None:
            self.mat_M_f_f = self.mat_M[self.free_dofs, :][:, self.free_dofs]
        if self.mat_D is not None:
            self.mat_D_f_f = self.mat_D[self.free_dofs, :][:, self.free_dofs]
        if self.mat_K is not None:
            self.mat_K_f_f = self.mat_K[self.free_dofs, :][:, self.free_dofs]

    def apply_dirichlet_F(self):
        if self.vec_F is not None:
            self.vec_F_f = self.vec_F[self.free_dofs]
            if self.dict_dirichlet_U is not None and self.mat_K is not None:
                for tag, list_u_dir in self.dict_dirichlet_U.items():
                    dirichlet_nodes_U = self.mesh.dict_tri_groups[tag].flatten()
                    dirichlet_nodes_U = list(set(dirichlet_nodes_U))
                    vec_U = np.zeros((self.mesh.n_dofs, ))
                    dirichlet_dofs_U = []
                    for node in dirichlet_nodes_U:
                        for str_dof, val_u in list_u_dir:
                            if str_dof == 'x':
                                modif = 0
                            elif str_dof == 'y':
                                modif = 1
                            elif str_dof == 'z':
                                modif = 2
                            else:
                                raise ValueError('Unrecognized DOF.')
                            dirichlet_dofs_U.append(node * 4 + modif)
                            vec_U[node * 4 + modif] = val_u
                    vec_U_d = vec_U[dirichlet_dofs_U]
                    mat_K_f_du = self.mat_K[self.free_dofs, :][:, dirichlet_dofs_U]
                    self.vec_F_f -= mat_K_f_du @ vec_U_d
            if self.dict_dirichlet_theta is not None and self.mat_K is not None:
                for tag, theta in self.dict_dirichlet_theta.items():
                    dirichlet_nodes_theta = self.mesh.dict_tri_groups[tag].flatten()
                    dirichlet_nodes_theta = list(set(dirichlet_nodes_theta))
                    vec_theta = np.zeros((self.mesh.n_dofs,))
                    dirichlet_dofs_theta = []
                    for node in dirichlet_nodes_theta:
                        dirichlet_dofs_theta.append(node * 4 + 3)
                        vec_theta[node * 4 + 3] = theta
                    vec_theta_d = vec_theta[dirichlet_dofs_theta]
                    mat_K_f_dt = self.mat_K[self.free_dofs, :][:, dirichlet_dofs_theta]
                    self.vec_F_f -= mat_K_f_dt @ vec_theta_d

    def compute_ROB_u(self, n_q_u):
        # must have already called method create_dof_lists and have global finite element matrices M and K assembled

        self.n_q_u = n_q_u

        mat_Muu = self.mat_M[self.free_dofs_U, :][:, self.free_dofs_U]
        mat_Kuu = self.mat_K[self.free_dofs_U, :][:, self.free_dofs_U]

        _, self.mat_phi_u = eigs(mat_Kuu, k=self.n_q_u, M=mat_Muu, which='SM')

        self.mat_phi_u = np.real(self.mat_phi_u)

    def compute_ROB_theta(self, n_q_t):
        # must have already called method create_dof_lists and have global finite element matrices D and K assembled

        self.n_q_t = n_q_t

        mat_Dtt = self.mat_D[self.free_dofs_theta, :][:, self.free_dofs_theta]
        mat_Ktt = self.mat_K[self.free_dofs_theta, :][:, self.free_dofs_theta]

        _, self.mat_phi_t = eigs(mat_Ktt, k=self.n_q_t, M=mat_Dtt, which='SM')

        self.mat_phi_t = np.real(self.mat_phi_t)

    def compute_ROM_matrices(self):
        # must have already called methods compute_ROB_u and compute_ROB_theta and have global finite element matrices
        # M, D, K assembled

        mat_Mrom_uu = (self.mat_phi_u.transpose() @ self.mat_M[self.free_dofs_U, :][:, self.free_dofs_U]
                       @ self.mat_phi_u)
        mat_Drom_uu = (self.mat_phi_u.transpose() @ self.mat_D[self.free_dofs_U, :][:, self.free_dofs_U]
                       @ self.mat_phi_u)
        mat_Drom_tu = (self.mat_phi_t.transpose() @ self.mat_D[self.free_dofs_theta, :][:, self.free_dofs_U]
                       @ self.mat_phi_u)
        mat_Drom_tt = (self.mat_phi_t.transpose() @ self.mat_D[self.free_dofs_theta, :][:, self.free_dofs_theta]
                       @ self.mat_phi_t)
        mat_Krom_uu = (self.mat_phi_u.transpose() @ self.mat_K[self.free_dofs_U, :][:, self.free_dofs_U]
                       @ self.mat_phi_u)
        mat_Krom_ut = (self.mat_phi_u.transpose() @ self.mat_K[self.free_dofs_U, :][:, self.free_dofs_theta]
                       @ self.mat_phi_t)
        mat_Krom_tt = (self.mat_phi_t.transpose() @ self.mat_K[self.free_dofs_theta, :][:, self.free_dofs_theta]
                       @ self.mat_phi_t)

        self.mat_Mrom = np.zeros((self.n_q_u + self.n_q_t, self.n_q_u + self.n_q_t))
        self.mat_Mrom[:self.n_q_u, :self.n_q_u] = mat_Mrom_uu
        self.mat_Drom = np.zeros((self.n_q_u + self.n_q_t, self.n_q_u + self.n_q_t))
        self.mat_Drom[:self.n_q_u, :self.n_q_u] = mat_Drom_uu
        self.mat_Drom[self.n_q_u:, :self.n_q_u] = mat_Drom_tu
        self.mat_Drom[self.n_q_u:, self.n_q_u:] = mat_Drom_tt
        self.mat_Krom = np.zeros((self.n_q_u + self.n_q_t, self.n_q_u + self.n_q_t))
        self.mat_Krom[:self.n_q_u, :self.n_q_u] = mat_Krom_uu
        self.mat_Krom[:self.n_q_u, self.n_q_u:] = mat_Krom_ut
        self.mat_Krom[self.n_q_u:, self.n_q_u:] = mat_Krom_tt

    def compute_ROM_forces(self):
        # must have already called methods compute_ROB_u and compute_ROB_theta and have global finite element force
        # vector F assembled

        vec_F_fu = self.vec_F[self.free_dofs_U]
        if self.dict_dirichlet_U is not None and self.mat_K is not None:
            for tag, list_u_dir in self.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
                vec_U = np.zeros((self.mesh.n_dofs,))
                dirichlet_dofs_U = []
                for node in dirichlet_nodes_U:
                    for str_dof, val_u in list_u_dir:
                        if str_dof == 'x':
                            modif = 0
                        elif str_dof == 'y':
                            modif = 1
                        elif str_dof == 'z':
                            modif = 2
                        else:
                            raise ValueError('Unrecognized DOF.')
                        dirichlet_dofs_U.append(node * 4 + modif)
                        vec_U[node * 4 + modif] = val_u
                vec_U_d = vec_U[dirichlet_dofs_U]
                mat_K_fu_du = self.mat_K[self.free_dofs_U, :][:, dirichlet_dofs_U]
                vec_F_fu -= mat_K_fu_du @ vec_U_d
        vec_From_u = self.mat_phi_u.transpose() @ vec_F_fu

        vec_F_ft = self.vec_F[self.free_dofs_theta]
        if self.dict_dirichlet_theta is not None and self.mat_K is not None:
            for tag, theta in self.dict_dirichlet_theta.items():
                dirichlet_nodes_theta = self.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_theta = list(set(dirichlet_nodes_theta))
                vec_theta = np.zeros((self.mesh.n_dofs,))
                dirichlet_dofs_theta = []
                for node in dirichlet_nodes_theta:
                    dirichlet_dofs_theta.append(node * 4 + 3)
                    vec_theta[node * 4 + 3] = theta
                vec_theta_d = vec_theta[dirichlet_dofs_theta]
                mat_K_ft_dt = self.mat_K[self.free_dofs_theta, :][:, dirichlet_dofs_theta]
                vec_F_ft -= mat_K_ft_dt @ vec_theta_d
        vec_From_t = self.mat_phi_t.transpose() @ vec_F_ft

        self.vec_From = np.zeros((self.n_q_u + self.n_q_t, ))
        self.vec_From[:self.n_q_u] = vec_From_u
        self.vec_From[self.n_q_u:] = vec_From_t
