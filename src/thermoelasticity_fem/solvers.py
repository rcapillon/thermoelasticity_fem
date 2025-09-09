import numpy as np
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from random_matrices.generators import SE_0_plus, SE_plus0, SE_rect


class LinearSteadyState:
    """
    Steady-state solver for linear thermoelasticity
    """
    def __init__(self, model):
        self.model = model

        self.X = None
        self.U = None
        self.T = None

    def solve(self):
        self.model.create_free_dofs_lists()
        self.model.assemble_K()
        self.model.assemble_F()
        self.model.apply_dirichlet_matrices()
        self.model.apply_dirichlet_F()

        vec_X_f = spsolve(self.model.mat_K_f_f, self.model.vec_F_f)
        self.X = np.zeros((self.model.mesh.n_dofs, ))
        self.X[self.model.free_dofs] = vec_X_f
        if self.model.dict_dirichlet_U is not None:
            for tag, list_u_dir in self.model.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
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
                        self.X[node * 4 + modif] = val_u
        if self.model.dict_dirichlet_theta is not None:
            for tag, theta in self.model.dict_dirichlet_theta.items():
                dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                for node in dirichlet_nodes_T:
                    self.X[node * 4 + 3] = theta

        reference_temperature = np.zeros((self.model.mesh.n_nodes, ))
        for i in range(self.model.mesh.n_nodes):
            n_count = 0
            for tag, material in self.model.mesh.dict_materials.items():
                table_tet = self.model.mesh.dict_tet_groups[tag]
                list_nodes = list(set(list(table_tet.flatten())))
                if i in list_nodes:
                    n_count += 1
                    reference_temperature[i] += material.T0
            reference_temperature[i] /= n_count

        self.U = np.zeros((self.model.mesh.n_nodes * 3, ))
        self.U[::3] = self.X[::4]
        self.U[1::3] = self.X[1::4]
        self.U[2::3] = self.X[2::4]
        self.T = self.X[3::4] + reference_temperature


class LinearTransient:
    """
    Transient solver (Newmark) for linear thermoelasticity
    """
    def __init__(self, model,
                 initial_U, initial_Udot,
                 initial_theta, initial_thetadot,
                 t_end, n_t,
                 gamma=0.5, beta=0.25):
        self.model = model
        self.gamma = gamma
        self.beta = beta
        self.t_end = t_end
        self.n_t = n_t
        self.initial_U = initial_U
        self.initial_Udot = initial_Udot
        self.initial_theta = initial_theta
        self.initial_thetadot = initial_thetadot

        self.vec_t = np.linspace(0., t_end, n_t)
        self.dt = t_end / (n_t - 1)

        self.X = None
        self.Xdot = None
        self.Xdotdot = None
        self.q = None
        self.qdot = None
        self.qdotdot = None
        self.U = None
        self.Udot = None
        self.Udotdot = None
        self.T = None
        self.Tdot = None
        self.Tdotdot = None

    def solve(self):
        self.model.create_free_dofs_lists()
        self.model.assemble_M()
        self.model.assemble_K()
        self.model.assemble_D()
        self.model.apply_dirichlet_matrices()

        prev_X = np.zeros((self.model.mesh.n_dofs, ))
        prev_Xdot = np.zeros((self.model.mesh.n_dofs, ))
        prev_Xdotdot = np.zeros((self.model.mesh.n_dofs, ))

        prev_X[::4] = self.initial_U[::3]
        prev_X[1::4] = self.initial_U[1::3]
        prev_X[2::4] = self.initial_U[2::3]
        prev_X[3::4] = self.initial_theta

        prev_Xdot[::4] = self.initial_Udot[::3]
        prev_Xdot[1::4] = self.initial_Udot[1::3]
        prev_Xdot[2::4] = self.initial_Udot[2::3]
        prev_Xdot[3::4] = self.initial_thetadot

        self.X = np.zeros((self.model.mesh.n_dofs, self.n_t))
        self.Xdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
        self.Xdotdot = np.zeros((self.model.mesh.n_dofs, self.n_t))

        self.X[:, 0] = prev_X
        self.Xdot[:, 0] = prev_Xdot
        # initial acceleration is assumed to be zero

        if self.model.dict_dirichlet_U is not None:
            for tag, list_u_dir in self.model.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
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
                        self.X[node * 4 + modif, :] = val_u
        if self.model.dict_dirichlet_theta is not None:
            for tag, theta in self.model.dict_dirichlet_theta.items():
                dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                for node in dirichlet_nodes_T:
                    self.X[node * 4 + 3, :] = theta

        prev_X_f = prev_X[self.model.free_dofs]
        prev_Xdot_f = prev_Xdot[self.model.free_dofs]
        prev_Xdotdot_f = prev_Xdotdot[self.model.free_dofs]

        mat_Kdyn_f_f = (self.model.mat_M_f_f
                        + self.gamma * self.dt * self.model.mat_D_f_f
                        + self.beta * (self.dt ** 2) * self.model.mat_K_f_f)

        for i in tqdm(range(1, self.n_t)):
            self.model.assemble_F(idx_t=i)
            self.model.apply_dirichlet_F()

            rhs = (self.model.vec_F_f
                   - self.model.mat_D_f_f @ (prev_Xdot_f + (1 - self.gamma) * self.dt * prev_Xdotdot_f)
                   - self.model.mat_K_f_f @ (prev_X_f + self.dt * prev_Xdot_f
                                             + 0.5 * (self.dt ** 2) * (1 - 2 * self.beta) * prev_Xdotdot_f))

            new_Xdotdot_f = spsolve(mat_Kdyn_f_f, rhs)
            new_Xdot_f = (prev_Xdot_f
                          + (1 - self.gamma) * self.dt * prev_Xdotdot_f
                          + self.gamma * self.dt * new_Xdotdot_f)
            new_X_f = (prev_X_f
                       + self.dt * prev_Xdot_f
                       + 0.5 * (self.dt ** 2) * ((1 - 2 * self.beta) * prev_Xdotdot_f + 2 * self.beta * new_Xdotdot_f))

            self.X[self.model.free_dofs, i] = new_X_f
            self.Xdot[self.model.free_dofs, i] = new_Xdot_f
            self.Xdotdot[self.model.free_dofs, i] = new_Xdotdot_f

            prev_X_f = new_X_f
            prev_Xdot_f = new_Xdot_f
            prev_Xdotdot_f = new_Xdotdot_f

        self.U = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
        self.Udot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
        self.Udotdot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))

        self.U[::3, :] = self.X[::4, :]
        self.U[1::3, :] = self.X[1::4, :]
        self.U[2::3, :] = self.X[2::4, :]

        self.Udot[::3, :] = self.Xdot[::4, :]
        self.Udot[1::3, :] = self.Xdot[1::4, :]
        self.Udot[2::3, :] = self.Xdot[2::4, :]

        self.Udotdot[::3, :] = self.Xdotdot[::4, :]
        self.Udotdot[1::3, :] = self.Xdotdot[1::4, :]
        self.Udotdot[2::3, :] = self.Xdotdot[2::4, :]

        reference_temperature = np.zeros((self.model.mesh.n_nodes,))
        for i in range(self.model.mesh.n_nodes):
            n_count = 0
            for tag, material in self.model.mesh.dict_materials.items():
                table_tet = self.model.mesh.dict_tet_groups[tag]
                list_nodes = list(set(list(table_tet.flatten())))
                if i in list_nodes:
                    n_count += 1
                    reference_temperature[i] += material.T0
            reference_temperature[i] /= n_count

        self.T = self.X[3::4, :] + np.tile(reference_temperature[:, np.newaxis], (1, self.n_t))
        self.Tdot = self.Xdot[3::4, :]
        self.Tdotdot = self.Xdotdot[3::4, :]

    def solve_ROM(self, n_modes_u, n_modes_theta):
        self.model.create_free_dofs_lists()
        self.model.assemble_M()
        self.model.assemble_K()
        self.model.assemble_D()
        print('Computing reduced order bases...')
        self.model.compute_ROB_u(n_modes_u)
        self.model.compute_ROB_theta(n_modes_theta)
        print('Computing reduced matrices...')
        self.model.compute_ROM_matrices()

        print('Solving...')

        prev_q = np.zeros((self.model.n_q_u + self.model.n_q_t, ))
        prev_qdot = np.zeros((self.model.n_q_u + self.model.n_q_t, ))
        prev_qdotdot = np.zeros((self.model.n_q_u + self.model.n_q_t, ))

        initial_X = np.zeros((self.model.mesh.n_dofs, ))
        initial_X[::4] = self.initial_U[::3]
        initial_X[1::4] = self.initial_U[::3]
        initial_X[2::4] = self.initial_U[::3]
        initial_X[3::4] = self.initial_theta

        prev_q[:self.model.n_q_u] = self.model.mat_phi_u.transpose() @ initial_X[self.model.free_dofs_U]
        prev_q[self.model.n_q_u:] = self.model.mat_phi_t.transpose() @ initial_X[self.model.free_dofs_theta]

        initial_Xdot = np.zeros((self.model.mesh.n_dofs,))
        initial_Xdot[::4] = self.initial_Udot[::3]
        initial_Xdot[1::4] = self.initial_Udot[::3]
        initial_Xdot[2::4] = self.initial_Udot[::3]
        initial_Xdot[3::4] = self.initial_thetadot

        prev_qdot[:self.model.n_q_u] = self.model.mat_phi_u.transpose() @ initial_Xdot[self.model.free_dofs_U]
        prev_qdot[self.model.n_q_u:] = self.model.mat_phi_t.transpose() @ initial_Xdot[self.model.free_dofs_theta]

        self.q = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))
        self.qdot = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))
        self.qdotdot = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))

        self.q[:, 0] = prev_q
        self.qdot[:, 0] = prev_qdot
        # initial acceleration is assumed to be zero

        mat_Kdyn_rom = (self.model.mat_Mrom
                        + self.gamma * self.dt * self.model.mat_Drom
                        + self.beta * (self.dt ** 2) * self.model.mat_Krom)

        for i in tqdm(range(1, self.n_t)):
            self.model.assemble_F(idx_t=i)
            self.model.compute_ROM_forces()

            rhs = (self.model.vec_From
                   - self.model.mat_Drom @ (prev_qdot + (1 - self.gamma) * self.dt * prev_qdotdot)
                   - self.model.mat_Krom @ (prev_q + self.dt * prev_qdot
                                             + 0.5 * (self.dt ** 2) * (1 - 2 * self.beta) * prev_qdotdot))

            new_qdotdot = np.linalg.solve(mat_Kdyn_rom, rhs)
            new_qdot = (prev_qdot
                        + (1 - self.gamma) * self.dt * prev_qdotdot
                        + self.gamma * self.dt * new_qdotdot)
            new_q = (prev_q
                     + self.dt * prev_qdot
                     + 0.5 * (self.dt ** 2) * ((1 - 2 * self.beta) * prev_qdotdot + 2 * self.beta * new_qdotdot))

            self.q[:, i] = new_q
            self.qdot[:, i] = new_qdot
            self.qdotdot[:, i] = new_qdotdot

            prev_q = new_q
            prev_qdot = new_qdot
            prev_qdotdot = new_qdotdot

        self.U = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
        self.Udot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
        self.Udotdot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))

        X = np.zeros((self.model.mesh.n_dofs, self.n_t))
        X[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.q[:self.model.n_q_u, :]
        X[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.q[self.model.n_q_u:, :]
        Xdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
        Xdot[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.qdot[:self.model.n_q_u, :]
        Xdot[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.qdot[self.model.n_q_u:, :]
        Xdotdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
        Xdotdot[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.qdotdot[:self.model.n_q_u, :]
        Xdotdot[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.qdotdot[self.model.n_q_u:, :]

        self.U[::3, :] = X[::4, :]
        self.U[1::3, :] = X[1::4, :]
        self.U[2::3, :] = X[2::4, :]

        self.Udot[::3, :] = Xdot[::4, :]
        self.Udot[1::3, :] = Xdot[1::4, :]
        self.Udot[2::3, :] = Xdot[2::4, :]

        self.Udotdot[::3, :] = Xdotdot[::4, :]
        self.Udotdot[1::3, :] = Xdotdot[1::4, :]
        self.Udotdot[2::3, :] = Xdotdot[2::4, :]

        self.T = X[3::4, :]
        self.Tdot = Xdot[3::4, :]
        self.Tdotdot = Xdotdot[3::4, :]

        if self.model.dict_dirichlet_U is not None:
            for tag, list_u_dir in self.model.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
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
                        self.U[node * 3 + modif, :] = val_u
        if self.model.dict_dirichlet_theta is not None:
            for tag, theta in self.model.dict_dirichlet_theta.items():
                dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                for node in dirichlet_nodes_T:
                    self.T[node, :] = theta

        reference_temperature = np.zeros((self.model.mesh.n_nodes,))
        for i in range(self.model.mesh.n_nodes):
            n_count = 0
            for tag, material in self.model.mesh.dict_materials.items():
                table_tet = self.model.mesh.dict_tet_groups[tag]
                list_nodes = list(set(list(table_tet.flatten())))
                if i in list_nodes:
                    n_count += 1
                    reference_temperature[i] += material.T0
            reference_temperature[i] /= n_count

        self.T += np.tile(reference_temperature[:, np.newaxis], (1, self.n_t))

    def solve_ROM_nonparametric(self, n_modes_u, n_modes_theta, n_samples, observed_dofs_U, observed_nodes_T):
        self.model.create_free_dofs_lists()
        self.model.assemble_M()
        self.model.assemble_K()
        self.model.assemble_D()
        print('Computing reduced order bases...')
        self.model.compute_ROB_u(n_modes_u)
        self.model.compute_ROB_theta(n_modes_theta)
        print('Computing reduced mean matrices...')
        self.model.compute_ROM_matrices()

        print("Computing mean matrices")
        mean_M_uu = self.model.mat_Mrom[:n_modes_u, :n_modes_u]
        mean_D_uu = self.model.mat_Drom[:n_modes_u, :n_modes_u]
        mean_D_tu = self.model.mat_Drom[n_modes_u:, :n_modes_u]
        mean_D_tt = self.model.mat_Drom[n_modes_u:, n_modes_u:]
        mean_K_uu = self.model.mat_Krom[:n_modes_u, :n_modes_u]
        mean_K_ut = self.model.mat_Krom[:n_modes_u, n_modes_u:]
        mean_K_tt = self.model.mat_Krom[n_modes_u:, n_modes_u:]

        reference_temperature = np.zeros((self.model.mesh.n_nodes,))
        for i in range(self.model.mesh.n_nodes):
            n_count = 0
            for tag, material in self.model.mesh.dict_materials.items():
                table_tet = self.model.mesh.dict_tet_groups[tag]
                list_nodes = list(set(list(table_tet.flatten())))
                if i in list_nodes:
                    n_count += 1
                    reference_temperature[i] += material.T0
            reference_temperature[i] /= n_count

        print('Solving deterministic model...')

        # Deterministic model
        prev_q = np.zeros((self.model.n_q_u + self.model.n_q_t,))
        prev_qdot = np.zeros((self.model.n_q_u + self.model.n_q_t,))
        prev_qdotdot = np.zeros((self.model.n_q_u + self.model.n_q_t,))

        initial_X = np.zeros((self.model.mesh.n_dofs,))
        initial_X[::4] = self.initial_U[::3]
        initial_X[1::4] = self.initial_U[::3]
        initial_X[2::4] = self.initial_U[::3]
        initial_X[3::4] = self.initial_theta

        prev_q[:self.model.n_q_u] = self.model.mat_phi_u.transpose() @ initial_X[self.model.free_dofs_U]
        prev_q[self.model.n_q_u:] = self.model.mat_phi_t.transpose() @ initial_X[self.model.free_dofs_theta]

        initial_Xdot = np.zeros((self.model.mesh.n_dofs,))
        initial_Xdot[::4] = self.initial_Udot[::3]
        initial_Xdot[1::4] = self.initial_Udot[::3]
        initial_Xdot[2::4] = self.initial_Udot[::3]
        initial_Xdot[3::4] = self.initial_thetadot

        prev_qdot[:self.model.n_q_u] = self.model.mat_phi_u.transpose() @ initial_Xdot[self.model.free_dofs_U]
        prev_qdot[self.model.n_q_u:] = self.model.mat_phi_t.transpose() @ initial_Xdot[self.model.free_dofs_theta]

        self.q = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))
        self.qdot = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))
        self.qdotdot = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))

        self.q[:, 0] = prev_q
        self.qdot[:, 0] = prev_qdot
        # initial acceleration is assumed to be zero

        mat_Kdyn_rom = (self.model.mat_Mrom
                        + self.gamma * self.dt * self.model.mat_Drom
                        + self.beta * (self.dt ** 2) * self.model.mat_Krom)

        for i in tqdm(range(1, self.n_t)):
            self.model.assemble_F(idx_t=i)
            self.model.compute_ROM_forces()

            rhs = (self.model.vec_From
                   - self.model.mat_Drom @ (prev_qdot + (1 - self.gamma) * self.dt * prev_qdotdot)
                   - self.model.mat_Krom @ (prev_q + self.dt * prev_qdot
                                            + 0.5 * (self.dt ** 2) * (1 - 2 * self.beta) * prev_qdotdot))

            new_qdotdot = np.linalg.solve(mat_Kdyn_rom, rhs)
            new_qdot = (prev_qdot
                        + (1 - self.gamma) * self.dt * prev_qdotdot
                        + self.gamma * self.dt * new_qdotdot)
            new_q = (prev_q
                     + self.dt * prev_qdot
                     + 0.5 * (self.dt ** 2) * ((1 - 2 * self.beta) * prev_qdotdot + 2 * self.beta * new_qdotdot))

            self.q[:, i] = new_q
            self.qdot[:, i] = new_qdot
            self.qdotdot[:, i] = new_qdotdot

            prev_q = new_q
            prev_qdot = new_qdot
            prev_qdotdot = new_qdotdot

        U = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
        Udot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
        Udotdot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))

        X = np.zeros((self.model.mesh.n_dofs, self.n_t))
        X[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.q[:self.model.n_q_u, :]
        X[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.q[self.model.n_q_u:, :]
        Xdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
        Xdot[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.qdot[:self.model.n_q_u, :]
        Xdot[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.qdot[self.model.n_q_u:, :]
        Xdotdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
        Xdotdot[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.qdotdot[:self.model.n_q_u, :]
        Xdotdot[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.qdotdot[self.model.n_q_u:, :]

        U[::3, :] = X[::4, :]
        U[1::3, :] = X[1::4, :]
        U[2::3, :] = X[2::4, :]

        Udot[::3, :] = Xdot[::4, :]
        Udot[1::3, :] = Xdot[1::4, :]
        Udot[2::3, :] = Xdot[2::4, :]

        Udotdot[::3, :] = Xdotdot[::4, :]
        Udotdot[1::3, :] = Xdotdot[1::4, :]
        Udotdot[2::3, :] = Xdotdot[2::4, :]

        T = X[3::4, :]
        Tdot = Xdot[3::4, :]
        Tdotdot = Xdotdot[3::4, :]

        if self.model.dict_dirichlet_U is not None:
            for tag, list_u_dir in self.model.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
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
                        U[node * 3 + modif, :] = val_u
        if self.model.dict_dirichlet_theta is not None:
            for tag, theta in self.model.dict_dirichlet_theta.items():
                dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                for node in dirichlet_nodes_T:
                    T[node, :] = theta

        T += np.tile(reference_temperature[:, np.newaxis], (1, self.n_t))

        deterministic_U = U[observed_dofs_U, :]
        deterministic_Udot = Udot[observed_dofs_U, :]
        deterministic_Udotdot = Udotdot[observed_dofs_U, :]
        deterministic_T = T[observed_nodes_T, :]
        deterministic_Tdot = Tdot[observed_nodes_T, :]
        deterministic_Tdotdot = Tdotdot[observed_nodes_T, :]

        # Stochastic model
        print('Solving stochastic model...')

        random_U = np.zeros((len(observed_dofs_U), self.n_t, n_samples))
        random_Udot = np.zeros((len(observed_dofs_U), self.n_t, n_samples))
        random_Udotdot = np.zeros((len(observed_dofs_U), self.n_t, n_samples))
        random_T = np.zeros((len(observed_nodes_T), self.n_t, n_samples))
        random_Tdot = np.zeros((len(observed_nodes_T), self.n_t, n_samples))
        random_Tdotdot = np.zeros((len(observed_nodes_T), self.n_t, n_samples))

        for j in tqdm(range(n_samples)):
            dispersion_coeff = 0.2
            random_M_uu = SE_0_plus(dispersion_coeff, mean_M_uu, 1)[:, :, 0]
            random_D_uu = SE_plus0(dispersion_coeff, mean_D_uu, 1, eps=1e-6, tol=1e-9)[:, :, 0]
            random_D_tu = SE_rect(dispersion_coeff, mean_D_tu, 1, eps=1e-6)[:, :, 0]
            random_D_tt = SE_0_plus(dispersion_coeff, mean_D_tt, 1)[:, :, 0]
            random_K_uu = SE_0_plus(dispersion_coeff, mean_K_uu, 1)[:, :, 0]
            random_K_ut = SE_rect(dispersion_coeff, mean_K_ut, 1, eps=1e-6)[:, :, 0]
            random_K_tt = SE_0_plus(dispersion_coeff, mean_K_tt, 1)[:, :, 0]

            random_Mrom = np.zeros((n_modes_u + n_modes_theta, n_modes_u + n_modes_theta))
            random_Drom = np.zeros((n_modes_u + n_modes_theta, n_modes_u + n_modes_theta))
            random_Krom = np.zeros((n_modes_u + n_modes_theta, n_modes_u + n_modes_theta))

            random_Mrom[:n_modes_u, :n_modes_u] = random_M_uu
            random_Drom[:n_modes_u, :n_modes_u] = random_D_uu
            random_Drom[n_modes_u:, :n_modes_u] = random_D_tu
            random_Drom[n_modes_u:, n_modes_u:] = random_D_tt
            random_Krom[:n_modes_u, :n_modes_u] = random_K_uu
            random_Krom[:n_modes_u, n_modes_u:] = random_K_ut
            random_Krom[n_modes_u:, n_modes_u:] = random_K_tt

            self.model.mat_Mrom = random_Mrom
            self.model.mat_Drom = random_Drom
            self.model.mat_Krom = random_Krom

            prev_q = np.zeros((self.model.n_q_u + self.model.n_q_t,))
            prev_qdot = np.zeros((self.model.n_q_u + self.model.n_q_t,))
            prev_qdotdot = np.zeros((self.model.n_q_u + self.model.n_q_t,))

            initial_X = np.zeros((self.model.mesh.n_dofs,))
            initial_X[::4] = self.initial_U[::3]
            initial_X[1::4] = self.initial_U[::3]
            initial_X[2::4] = self.initial_U[::3]
            initial_X[3::4] = self.initial_theta

            prev_q[:self.model.n_q_u] = self.model.mat_phi_u.transpose() @ initial_X[self.model.free_dofs_U]
            prev_q[self.model.n_q_u:] = self.model.mat_phi_t.transpose() @ initial_X[self.model.free_dofs_theta]

            initial_Xdot = np.zeros((self.model.mesh.n_dofs,))
            initial_Xdot[::4] = self.initial_Udot[::3]
            initial_Xdot[1::4] = self.initial_Udot[::3]
            initial_Xdot[2::4] = self.initial_Udot[::3]
            initial_Xdot[3::4] = self.initial_thetadot

            prev_qdot[:self.model.n_q_u] = self.model.mat_phi_u.transpose() @ initial_Xdot[self.model.free_dofs_U]
            prev_qdot[self.model.n_q_u:] = self.model.mat_phi_t.transpose() @ initial_Xdot[self.model.free_dofs_theta]

            self.q = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))
            self.qdot = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))
            self.qdotdot = np.zeros((self.model.n_q_u + self.model.n_q_t, self.n_t))

            self.q[:, 0] = prev_q
            self.qdot[:, 0] = prev_qdot
            # initial acceleration is assumed to be zero

            mat_Kdyn_rom = (self.model.mat_Mrom
                            + self.gamma * self.dt * self.model.mat_Drom
                            + self.beta * (self.dt ** 2) * self.model.mat_Krom)

            for i in range(1, self.n_t):
                self.model.assemble_F(idx_t=i)
                self.model.compute_ROM_forces()

                rhs = (self.model.vec_From
                       - self.model.mat_Drom @ (prev_qdot + (1 - self.gamma) * self.dt * prev_qdotdot)
                       - self.model.mat_Krom @ (prev_q + self.dt * prev_qdot
                                                + 0.5 * (self.dt ** 2) * (1 - 2 * self.beta) * prev_qdotdot))

                new_qdotdot = np.linalg.solve(mat_Kdyn_rom, rhs)
                new_qdot = (prev_qdot
                            + (1 - self.gamma) * self.dt * prev_qdotdot
                            + self.gamma * self.dt * new_qdotdot)
                new_q = (prev_q
                         + self.dt * prev_qdot
                         + 0.5 * (self.dt ** 2) * ((1 - 2 * self.beta) * prev_qdotdot + 2 * self.beta * new_qdotdot))

                self.q[:, i] = new_q
                self.qdot[:, i] = new_qdot
                self.qdotdot[:, i] = new_qdotdot

                prev_q = new_q
                prev_qdot = new_qdot
                prev_qdotdot = new_qdotdot

            U = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
            Udot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))
            Udotdot = np.zeros((self.model.mesh.n_nodes * 3, self.n_t))

            X = np.zeros((self.model.mesh.n_dofs, self.n_t))
            X[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.q[:self.model.n_q_u, :]
            X[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.q[self.model.n_q_u:, :]
            Xdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
            Xdot[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.qdot[:self.model.n_q_u, :]
            Xdot[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.qdot[self.model.n_q_u:, :]
            Xdotdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
            Xdotdot[self.model.free_dofs_U, :] = self.model.mat_phi_u @ self.qdotdot[:self.model.n_q_u, :]
            Xdotdot[self.model.free_dofs_theta, :] = self.model.mat_phi_t @ self.qdotdot[self.model.n_q_u:, :]

            U[::3, :] = X[::4, :]
            U[1::3, :] = X[1::4, :]
            U[2::3, :] = X[2::4, :]

            Udot[::3, :] = Xdot[::4, :]
            Udot[1::3, :] = Xdot[1::4, :]
            Udot[2::3, :] = Xdot[2::4, :]

            Udotdot[::3, :] = Xdotdot[::4, :]
            Udotdot[1::3, :] = Xdotdot[1::4, :]
            Udotdot[2::3, :] = Xdotdot[2::4, :]

            T = X[3::4, :]
            Tdot = Xdot[3::4, :]
            Tdotdot = Xdotdot[3::4, :]

            if self.model.dict_dirichlet_U is not None:
                for tag, list_u_dir in self.model.dict_dirichlet_U.items():
                    dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                    dirichlet_nodes_U = list(set(dirichlet_nodes_U))
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
                            U[node * 3 + modif, :] = val_u
            if self.model.dict_dirichlet_theta is not None:
                for tag, theta in self.model.dict_dirichlet_theta.items():
                    dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                    dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                    for node in dirichlet_nodes_T:
                        T[node, :] = theta

            T += np.tile(reference_temperature[:, np.newaxis], (1, self.n_t))

            random_U[:, :, j] = U[observed_dofs_U, :]
            random_Udot[:, :, j] = Udot[observed_dofs_U, :]
            random_Udotdot[:, :, j] = Udotdot[observed_dofs_U, :]
            random_T[:, :, j] = T[observed_nodes_T, :]
            random_Tdot[:, :, j] = Tdot[observed_nodes_T, :]
            random_Tdotdot[:, :, j] = Tdotdot[observed_nodes_T, :]

        return (deterministic_U, deterministic_Udot, deterministic_Udotdot,
                deterministic_T, deterministic_Tdot, deterministic_Tdotdot,
                random_U, random_Udot, random_Udotdot,
                random_T, random_Tdot, random_Tdotdot)