import numpy as np
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


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
        self.model.apply_dirichlet()

        vec_X_f = spsolve(self.model.mat_K_f_f, self.model.vec_F_f)
        self.X = np.zeros((self.model.mesh.n_dofs, ))
        self.X[self.model.free_dofs] = vec_X_f
        if self.model.dict_dirichlet_U is not None:
            for tag, vec_u in self.model.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
                for node in dirichlet_nodes_U:
                    self.X[[node * 4, node * 4 + 1, node * 4 + 2]] = vec_u
        if self.model.dict_dirichlet_T is not None:
            for tag, T in self.model.dict_dirichlet_T.items():
                dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                for node in dirichlet_nodes_T:
                    self.X[node * 4 + 3] = T

        self.U = np.zeros((self.model.mesh.n_nodes * 3, ))
        self.U[::3] = self.X[::4]
        self.U[1::3] = self.X[1::4]
        self.U[2::3] = self.X[2::4]
        self.T = self.X[3::4]


class LinearTransient:
    """
    Transient solver (Newmark) for linear thermoelasticity
    """
    def __init__(self, model,
                 initial_U, initial_Udot,
                 initial_T, initial_Tdot,
                 t_end, n_t,
                 gamma=0.5, beta=0.25):
        self.model = model
        self.gamma = gamma
        self.beta = beta
        self.t_end = t_end
        self.n_t = n_t
        self.initial_U = initial_U
        self.initial_Udot = initial_Udot
        self.initial_T = initial_T
        self.initial_Tdot = initial_Tdot

        self.vec_t = np.linspace(0., t_end, n_t)
        self.dt = t_end / (n_t - 1)

        self.X = None
        self.Xdot = None
        self.Xdotdot = None
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
        self.model.assemble_F()
        self.model.apply_dirichlet()

        prev_X = np.zeros((self.model.mesh.n_dofs, ))
        prev_Xdot = np.zeros((self.model.mesh.n_dofs, ))
        prev_Xdotdot = np.zeros((self.model.mesh.n_dofs, ))

        prev_X[::4] = self.initial_U[::3]
        prev_X[1::4] = self.initial_U[1::3]
        prev_X[2::4] = self.initial_U[2::3]
        prev_X[3::4] = self.initial_T

        prev_Xdot[::4] = self.initial_Udot[::3]
        prev_Xdot[1::4] = self.initial_Udot[1::3]
        prev_Xdot[2::4] = self.initial_Udot[2::3]
        prev_Xdot[3::4] = self.initial_Tdot

        self.X = np.zeros((self.model.mesh.n_dofs, self.n_t))
        self.Xdot = np.zeros((self.model.mesh.n_dofs, self.n_t))
        self.Xdotdot = np.zeros((self.model.mesh.n_dofs, self.n_t))

        self.X[:, 0] = prev_X
        self.Xdot[:, 0] = prev_Xdot
        # initial acceleration is assumed to be zero

        if self.model.dict_dirichlet_U is not None:
            for tag, vec_u in self.model.dict_dirichlet_U.items():
                dirichlet_nodes_U = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_U = list(set(dirichlet_nodes_U))
                for node in dirichlet_nodes_U:
                    self.X[[node * 4, node * 4 + 1, node * 4 + 2], :] = np.tile(vec_u[:, np.newaxis], (1, self.n_t))
        if self.model.dict_dirichlet_T is not None:
            for tag, T in self.model.dict_dirichlet_T.items():
                dirichlet_nodes_T = self.model.mesh.dict_tri_groups[tag].flatten()
                dirichlet_nodes_T = list(set(dirichlet_nodes_T))
                for node in dirichlet_nodes_T:
                    self.X[node * 4 + 3, :] = T

        prev_X_f = prev_X[self.model.free_dofs]
        prev_Xdot_f = prev_Xdot[self.model.free_dofs]
        prev_Xdotdot_f = prev_Xdotdot[self.model.free_dofs]

        mat_Kdyn_f_f = (self.model.mat_M_f_f
                        + self.gamma * self.dt * self.model.mat_D_f_f
                        + self.beta * (self.dt ** 2) * self.model.mat_K_f_f)

        for i in tqdm(range(1, self.n_t)):
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
        self.T = np.zeros((self.model.mesh.n_nodes, self.n_t))
        self.Tdot = np.zeros((self.model.mesh.n_nodes, self.n_t))
        self.Tdotdot = np.zeros((self.model.mesh.n_nodes, self.n_t))

        self.U[::3, :] = self.X[::4, :]
        self.U[1::3, :] = self.X[1::4, :]
        self.U[2::3, :] = self.X[2::4, :]

        self.Udot[::3, :] = self.Xdot[::4, :]
        self.Udot[1::3, :] = self.Xdot[1::4, :]
        self.Udot[2::3, :] = self.Xdot[2::4, :]

        self.Udotdot[::3, :] = self.Xdotdot[::4, :]
        self.Udotdot[1::3, :] = self.Xdotdot[1::4, :]
        self.Udotdot[2::3, :] = self.Xdotdot[2::4, :]

        self.T = self.X[3::4, :]
        self.Tdot = self.Xdot[3::4, :]
        self.Tdotdot = self.Xdotdot[3::4, :]
