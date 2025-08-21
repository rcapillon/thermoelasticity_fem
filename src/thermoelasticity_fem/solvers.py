import numpy as np
from scipy.sparse.linalg import spsolve


class LinearSteadyState:
    """
    Steady-state solver for linear thermoelasticity
    """
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

        self.displacement = np.zeros((self.model.mesh.n_nodes * 3, ))
        self.displacement[::3] = self.X[::4]
        self.displacement[1::3] = self.X[1::4]
        self.displacement[2::3] = self.X[2::4]
        self.temperature = self.X[3::4]


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

        self.model.create_free_dofs_lists()
        self.model.assemble_M()
        self.model.assemble_K()
        self.model.assemble_D()
        self.model.assemble_F()
        self.model.apply_dirichlet()

        mat_Kdyn_f_f = (self.model.mat_M_f_f
                        + self.gamma * self.dt * self.model.mat_D_f_f
                        + self.beta * (self.dt ** 2) * self.model.mat_K_f_f)
