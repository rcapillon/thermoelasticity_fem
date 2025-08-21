import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np

from thermoelasticity_fem.materials import LinearThermoElastic
from thermoelasticity_fem.mesh import Mesh
from thermoelasticity_fem.model import Model
from thermoelasticity_fem.solvers import LinearTransient
from thermoelasticity_fem.plots import animate_U_T


if __name__ == '__main__':
    mesh = Mesh()
    mesh.load_mesh('../data/sandwich.msh')

    # fictive material
    material = LinearThermoElastic(rho=2300, Y=64e9, nu=0.1, k=1e3, c=500e-3, alpha=4e-6, T0=20.)
    dict_materials = {
        1: material,
        2: material,
        3: material
    }
    mesh.set_materials(dict_materials)
    mesh.make_elements()

    vec_u_dir1 = np.array([0., 0., 0.])
    vec_u_dir2 = np.array([0., 0., 0.])
    dict_dirichlet_U = {
        4: vec_u_dir1,
        5: vec_u_dir2
    }

    ambient_temperature = 20.

    T_dir1 = ambient_temperature
    T_dir2 = ambient_temperature
    # T_dir3 = 40.
    dict_dirichlet_T = {
        4: T_dir1,
        5: T_dir2,
        # 7: T_dir3
    }
    # dict_dirichlet_T = None

    vec_f_surf = np.array([0., 0., 1e7])
    dict_surface_forces = {
        7: vec_f_surf
    }
    # dict_surface_forces = None

    q = -1e4
    dict_heat_flux = {
        6: q,
    }
    # dict_heat_flux = None

    # rho = material.rho
    # R = 1e3
    # dict_heat_source = {
    #     2: rho * R
    # }
    dict_heat_source = None

    alpha_M = 2e-1
    alpha_K = 2e-1

    model = Model(mesh,
                  dict_dirichlet_U=dict_dirichlet_U, dict_dirichlet_T=dict_dirichlet_T,
                  dict_surface_forces=dict_surface_forces,
                  dict_heat_flux=dict_heat_flux, dict_heat_source=dict_heat_source,
                  alpha_M=alpha_M, alpha_K=alpha_K)

    t_end = 1e0
    n_t = int(1e2)
    gamma = 1/2
    beta = 1/4
    initial_U = np.zeros((model.mesh.n_nodes * 3, ))
    initial_Udot = np.zeros((model.mesh.n_nodes * 3, ))
    initial_T = ambient_temperature * np.ones((model.mesh.n_nodes, ))
    initial_Tdot = np.zeros((model.mesh.n_nodes, ))

    solver = LinearTransient(model, initial_U, initial_Udot, initial_T, initial_Tdot, t_end, n_t, gamma, beta)
    solver.solve()

    ####
    # Animation (deformed mesh, temperature as color)
    amplification_factor_U = 1e2
    save_path = './test_animation.mp4'
    animate_U_T(solver.model.mesh, solver.T, solver.U, solver.vec_t, save_path,
                amplification_factor_U=amplification_factor_U,
                fps=25, quality=5)
