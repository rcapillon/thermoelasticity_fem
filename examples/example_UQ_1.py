import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import time

from thermoelasticity_fem.materials import LinearThermoElastic
from thermoelasticity_fem.mesh import Mesh
from thermoelasticity_fem.model import Model
from thermoelasticity_fem.solvers import LinearTransient
from thermoelasticity_fem.plots import plot_dofU_vs_t_random, plot_nodeT_vs_t_random


if __name__ == '__main__':
    t0 = time.time()

    # Time interval and number of timesteps, defined here to make time-varying loads
    t_end = 1800e0
    n_t = int(1e3)
    vec_t = np.linspace(0., t_end, n_t)

    # Load mesh
    mesh = Mesh()
    mesh.load_mesh('../data/sandwich.msh')

    # fictive material
    material = LinearThermoElastic(rho=2300, Y=64e9, nu=0.1, k=1., c=1., alpha=4e-6, T0=20.)
    dict_materials = {
        1: material,
        2: material,
        3: material
    }
    mesh.set_materials(dict_materials)
    mesh.make_elements()

    dict_dirichlet_U = {
        4: [('x', 0.), ('y', 0.), ('z', 0.)],
        5: [('y', 0.), ('z', 0.)]
    }

    theta_dir1 = 0.
    theta_dir2 = 0.
    # T_dir3 = 40.
    dict_dirichlet_theta = {
        4: theta_dir1,
        5: theta_dir2,
        # 7: T_dir3
    }
    # dict_dirichlet_T = None

    amplitude_fx = -1e9
    arr_f_surf = np.zeros((3, n_t))
    arr_f_surf[0, :] = np.sin(2 * np.pi * vec_t / 900.) * amplitude_fx
    # vec_f_surf = np.array([amplitude_fx, 0., 0.])
    dict_surface_forces = {
        5: arr_f_surf
    }
    # dict_surface_forces = None

    # vec_f_vol = np.array([0., 0., -material.rho * 9.81])
    # dict_volume_forces = {
    #     1: vec_f_vol,
    #     2: vec_f_vol,
    #     3: vec_f_vol
    # }
    dict_volume_forces = None

    # q = -1e4
    # dict_heat_flux = {
    #     6: q,
    # }
    dict_heat_flux = None

    # rho = material.rho
    # R = 1e3
    # dict_heat_source = {
    #     2: rho * R
    # }
    dict_heat_source = None

    alpha_M = 2e-1
    alpha_K = 2e-1

    model = Model(mesh,
                  dict_dirichlet_U=dict_dirichlet_U, dict_dirichlet_theta=dict_dirichlet_theta,
                  dict_surface_forces=dict_surface_forces, dict_volume_forces=dict_volume_forces,
                  dict_heat_flux=dict_heat_flux, dict_heat_source=dict_heat_source,
                  alpha_M=alpha_M, alpha_K=alpha_K)

    gamma = 1/2
    beta = 1/4
    initial_U = np.zeros((model.mesh.n_nodes * 3, ))
    initial_Udot = np.zeros((model.mesh.n_nodes * 3, ))
    initial_theta = np.zeros((model.mesh.n_nodes, ))
    initial_thetadot = np.zeros((model.mesh.n_nodes, ))

    n_modes_u = 30
    n_modes_theta = 30

    n_samples = 50
    dict_dispersion_coeff = {'M_uu': 0.3, 'D_uu': 0.3, 'D_tu': 0.3, 'D_tt': 0.3, 'K_uu': 0.3, 'K_ut': 0.3, 'K_tt': 0.3}
    node_num = model.mesh.dict_nodes_groups[8]
    Udof_num = node_num * 3

    solver = LinearTransient(model, initial_U, initial_Udot, initial_theta, initial_thetadot, t_end, n_t, gamma, beta)
    (deterministic_U, deterministic_Udot, deterministic_Udotdot,
     deterministic_T, deterministic_Tdot, deterministic_Tdotdot,
     random_U, random_Udot, random_Udotdot,
     random_T, random_Tdot, random_Tdotdot) = solver.solve_ROM_nonparametric(n_modes_u, n_modes_theta,
                                                                             dict_dispersion_coeff, n_samples,
                                                                             [Udof_num], [node_num])

    ####
    # Plots
    confidence_level = 0.99

    plot_dofU_vs_t_random(vec_t, random_U[0, :, :], deterministic_U[0, :],
                          confidence_level=confidence_level, save_path='./example_UQ_1_U.png')
    plot_nodeT_vs_t_random(vec_t, random_T[0, :, :], deterministic_T[0, :, :],
                           confidence_level=confidence_level, save_path='./example_UQ_1_T.png')

    tf = time.time()
    print(f'Total time: {tf - t0:.2f} seconds.')