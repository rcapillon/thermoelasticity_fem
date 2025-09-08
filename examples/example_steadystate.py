# import os
# import sys
#
# sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import time

from thermoelasticity_fem.materials import LinearThermoElastic
from thermoelasticity_fem.mesh import Mesh
from thermoelasticity_fem.model import Model
from thermoelasticity_fem.solvers import LinearSteadyState
from thermoelasticity_fem.plots import plot_U_T


if __name__ == '__main__':
    t0 = time.time()

    mesh = Mesh()
    mesh.load_mesh('../data/sandwich.msh')

    material = LinearThermoElastic(rho=2300, Y=64e9, nu=0.1, k=1., c=750., alpha=4e-6, T0=20.)
    dict_materials = {
        1: material,
        2: material,
        3: material
    }
    mesh.set_materials(dict_materials)
    mesh.make_elements()

    dict_dirichlet_U = {
        4: [('x', 0.), ('y', 0.), ('z', 0.)],
        5: [('x', 0.), ('y', 0.), ('z', 0.)]
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

    dict_surface_forces = {
        7: np.array([0., 0., -1e9])
    }
    # dict_surface_forces = None

    # vec_f_vol = np.array([0., 0., -material.rho * 9.81])
    # dict_volume_forces = {
    #     1: vec_f_vol,
    #     2: vec_f_vol,
    #     3: vec_f_vol
    # }
    dict_volume_forces = None

    q = -25.
    dict_heat_flux = {
        6: q,
    }
    # dict_heat_flux = None

    # R = 1e3
    # dict_heat_source = {
    #     2: R
    # }
    dict_heat_source = None

    model = Model(mesh,
                  dict_dirichlet_U=dict_dirichlet_U, dict_dirichlet_theta=dict_dirichlet_theta,
                  dict_surface_forces=dict_surface_forces, dict_volume_forces=dict_volume_forces,
                  dict_heat_flux=dict_heat_flux, dict_heat_source=dict_heat_source)

    solver = LinearSteadyState(model)
    solver.solve()

    ####
    # Interactive plot (deformed mesh, temperature as color)
    save_path = './steadystate.png'

    plot_U_T(solver.model.mesh, solver.T, solver.U, save_path=save_path)

    tf = time.time()
    print(f'Total time: {tf - t0:.2f} seconds.')
