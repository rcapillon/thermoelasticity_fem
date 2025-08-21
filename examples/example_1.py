import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pyvista as pv

from thermoelasticity_fem.materials import LinearThermoElastic
from thermoelasticity_fem.mesh import Mesh
from thermoelasticity_fem.model import Model
from thermoelasticity_fem.solvers import LinearSteadyState


if __name__ == '__main__':
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

    vec_u_dir1 = np.array([0., 0., 0.])
    vec_u_dir2 = np.array([0., 0., 0.])
    dict_dirichlet_U = {
        4: vec_u_dir1,
        5: vec_u_dir2
    }

    T_dir1 = 20.
    T_dir2 = 20.
    # T_dir3 = 40.
    dict_dirichlet_T = {
        4: T_dir1,
        5: T_dir2,
        # 7: T_dir3
    }
    # dict_dirichlet_T = None

    vec_f_surf = np.array([0., 0., -1e9])
    dict_surface_forces = {
        7: vec_f_surf
    }
    # dict_surface_forces = None

    q = -25.
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

    model = Model(mesh,
                  dict_dirichlet_U=dict_dirichlet_U, dict_dirichlet_T=dict_dirichlet_T,
                  dict_surface_forces=dict_surface_forces,
                  dict_heat_flux=dict_heat_flux, dict_heat_source=dict_heat_source)

    solver = LinearSteadyState(model)
    solver.solve()

    ####
    # Interactive plot (deformed mesh, temperature as color)

    mio_mesh = mesh.make_meshio_mesh()

    p = pv.Plotter()
    pv_mesh = pv.from_meshio(mio_mesh)
    pv_mesh.point_data['T'] = solver.temperature
    pv_mesh.set_active_scalars('T')
    pv_mesh['U'] = solver.displacement.reshape((mesh.n_nodes, 3))
    warped_mesh = pv_mesh.warp_by_vector('U', factor=1.)
    p.add_mesh(warped_mesh, show_edges=True)
    p.show_axes()
    p.show()