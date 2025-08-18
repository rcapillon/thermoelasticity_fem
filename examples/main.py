import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pyvista as pv

from thermoelasticity_fem.mesh import Mesh


if __name__ == '__main__':
    msh = Mesh()
    msh.load_mesh('../data/sandwich.msh')

    mesh = msh.make_meshio_mesh()

    p = pv.Plotter()
    pv_mesh = pv.from_meshio(mesh)
    p.add_mesh(mesh, show_edges=True)
    p.show_axes()
    p.show()