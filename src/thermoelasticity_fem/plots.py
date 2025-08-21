import pyvista as pv


def plot_U_T(mesh, temperature, displacement):
    mio_mesh = mesh.make_meshio_mesh()

    p = pv.Plotter()
    pv_mesh = pv.from_meshio(mio_mesh)
    pv_mesh.point_data['T'] = temperature
    pv_mesh.set_active_scalars('T')
    pv_mesh['U'] = displacement.reshape((mesh.n_nodes, 3))
    warped_mesh = pv_mesh.warp_by_vector('U', factor=1.)
    p.add_mesh(warped_mesh, show_edges=True)
    p.show_axes()
    p.show()