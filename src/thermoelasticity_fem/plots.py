import numpy as np
import matplotlib.pyplot as plt
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


def animate_U_T(mesh, temperature, displacement, vec_t, save_path, fps=10, quality=5):
    mio_mesh = mesh.make_meshio_mesh()

    val_min = np.amin(temperature.flatten())
    val_max = np.amax(temperature.flatten())

    clim = [val_min, val_max]
    cmap = plt.cm.get_cmap("jet")

    p = pv.Plotter()
    p.open_movie(save_path, framerate=fps, quality=quality)
    pv_mesh = pv.from_meshio(mio_mesh)
    pv_mesh.point_data['T'] = temperature[:, 0]
    pv_mesh.set_active_scalars('T')
    pv_mesh['U'] = displacement[:, 0].reshape((mesh.n_nodes, 3))
    warped_mesh = pv_mesh.warp_by_vector('U', factor=1.)
    actor = p.add_mesh(
        warped_mesh,
        cmap=cmap,
        clim=clim,
        # scalar_bar_args={'background_color': [1.0, 1.0, 1.0]},
        show_edges=True,
        # interpolate_before_map=False
    )
    p.show_axes()
    p.add_title(f't = {vec_t[0]:.1f} s')
    p.write_frame()

    for i in range(1, vec_t.size):
        p.remove_actor(actor)
        pv_mesh = pv.from_meshio(mio_mesh)
        pv_mesh.point_data['T'] = temperature[:, i]
        pv_mesh.set_active_scalars('T')
        pv_mesh['U'] = displacement[:, i].reshape((mesh.n_nodes, 3))
        warped_mesh = pv_mesh.warp_by_vector('U', factor=1.)
        actor = p.add_mesh(
            warped_mesh,
            cmap=cmap,
            clim=clim,
            # scalar_bar_args={'background_color': [1.0, 1.0, 1.0]},
            show_edges=True,
            # interpolate_before_map=False
        )
        p.show_axes()
        p.add_title(f't = {vec_t[i]:.1f} s')
        p.write_frame()

    p.close()