import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def plot_U_T(mesh, temperature, displacement, amplification_factor_U=1., save_path=None):
    mio_mesh = mesh.make_meshio_mesh()

    p = pv.Plotter()
    pv_mesh = pv.from_meshio(mio_mesh)
    pv_mesh.point_data['T'] = temperature
    pv_mesh.set_active_scalars('T')
    pv_mesh['U'] = displacement.reshape((mesh.n_nodes, 3))
    warped_mesh = pv_mesh.warp_by_vector('U', factor=amplification_factor_U)
    p.add_mesh(warped_mesh, show_edges=True)
    p.show_axes()
    p.show()

    if save_path is not None:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(p.image)
        plt.savefig(save_path)


def animate_U_T(mesh, temperature, displacement, vec_t, save_path,
                amplification_factor_U=1., fps=10, quality=5, step=1):
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
    warped_mesh = pv_mesh.warp_by_vector('U', factor=amplification_factor_U)
    actor = p.add_mesh(
        warped_mesh,
        cmap=cmap,
        clim=clim,
        # scalar_bar_args={'background_color': [1.0, 1.0, 1.0]},
        show_edges=True,
        # interpolate_before_map=False
    )
    p.show_axes()
    p.add_title(f't = {vec_t[0]:.3f} s')
    p.write_frame()

    for i in range(1, vec_t.size, step):
        p.remove_actor(actor)
        pv_mesh = pv.from_meshio(mio_mesh)
        pv_mesh.point_data['T'] = temperature[:, i]
        pv_mesh.set_active_scalars('T')
        pv_mesh['U'] = displacement[:, i].reshape((mesh.n_nodes, 3))
        warped_mesh = pv_mesh.warp_by_vector('U', factor=amplification_factor_U)
        actor = p.add_mesh(
            warped_mesh,
            cmap=cmap,
            clim=clim,
            # scalar_bar_args={'background_color': [1.0, 1.0, 1.0]},
            show_edges=True,
            # interpolate_before_map=False
        )
        p.show_axes()
        p.add_title(f't = {vec_t[i]:.3f} s')
        p.write_frame()

    p.close()


def plot_dofU_vs_t(dof_num, vec_t, displacement, save_path=None):
    _, ax = plt.subplots()
    ax.plot(vec_t, displacement[dof_num, :], '-b')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('displacement [m]')
    ax.set_title(f'Displacement, DOF {dof_num}')
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_nodeT_vs_t(node_num, vec_t, temperature, save_path=None):
    _, ax = plt.subplots()
    ax.plot(vec_t, temperature[node_num, :], '-b')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Temperature [degrees]')
    ax.set_title(f'Temperature, node {node_num}')
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
