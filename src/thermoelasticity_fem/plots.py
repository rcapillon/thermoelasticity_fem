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


def plot_dofU_vs_t(Udof_num, vec_t, displacement, save_path=None):
    _, ax = plt.subplots()
    ax.plot(vec_t, displacement[Udof_num, :].flatten(), '-b')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('displacement [m]')
    ax.set_title(f'Displacement, U_DOF {Udof_num}')
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_nodeT_vs_t(node_num, vec_t, temperature, save_path=None):
    _, ax = plt.subplots()
    ax.plot(vec_t, temperature[node_num, :].flatten(), '-b')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Temperature [degrees]')
    ax.set_title(f'Temperature, node {node_num}')
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_dofU_vs_t_random(vec_t, displacement_random, displacement_deterministic,
                          confidence_level=None, save_path=None):
    Udof_mean = np.mean(displacement_random[:, :], axis=-1)

    _, ax = plt.subplots()
    ax.plot(vec_t, Udof_mean.flatten(), '-b', label='statistical mean')
    ax.plot(vec_t, displacement_deterministic, '-r', label='deterministic response')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('displacement [m]')
    ax.set_title('Displacement, observed dof')
    ax.legend()
    ax.grid()

    if confidence_level is not None:
        n_rejected_samples = int(np.floor((1 - confidence_level) * displacement_random.shape[2] / 2))
        ordered_samples = np.zeros((displacement_random.shape[1], displacement_random.shape[2]))
        lower_confidence_bound = np.zeros((displacement_random.shape[1], ))
        upper_confidence_bound = np.zeros((displacement_random.shape[1], ))
        for i in range(displacement_random.shape[1]):
            ordered_samples[i, :] = np.sort(displacement_random[i, :])
            lower_confidence_bound[i] = ordered_samples[i, n_rejected_samples]
            upper_confidence_bound[i] = ordered_samples[i, -(n_rejected_samples + 1)]
        ax.plot(vec_t, lower_confidence_bound, '--g', label=f'{confidence_level * 100}% confidence interval')
        ax.plot(vec_t, upper_confidence_bound, '--g')
        ax.fill_between(vec_t, lower_confidence_bound, upper_confidence_bound, color='cyan')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_nodeT_vs_t_random(vec_t, temperature_random, temperature_deterministic,
                          confidence_level=None, save_path=None):
    Tnode_mean = np.mean(temperature_random, axis=-1)

    _, ax = plt.subplots()
    ax.plot(vec_t, Tnode_mean.flatten(), '-b', label='statistical mean')
    ax.plot(vec_t, temperature_deterministic, '-r', label='deterministic response')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('temperature [degrees]')
    ax.set_title('Temperature, observed node')
    ax.legend()
    ax.grid()

    if confidence_level is not None:
        n_rejected_samples = int(np.floor((1 - confidence_level) * temperature_random.shape[2] / 2))
        ordered_samples = np.zeros((temperature_random.shape[1], temperature_random.shape[2]))
        lower_confidence_bound = np.zeros((temperature_random.shape[1], ))
        upper_confidence_bound = np.zeros((temperature_random.shape[1], ))
        for i in range(temperature_random.shape[1]):
            ordered_samples[i, :] = np.sort(temperature_random[i, :])
            lower_confidence_bound[i] = ordered_samples[i, n_rejected_samples]
            upper_confidence_bound[i] = ordered_samples[i, -(n_rejected_samples + 1)]
        ax.plot(vec_t, lower_confidence_bound, '--g', label=f'{confidence_level * 100}% confidence interval')
        ax.plot(vec_t, upper_confidence_bound, '--g')
        ax.fill_between(vec_t, lower_confidence_bound, upper_confidence_bound, color='cyan')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
