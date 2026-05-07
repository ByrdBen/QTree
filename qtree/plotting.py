import matplotlib.pyplot as plt
import numpy as np

def big_heatmaps(dat_list, key_list, param_values, name_list, save_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    #name_list = [r'$⟨n_c⟩ = 0$', r'$⟨n_c⟩ = 1$', r'$⟨n_c⟩ = 2$', r'$⟨n_c⟩ = 3$']
    ims = []
    if 'q0' in key_list[0]:
        title = r'$\Psi_{init} = |0⟩$'
        save_name = 'ground_state_crit_cap.jpg'
    else:
        title = r'$\Psi_{init} = |1⟩$'
        save_name = 'excited_state_crit_cap.jpg'
    for ax, key, name in zip(axes, key_list, name_list):
        #ax.imshow(dat0[key].T, extent=extent, cmap='magma')
        ax.set_title(name)
        ax.set_xlabel('Flux (Φ/Φ₀)')
        ax.set_ylabel(r'$g_{chain}$ (GHz)')
        crit_list = [get_ncrit(data, key) for data in dat_list]
        
        extent = [0, .5, np.min(param_values), np.max(param_values)]
        min_len = min(len(c) for c in crit_list)
        crit_arr = np.array([c[:min_len] for c in crit_list])
        im = ax.imshow(np.log10(crit_arr), aspect='auto', cmap='magma', extent=extent)
        ims.append(im)
        ax.set_aspect("auto")
        
    fig.subplots_adjust(left=0.08, right=0.88, wspace=0.2, hspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.125, 0.02, .75])  # [left, bottom, width, height]
    fig.colorbar(ims[0], cax=cbar_ax, label=r'$\log{n_{crit}}$')
    fig.suptitle(title, fontsize=16)
    plt.savefig(save_name, dpi=200)
    plt.show()

def get_intersections_colored(starting_state, flux_arr, e_list_flux, key_list):
    flux_arr = np.array(flux_arr)
    n_r_list = np.arange(0, 80, 1)
    w_r = 6.627
    n_vals = [1]
    m_vals = np.arange(1, 40, 1)
    tol = 0.0002

    intersections = {}  # (i,j): list of (Φ, n_r)

    key_list2 = [starting_state]

    for f_idx, Φ in enumerate(flux_arr):
        e_dat = e_list_flux[f_idx]
        for key_i in key_list2:
            i = int(key_i)
            for j, key_j in enumerate(key_list):
                if j <= i:
                    continue
                E_i = np.array(e_dat[key_i][0])
                E_j = np.array(e_dat[key_j][0])
                ω_ij = E_j - E_i
                for n in n_vals:
                    for m in m_vals:
                        ratio = ((ω_ij) / (m * w_r))
                        idx_match = np.where(np.abs(ratio - 1) < tol)[0]
                        if len(idx_match) > 0:
                            intersections.setdefault((i, j), []).extend(
                                [(Φ, n_r_list[idx]) for idx in idx_match]
                            )

    # --- plotting ---
    import matplotlib as mpl
    cmap = mpl.colormaps.get('tab10')
    
    plt.figure(figsize=(8, 5))
    for k, ((i, j), pts) in enumerate(intersections.items()):
        if not pts:
            continue
        pts = np.array(pts)
        plt.scatter(
            pts[:, 0], pts[:, 1],
            s=.2,
            color=cmap(k / max(1, len(intersections)-1)),
            label=f'{i}->{j}'
        )
    plt.xlabel("Flux (Φ/Φ₀)")
    plt.ylabel(r"$⟨n_r⟩$")
    plt.legend(title="Transition (i→j)", fontsize=8)
    plt.tight_layout()
    #plt.show()