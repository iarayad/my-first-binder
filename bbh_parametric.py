def compute_observables_parametric(L, params_list, E_Fermi = -0.5):
    """
    L (int): width and height of square lattice to generate.
    params_list (list of dict): builder parameters to evaluate Hamiltonian.
    E_Fermi (float): Fermi level to calculate occupied states.
    
    Returns finalized system and saves total charge, charge, charge density and dimensions of integration region.
    """
    sysf = create_parametric(L)

    hams = [sysf.hamiltonian_submatrix(params=params) for params in params_list]
    eigproblem = [np.linalg.eigh(ham) for ham in hams]

    Rmax = int(L // 2)
    sizes = np.arange(0, Rmax + 1)
    
    density_op = kwant.operator.Density(sysf)

    for i in range(len(params_list)):
        eigvals, eigvecs = eigproblem[i]
        density = np.sum([density_op(v) for v in eigvecs[:, eigvals < E_Fermi].T], axis=0)
        density_reshaped = density.reshape(4, L, L)
        density_reshaped = np.sum(density_reshaped-0.5, axis=0)

        defect_charge_abs = np.array([np.sum(np.abs(density_reshaped[L//2-d:L//2+d, L//2-d:L//2+d])) for d in sizes])
        defect_charge_cond = np.array([np.abs(np.sum(density_reshaped[L//2-d:L//2+d, L//2-d:L//2+d])) for d in sizes])
        params = params_list[i]
        common_path = "data/parametric_L_{}_gcx_{}_gcy_{}_gr_{}_phi0_{}_c_{}_d_{}_disorder_{}"
        path = common_path.format(L,
                                  params['g_cx'],
                                  params['g_cy'],
                                  params['g_r'],
                                  params['phi0'],
                                  params['c'],
                                  params['d'],
                                  params['disorder']
                                   )

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        np.save(path+'/abs_charge.npy', defect_charge_abs)
        np.save(path+'/cond_charge.npy', defect_charge_cond)
        np.save(path+'/d.npy', sizes)
        np.save(path+'/density', density)
    return sysf
