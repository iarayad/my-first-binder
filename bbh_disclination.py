import kwant
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import tinyarray
from matplotlib import cm
import matplotlib.transforms as transforms
import os
from copy import copy
from collections import namedtuple


# ### Quadrupole model
# * $\gamma$ is intra-cell hopping
# * $\lambda$ is inter-cell hopping
# * $d$ is NNN hopping
# * $\alpha$ amplifies position dependence

# +
def create_template(L=1):
    """
    L (int): width and height of square lattice to generate.

    Returns kwant.builder.FiniteSystem template.
    """
    a = 1
    b = 0.5
    if L < a:
        raise ValueError("L must be bigger than a")
    if a <= b:
        raise ValueError("a must be bigger than b")

    # position of atoms counterclockwise starting from left bottom
    a_1 = np.array([a / 2 + b / 2, a / 2 + b / 2])
    a_2 = np.array([a / 2 - b / 2, a / 2 - b / 2])
    a_3 = np.array([a / 2 - b / 2, a / 2 + b / 2])
    a_4 = np.array([a / 2 + b / 2, a / 2 - b / 2])
    square = kwant.lattice.general([[a, 0], [0, a]], [a_1, a_2, a_3, a_4], norbs=1)  # coordinates of the sites
    s_1, s_2, s_3, s_4 = square.sublattices

    # building system and setting hoppings
    syst = kwant.Builder(symmetry=kwant.TranslationalSymmetry([a, 0], [0, a]))

    def square_c(pos):
        x, y = pos
        return np.abs(x) <= L and np.abs(y) <= L

    def on_site(site, disorder, salt):
        return disorder * kwant.digest.gauss(site.tag, salt)

    # onsite
    syst[square.shape(square_c, (0, 0))] = 0

    # hopping inside cell
    syst[kwant.HoppingKind((0, 0), s_1, s_3)] = hop_g_x
    syst[kwant.HoppingKind((0, 0), s_2, s_3)] = hop_g_y_23
    syst[kwant.HoppingKind((0, 0), s_1, s_4)] = hop_g_y_14
    syst[kwant.HoppingKind((0, 0), s_2, s_4)] = hop_g_x

    # hopping outside cell x-direction
    syst[kwant.HoppingKind((-1, 0), s_1, s_3)] = hop_l_x
    syst[kwant.HoppingKind((-1, 0), s_4, s_2)] = hop_l_x

    # hopping outside cell y-direction
    syst[kwant.HoppingKind((0, -1), s_3, s_2)] = hop_l_y_23
    syst[kwant.HoppingKind((0, -1), s_1, s_4)] = hop_l_y_14

    #2nd neighbor hopping outside cell x-direction
    syst[kwant.HoppingKind((-1, 0), s_1, s_1)] = hop_d
    syst[kwant.HoppingKind((-1, 0), s_2, s_2)] = hop_d
    syst[kwant.HoppingKind((-1, 0), s_3, s_3)] = hop_d
    syst[kwant.HoppingKind((-1, 0), s_4, s_4)] = hop_d

    #2nd neighbor hopping outside cell y-direction
    syst[kwant.HoppingKind((0, -1), s_1, s_1)] = hop_d
    syst[kwant.HoppingKind((0, -1), s_2, s_2)] = hop_d
    syst[kwant.HoppingKind((0, -1), s_3, s_3)] = hop_d
    syst[kwant.HoppingKind((0, -1), s_4, s_4)] = hop_d

    return syst, square


def hop_g_x(site1, site2, g, alpha_g, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    g (float): gamma parameter
    alpha_g (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent lambda parameter
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return g * (1 + alpha_g * hop_pos_dep(pos1, pos2))


def hop_g_y_23(site1, site2, g, alpha_g, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    g (float): gamma parameter
    alpha_g (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent gamma parameter with negative sign
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return -g * (1 + alpha_g * hop_pos_dep(pos1, pos2))


def hop_g_y_14(site1, site2, g, alpha_g, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    g (float): gamma parameter
    alpha_g (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent gamma parameter
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return g * (1 + alpha_g * hop_pos_dep(pos1, pos2))


def hop_l_x(site1, site2, l, alpha_l, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    l (float): lambda parameter
    alpha_l (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent lambda parameter
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return l * (1 + alpha_l * hop_pos_dep(pos1, pos2))


def hop_l_y_23(site1, site2, l, alpha_l, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    l (float): lambda parameter
    alpha_l (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent lambda parameter with negative sign
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return -l * (1 + alpha_l * hop_pos_dep(pos1, pos2))


def hop_l_y_14(site1, site2, l, alpha_l, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    l (float): lambda parameter
    alpha_l (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent lambda parameter
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return l * (1 + alpha_l * hop_pos_dep(pos1, pos2))


def hop_d(site1, site2, d, alpha_d, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    d (float): delta parameter
    alpha_d (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent sublattice symmetry breaking hopping
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return d * (1 + alpha_d * hop_pos_dep(pos1, pos2))


def hop_d_minus(site1, site2, d, alpha_d, trf):
    """
    site1 (kwant.Builder.Site): site connected by hopping
    site2 (kwant.Builder.Site): site connected by hopping
    d (float): delta parameter
    alpha_d (float): electron-lattice coupling
    trf (function): maps the lattice points to a different position
    
    Returns position-dependent sublattice symmetry breaking hopping with a negative sign
    """
    pos1 = trf(site1.pos)
    pos2 = trf(site2.pos)
    return -d * (1 + alpha_d * hop_pos_dep(pos1, pos2))


# +
def hop_pos_dep(pos1, pos2):
    """
    pos1, pos2 (list): x and y components of sites' position.
    
    Returns deviation of bond length from rest value.
    """
    return la.norm(pos1 - pos2) - (1 / 2) * (4 / 3)

def ang(pos):
    """
    pos (float): list of x and y components of a site's position.
    
    Returns anglular component of site's positon with respect to the x-axis.
    """
    x, y = pos
    return np.arctan2(y, x)

def C4_defect_trf(pos):
    """
    pos (list): x and y components of a site's position.
    
    Returns a list with the transformed x and y components such that r-> 4/3 * r and theta = 4/3 * theta.
    """
    r = la.norm(pos)
    angle = ang(pos)
    angle = angle * 4 / 3
    return 4 / 3 * r * np.array([np.cos(angle), np.sin(angle)])

def rotate_angle(pos, rot_angle):
    """
    pos (list): x and y components of a site's position.
    angle (float): angle of rotation of the lattice
    
    Returns a list with the transformed x and y components such that (r, theta) -> (r, angle + theta).
    """
    r = la.norm(pos)
    angle = ang(pos)
    exp_angle = np.exp(1j*angle)
    exp_rot_angle = np.exp(1j*rot_angle)
    return r * np.array([np.real(exp_angle*exp_rot_angle), np.imag(exp_angle*exp_rot_angle)])


# -

# ### Disclination

# +
def quarter(pos, delta=1e-5):
    """
    pos (float): list of x and y components of a site's position.
    delta (float): error threshold.
    
    Returns True if a site is located in the upper left quarter of the lattice (to be removed), False otherwise.
    """
    r = la.norm(pos)
    angle = ang(pos)
    return (-np.pi + delta > angle or np.pi / 2 + delta < angle) and r > delta

def cross_ray(pos1, pos2, phi, r0):
    """
    pos1, pos2 (float): list of x and y components of sites' position.
    phi (float): angular component of ray.
    r0 (float): list of x and y component's of the ray's origin.
    
    Returns True if the ray crosses the line that connects two sites, False otherwise.
    """
    z1 = np.exp(-1j * phi) * complex(*(pos1 - r0))
    z2 = np.exp(-1j * phi) * complex(*(pos2 - r0))
    return z1.imag > 0 and z2.imag < 0 and (z1 / z2).imag > 0


def add_defect(builder, lattice, remove_center=False):
    """
    builder (kwant.Builder): kwant system in which to insert the disclination
    lattice (kwant.lattice): builder's lattice
    remove_center (Bool): whether to remove the lattice points near the origin
    
    Returns the builder with a disclination
    """
    syst = kwant.Builder()
    syst.update(builder)
    builder = syst
    s_1, s_2, s_3, s_4 = lattice.sublattices

    # find hopping across the cut to keep, s1 is kept, s2 will be cut out
    delta = 1e-5
    hopping_keep_1 = [
        ((s1, s2), hop)
        for (s1, s2), hop in builder.hopping_value_pairs()
        if cross_ray(s1.pos, s2.pos, np.pi / 2, np.array([0, 0]))
    ]

    hopping_keep_2 = [
        ((s1, s2), hop)
        for (s1, s2), hop in builder.hopping_value_pairs()
        if cross_ray(s2.pos, s1.pos, np.pi / 2, np.array([0, 0]))
    ]

    print(f"{len(hopping_keep_1)=}, {len(hopping_keep_2)=}")
    # find the image sites of the removed ones
    phi = np.pi / 2
    R = la.inv(np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]))

    sites = list(builder.sites())
    pos_array = np.array([s.pos for s in sites]).T
    trf_pos_array = R @ pos_array
    dxs = np.add.outer(pos_array[0], -trf_pos_array[0])
    dys = np.add.outer(pos_array[1], -trf_pos_array[1])
    permutation = np.argwhere(np.logical_and(np.isclose(dxs, 0), np.isclose(dys, 0)))
    perm_dict = {sites[original]: sites[trf] for trf, original in permutation}

    # add back the hoppings to the site across the cut
    for (s1, s2), hop in hopping_keep_1:
        s1_image = perm_dict[s1]
        if np.allclose(np.array(s1_image.pos) - np.array(s2.pos), 0):
            continue
        if hop == hop_d and s2.family == s_3:
            builder[s1_image, s2] = hop_d_minus
        else:
            builder[s1_image, s2] = hop

    for (s1, s2), hop in hopping_keep_2:
        s2_image = perm_dict[s2]
        if np.allclose(np.array(s1.pos) - np.array(s2_image.pos), 0):
            continue
        if hop == hop_d and s1.family == s_3:
            builder[s1, s2_image] = hop_d_minus
        else:
            builder[s1, s2_image] = hop

    if remove_center:
        remove_sites = [
            s
            for s in builder.sites()
            if quarter(s.pos) or np.allclose(s.pos, np.zeros((2)), atol=delta)
        ]
    else:
        remove_sites = [s for s in builder.sites() if quarter(s.pos)]

    print(len(remove_sites), len(builder.sites()))

    for s in remove_sites:
        del builder[s]

    return builder


# -

def create_disclination(L):
    """
    L (int): width and height of a square lattice
    
    Returns two finalized kwant systems with the disclination. The first one is not position transformed, but the second one is. 
    """
    L_template = 1
    template, square = create_template(L_template)

    def shape_square(site):
        x, y = site.pos
        return (np.abs(x) <= L / 2) and (np.abs(y) < L / 2)

    syst = kwant.Builder()
    syst.fill(template, shape_square, (0.25, 0.25))
    sysf = add_defect(syst, square, remove_center=True).finalized()
    
    Site = namedtuple("Site", "pos")
    transformed_sysf = copy(sysf)
    transformed_sysf.sites = [Site(C4_defect_trf(site.pos)) for site in transformed_sysf.sites]
    transformed_sysf.sites = [Site(rotate_angle(site.pos, -np.pi/2)) for site in transformed_sysf.sites]
    return sysf, transformed_sysf


# ### Defect charge

def calculate_defect_charge(sysf, eigvecs, eigvals, E_Fermi):
    """
    sysf (finalized kwant.Builder): finalized system of the disclination
    eigvecs (np.array): eigenvectors of the system's Hamiltonian
    eigvals (np.array): eigenvalues of the system's Hamiltonian
    E_Fermi (float): Fermi level
    """
    def defect_charge_triangle(d):
        def where(site):
            """Returns true in squared region of side d at the center. False otherwise,"""
            return all([(d >= np.abs(site.pos[0]) and d >= np.abs(site.pos[1]))])

        defect_density_op = kwant.operator.Density(sysf, where=where, sum=False)
        defect_density = np.array([defect_density_op(v) for v in eigvecs[:, eigvals < E_Fermi].T])
        defect_charge_abs = np.sum(np.abs(np.sum(defect_density, axis=0) - 0.5), axis=0) #first sum adds over occupied states, second sum adds density over sites
        defect_charge_cond = np.sum(np.sum(defect_density, axis=0) - 0.5, axis=0)
        return np.abs(defect_charge_abs), np.abs(defect_charge_cond)

    Rmax = int(np.sqrt(len(eigvals)/4) // 2)
    charge_abs = np.zeros(Rmax+1)
    charge_cond = np.zeros(Rmax+1)
    sizes = np.arange(0, Rmax + 1)
    for i in range(Rmax):
        charge_absi, charge_condi = defect_charge_triangle(sizes[i])
        charge_abs[i] = charge_absi
        charge_cond[i] = charge_condi
    return sizes, charge_abs, charge_cond


# ### Density of charge

def compute_observables_disclination(L, params_list, E_Fermi = -0.5):
    """
    L (int): width and height of a square lattice
    params_list (list of dictionaries): parameters to which compute observables in disclination system
    E_Fermi (float): Fermi level
    
    Computes and saves the absolute defect charge deviation, the total defect charge and the local charge density together with their integration areas.
    Returns two finalized kwant systems with the disclination. The first one is not position transformed, but the second one is. 
    """
    sysf, transformed_sysf = create_disclination(L)

    hams = [sysf.hamiltonian_submatrix(params=params) for params in params_list]
    eigproblem = [np.linalg.eigh(ham) for ham in hams]

    density_op = kwant.operator.Density(sysf)
    densities = [np.sum([density_op(v) for v in eig[1][:, eig[0] < E_Fermi].T], axis=0) for eig in eigproblem]

    results = [calculate_defect_charge(sysf, eig[1], eig[0], E_Fermi) for eig in eigproblem]
    sizes = [result[0] for result in results]
    defect_charge_abs = [result[1] for result in results]
    defect_charge_cond = [result[2] for result in results]

    for i in range(len(params_list)):
        params = params_list[i]
        path = 'data/disclination_L_{}_g_{}_l_{}_d_{}_ag_{}_al_{}_ad_{}'.format(L,
                                                                                params['g'],
                                                                                params['l'],
                                                                                params['d'],
                                                                                params['alpha_g'],
                                                                                params['alpha_l'],
                                                                                params['alpha_d']
                                                                               )
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        np.save(path+'/abs_charge.npy', defect_charge_abs[i])
        np.save(path+'/cond_charge.npy', defect_charge_cond[i])
        np.save(path+'/d.npy', sizes[i])
        np.save(path+'/density', densities[i])
    return sysf, transformed_sysf
