import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import kwant
import matplotlib
from matplotlib import cm
import tinyarray
import os

# +
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
    
def create_parametric(L):
    """
    L (int): width and height of square lattice to generate.
    
    Returns finalized kwant.builder.FiniteSystem of dimensions LxL with 4 orbitals separated by b in a unit cell of axa.
    Hoppings gamma_x and gamma_y are given according to a parametrization of radius g_r around (g, g).
    """
    a = 1
    b = 0.5

    #position of atoms counterclockwise starting from left bottom
    a_1 = [a/2+b/2, a/2+b/2]
    a_2 = [a/2-b/2, a/2-b/2]
    a_3 = [a/2-b/2, a/2+b/2]
    a_4 = [a/2+b/2, a/2-b/2]
    squares = kwant.lattice.general([[a, 0], [0, a]], # lattice vectors
                                 [a_1, a_2, a_3, a_4], norbs=1) #coordinates of the sites
    s_1, s_2, s_3, s_4 = squares.sublattices

    # building system and setting hoppings
    syst = kwant.Builder()

    def square_c(pos):
        x, y = pos
        return np.abs(x) <= L//2 and np.abs(y) <= L//2

    def on_site(site, disorder, salt): 
        return disorder * kwant.digest.gauss(site.tag, salt)

    # hoppings
    def hop_g_x(site1, site2, g_cx, g_r, c, phi0):
        """
        site1 (kwant.Builder.Site): site connected by hopping
        site2 (kwant.Builder.Site): site connected by hopping
        g_cx (float): x-coordinate of the center of the parametrization
        g_r (float): radius of the parametrization
        c (float): parameter for a more general parametrization, not just a circle (c=0)
        phi0 (float): rotation angle of the parametrization
        
        Returns position-dependent gamma parameter along the x-direction
        """
        pos1 = site1.pos
        pos2 = site2.pos
        posc = (pos1 + pos2)/2
        phi = np.arctan2(posc[1], posc[0])
        return g_cx + parametrization(g_r, c, phi, phi0)[0] # x-projection

    def hop_g_y_23(site1, site2, g_cy, g_r, c, phi0):
        """
        site1 (kwant.Builder.Site): site connected by hopping
        site2 (kwant.Builder.Site): site connected by hopping
        g_cy (float): y-coordinate of the center of the parametrization
        g_r (float): radius of the parametrization
        c (float): parameter for a more general parametrization, not just a circle (c=0)
        phi0 (float): rotation angle of the parametrization
        
        Returns position-dependent gamma parameter along the y-direction with a negative sign
        """
        pos1 = site1.pos
        pos2 = site2.pos
        posc = (pos1 + pos2)/2
        phi = np.arctan2(posc[1], posc[0])
        return -(g_cy + parametrization(g_r, c, phi, phi0)[1]) # y-projection

    def hop_g_y_14(site1, site2, g_cy, g_r, c, phi0):
        """
        site1 (kwant.Builder.Site): site connected by hopping
        site2 (kwant.Builder.Site): site connected by hopping
        g_cy (float): y-coordinate of the center of the parametrization
        g_r (float): radius of the parametrization
        c (float): parameter for a more general parametrization, not just a circle (c=0)
        phi0 (float): rotation angle of the parametrization
        
        Returns position-dependent gamma parameter along the y-direction
        """
        pos1 = site1.pos
        pos2 = site2.pos
        posc = (pos1 + pos2)/2
        phi = np.arctan2(posc[1], posc[0])
        return g_cy + parametrization(g_r, c, phi, phi0)[1] # y-projection

    def hop_d_x(site1, site2, d):
        """
        site1 (kwant.Builder.Site): site connected by hopping
        site2 (kwant.Builder.Site): site connected by hopping
        d (float): hopping value
        
        Returns sublattice symmetry breaking parameter along x-direction
        """
        return d

    def hop_d_y(site1, site2, d):
        """
        site1 (kwant.Builder.Site): site connected by hopping
        site2 (kwant.Builder.Site): site connected by hopping
        d (float): hopping value
        
        Returns sublattice symmetry breaking parameter along y-direction
        """
        return d
    
    # onsite
    syst[squares.shape(square_c, (0, 0))] = on_site

    # hopping inside cell
    syst[kwant.HoppingKind((0, 0), s_1, s_3)] = hop_g_x
    syst[kwant.HoppingKind((0, 0), s_2, s_3)] = hop_g_y_23
    syst[kwant.HoppingKind((0, 0), s_1, s_4)] = hop_g_y_14
    syst[kwant.HoppingKind((0, 0), s_2, s_4)] = hop_g_x

    #hopping outside cell x-direction        
    syst[kwant.HoppingKind((1, 0), s_1, s_3)] = 1
    syst[kwant.HoppingKind((1, 0), s_4, s_2)] = 1

    #hopping outside cell y-direction 
    syst[kwant.HoppingKind((0, 1), s_3, s_2)] = -1
    syst[kwant.HoppingKind((0, 1), s_1, s_4)] = 1

    #2nd neighbor hopping outside cell x-direction
    syst[kwant.HoppingKind((1, 0), s_1, s_1)] = hop_d_x
    syst[kwant.HoppingKind((1, 0), s_2, s_2)] = hop_d_x
    syst[kwant.HoppingKind((1, 0), s_3, s_3)] = hop_d_x
    syst[kwant.HoppingKind((1, 0), s_4, s_4)] = hop_d_x

    #2nd neighbor hopping outside cell y-direction
    syst[kwant.HoppingKind((0, 1), s_1, s_1)] = hop_d_y
    syst[kwant.HoppingKind((0, 1), s_2, s_2)] = hop_d_y
    syst[kwant.HoppingKind((0, 1), s_3, s_3)] = hop_d_y
    syst[kwant.HoppingKind((0, 1), s_4, s_4)] = hop_d_y

    return syst.finalized()


# -

def parametrization(radius, coef, angle, angle0):
    """
    radius (float): radius of parametrization.
    coef (float): deformation parameter.
    angle (float): angle between radial component and x-axis.
    angle0 (float): angular offset.
    
    Returns x and y components of a limacon parametrization around defect in parameter space.
    
    Circular parametrization is given for coef = 0.
    If radius > coef the parametrization will not have inner loops.

    Look at limacon shape here: https://www.math.uh.edu/~jiwenhe/Math1432/lectures/lecture14_handout.pdf
    """
    r = (radius + coef * np.cos(angle))
    x = r*np.cos(angle + angle0)
    y = r*np.sin(angle + angle0)
    return x, y


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
