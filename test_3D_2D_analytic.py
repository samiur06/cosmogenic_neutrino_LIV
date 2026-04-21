from LIV_fraction import * 

def get_pmns_test(E, d, a_eff=None, c_eff=None):
    E    = np.asarray(E, dtype=float)
    scalar_input = E.ndim == 0
    E    = np.atleast_1d(E)
    H0 = H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta)
    # H0   = H0_flavor(E, 0, dm31, 0, theta23, 0, 0)
    HLIV = H_LIV_paper(E, d, a_eff=a_eff, c_eff=c_eff)
    Htot = H0 + HLIV
    # print('H0: ',H0)
    # print('HLIV: ',HLIV)
    # scipy_eigh doesn't support batched (N,3,3) — loop over N
    N = Htot.shape[0]
    evecs = np.empty((N, 3, 3), dtype=complex)
    for i in range(N):
        _, evecs[i] = scipy_eigh(Htot[i])

    if scalar_input:
        return evecs[0]
    return evecs

def prob_avg_redshifted_test(E, z, d, a_eff=None, c_eff=None):
    """
    Returns avgP of shape (3,3) for scalar E/z, or (N,3,3) for arrays.
    avgP[alpha, beta] = sum_i |Vprod[alpha,i]|^2 * |Vdet[beta,i]|^2
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)
    scalar_input = E.ndim == 0
    E = np.atleast_1d(E)
    z = np.atleast_1d(z)

    Vprod = get_pmns_test(E * (1 + z), d, a_eff, c_eff)          # (N,3,3)
    Vdet  = get_pmns_test(E, d, a_eff, c_eff)                    # (N,3,3)

    absProd2 = np.abs(Vprod)**2                          # (N,3,3)
    absDet2  = np.abs(Vdet)**2                           # (N,3,3)

    # avgP[n,alpha,beta] = sum_i absProd2[n,alpha,i] * absDet2[n,beta,i]
    avgP = np.einsum('nai,nbi->nab', absProd2, absDet2)  # (N,3,3)

    print(np.round(Vprod,3))
    print(np.round(Vdet,3))


    if scalar_input:
        return avgP[0]                                   # (3,3)
    return avgP                                          # (N,3,3)


LIVmatrix = np.zeros((3, 3), dtype=complex)

LIVmatrix[1,2] = 1e-54  # GeV^{-n}

tada = prob_avg_redshifted_test(1e9 * 1.58e8, 1, 
    6, a_eff=LIVmatrix, c_eff=LIVmatrix )#[0,0]

dm21, dm31 = np.array([7.49e-5, 2.534e-3])
theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                              np.arcsin(np.sqrt(0.02225)), 
                              np.arcsin(np.sqrt(0.572))]

delta = 197*np.pi/180

print((tada@[1/3,2/3,0])[2])

print(flavor_fraction_redshifted(1e9 * 1.58e8, 1, dm21=0, dm31=dm31, 
                                                      theta12=0, theta23=theta23, theta13=0, delta=0,
                                                      d=6, 
                                                      a_eff=LIVmatrix, 
                                                      c_eff=LIVmatrix,
                                                      src_ratio=[1,2,0], flavor='tau'))


tada = prob_avg_redshifted_test(1e9 * 1.58e8, 0, 
    6, a_eff=LIVmatrix, c_eff=LIVmatrix )#[0,0]

## 2D stuff python made from mathematica

def mix2new_numerical(aee, amm, aem, dm21, En, theta, d):
    # Mixing matrix
    mix2 = np.array([
        [ np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    # Free Hamiltonian in mass basis, then rotate to flavor basis
    free2    = np.array([[0, 0], [0, dm21 / (2 * En)]])
    free2v2  = mix2 @ free2 @ mix2.T

    # LIV matrix
    liv_mat  = (-1)**(d + 1) * En**(d - 3) * np.array([
        [aee, aem],
        [aem, amm]
    ])

    mat2 = free2v2 + liv_mat

    # Eigensystem
    vals, vecs = np.linalg.eig(mat2)
    vecs = vecs.T  # numpy returns column eigenvectors, so transpose to get rows

    # Normalize
    normvecs = np.array([v / np.linalg.norm(v) for v in vecs])

    # Order by eigenvalue (ascending)
    ord_idx  = np.argsort(vals)
    mix2new  = normvecs[ord_idx].T  # transpose to get column format like Mathematica

    return mix2new, vals[ord_idx]

# Numerical substitution
result, eigenvalues = mix2new_numerical(
    aee   = 0,
    amm   = 0,
    aem   = 1e-54,
    dm21  = 2.5e-21,
    En    = 1.58e8,
    theta = np.arcsin(np.sqrt(0.572)),#np.radians(43.3),
    d     = 6
)

print("mix2new =\n", result)
print("eigenvalues =", eigenvalues)