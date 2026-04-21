from LIV_fraction import * 
from LIV_tau_count import * 

def chisq(NBSM, N0):
    if NBSM==0 and N0==0:
        return 0 
    if NBSM==0 and N0!=0:
        raise Exception("NBSM is zero, N0 is not!!")
        
    frac = N0/NBSM
    if N0==0:
        frac=1 # evading log(0)
    # print(NBSM,N0,frac)
    return 2*(NBSM - N0 + N0 * np.log(frac))

# Integration limits in eV
E_min = 1e15      # 1 PeV = 1e15 eV
E_max = 100e18      # 100 EeV = 1e20 eV
year = 365*24*3600

exp=input("Enter experiment (poemma/ICgen2radio/grand200k): ")
d=int(input("Enter dimension: "))
# print(tau_count('poemma'), tau_count('ICgen2radio'), tau_count('grand200k'))
print((fluxtype, exp, d))

# upper_guess = 60 + 20*(d-3)
# lower_guess = 20 + 20*(d-3)
### new list only for POEMMA weak flux sensitivity 
upper_guess = 40 + 10*(d-3)
lower_guess = 20 + 10*(d-3) # or + 5*(d-3) for poemma weak fluxes
guesses = np.logspace(np.log10(10**( - lower_guess )), np.log10(10**( - upper_guess )), 20) 
# 20 is good enough via trial
# 50 for poemma weak flux e.g. mod, pess
tau_count_track = {}

src_scan = [[float(x), float(1-x), 0] for x in np.linspace(0, 1, 20)] # main
# src_scan = [[float(x), float(1-x), 0] for x in np.linspace(0, 1, 10)] # new for 50 guesses of poemma weak
src_scan.append([1,2,0])
src_scan=np.array(src_scan)

### 1. perform tau count
"""
print("==========================\ncomputing the number of tau...\n==========================")
for src_ratio in src_scan:# [[1,0,0]]:#,[0,1,0],[1,2,0]]:
    srctext = ''.join(f"{x}" for x in src_ratio)
    tau_count_track[srctext] = {}
    print(srctext)
    std_temp = tau_count(exp, src_ratio=src_ratio)
    for alpha in range(3):
        for beta in range(alpha,3):
            tau_vals = []
            tau_vals_std = []
            for sol in guesses:
                LIVmatrix = np.zeros((3, 3), dtype=complex)
                LIVmatrix[alpha, beta] = sol  # GeV^{-n}
                tau_vals.append([sol,
                                 tau_count(exp,
                                           d=d,
                                           a_eff=LIVmatrix, 
                                           c_eff=LIVmatrix,
                                           src_ratio=src_ratio)
                                ])
                tau_vals_std.append([sol, std_temp])

                # E=1e9*1e9
                # dm21, dm31 = np.array([7.49e-5, 2.534e-3])#*1e-18
                # theta12, theta13, theta23 =  [np.arcsin(np.sqrt(0.303)), 
                #                               np.arcsin(np.sqrt(0.02225)), 
                #                               np.arcsin(np.sqrt(0.572))]

                # delta = 197*np.pi/180

                # H0 = H0_flavor(E, dm21, dm31, theta12, theta23, theta13, delta)
                # HLIV = H_LIV_paper(E, 6, a_eff=LIVmatrix, c_eff=LIVmatrix)
                # print(f'H0={H0}')
                # print(f'HLIV={HLIV}')

            tau_vals = np.array(tau_vals)
            tau_vals_std = np.array(tau_vals_std)
            tau_count_track[srctext][alpha,beta] = {'LIV':tau_vals,
                                    'std':tau_vals_std}


print(f"time processed:{np.round(timeit.default_timer(),2)} s")

with open(f'data/tau_count/flux{fluxtype}_{exp}_dim{d}.pkl', 'wb') as f:
    pickle.dump(tau_count_track, f)
"""

### 2. computing sensitivity

with open(f'data/tau_count/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    tau_count_track = pickle.load(f)

print("==========================\nsolving for parameters\n==========================")

#### 2A. compute LIVparam sensitivity for each src 
# """
LIVparam = {d:{}}

for src_ratio in src_scan:#[[1,0,0]]:#,[0,1,0],[1,2,0]]:
    srctext = ''.join(f"{x}" for x in src_ratio)
    LIVparam[d][srctext] = {}
    for (alpha, beta), data in tau_count_track[srctext].items():
        # print(f"{srctext, alpha, beta}")
        tau_vals = data['LIV']   # shape (N,2)
        c_vals = tau_vals[:,0]
        NBSM_vals = tau_vals[:,1]
        N0 = tau_count(exp, src_ratio=src_ratio)

        # # compute diff array
        # diff_vals = NBSM_vals - N0 
        # # diff_vals=np.abs(NBSM_vals - N0)

        # # mask out zero-diff points
        # mask = diff_vals != 0
        # diff_vals_f = diff_vals[mask]
        # c_vals_f = c_vals[mask]

        # # build interpolator on filtered data
        # diff_of_c = interp1d(c_vals_f, diff_vals_f, 
        #                      kind='cubic', fill_value="extrapolate")
        

        # build interpolator on filtered data
        NBSM_function = interp1d(c_vals, NBSM_vals, 
                             kind='linear', fill_value="extrapolate")

        def makezero(cexp):
            c = 10**cexp
            target = chi2.ppf(0.95, 1)
            return chisq(NBSM_function(c), N0) - target        
        ### brentq
        # upper_guess = 40 + 10*(d-3)
        # lower_guess = 20 + 10*(d-3) # or + 5*(d-3) for poemma weak fluxes

        lower_bound_exp = -120 
        upper_bound_exp = -20
        
        # ---- Find a valid bracket automatically ----
        exp_grid = np.linspace(- (40 + 10*(d-3)), - (20 + 10*(d-3)), 1000+100*(d-3))
        fvals = np.array([makezero(e) for e in exp_grid])

        bracket_found = False
        for i in range(len(exp_grid)-1):
            if fvals[i] * fvals[i+1] < 0:
                lower_bound_exp = exp_grid[i]
                upper_bound_exp = exp_grid[i+1]
                bracket_found = True
                break

        if not bracket_found:
            print("No sign change found in scan range.")

        # print(lower_bound_exp, upper_bound_exp)

        try:
            # brentq takes the function and the two ends of the bracket
            cexp_sol = brentq(makezero, lower_bound_exp, upper_bound_exp, xtol=1e-12)
            
            c_solution = 10**cexp_sol
            # solution is within the guess range. This removes absurd extrapolated c_solution. 
            if not (min(c_vals) <= c_solution <= max(c_vals)):
                c_solution = 1e100 # arbitrary large value, not sensitive
            
            # print(f"Root found at c = {c_solution:.2e}, Verification (should be ~0): {makezero(cexp_sol)}")

        except ValueError as e:
            print(f"Root find failed: {e}")
            print("This usually means makezero(a) and makezero(b) have the same sign.")
            print("Check if your target is within the range of your chisq values.")
        if not bracket_found:
            c_solution=1e100 # arbitrary large value, not sensitive

        # print(chisq(NBSM_vals[0], N0),
        #       makezero(np.log10(c_vals[0])) )

        ### fsolve, needs good guess 

        # c_solution = 0
        # list_2 = [ np.sqrt(guesses[i] * guesses[i+1]) for i in range(len(guesses) - 1)]
        # nstz = 10 # number of ansatz to try, higher means longer
        # cstep = int(len(guesses)/nstz) 
        # ansatz_list = list_2[1::cstep]
        # # for c_guess in [1e-57, 1e-59, 1e-61, 1e-62]:
        # # for c_guess in [1e-57, 1e-58, 1e-59, 1e-60, 1e-61, 1e-62]:
        # for c_guess in np.log10(ansatz_list): #np.logspace(np.log10(1e-42), np.log10(1e-117), 5):
        #     cexp_sol = fsolve(makezero, c_guess, xtol=1e-12)[0]
        #     # print(makezero(c_solution), makezero(csol))
        #     csol = 10**cexp_sol
        #     if np.abs(makezero(cexp_sol))<0.001 and csol>0:
        #         c_solution = csol
        #     print(makezero(cexp_sol),csol,c_solution)

        # print(f"{alpha, beta}, c =", c_solution,
        #   np.round(makezero(np.log10(c_solution)),4))

        LIVparam[d][srctext][f"{alpha,beta}"] = c_solution
print(f"time processed:{np.round(timeit.default_timer(),2)} s")

with open(f'data/param/flux{fluxtype}_{exp}_dim{d}.pkl', 'wb') as f:
    pickle.dump(LIVparam, f)

with open(f'data/param/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    LIVparam = pickle.load(f)
"""

#### 2B. compute LIVparam sensitivity for uniform x result src=(x:1-x:0) 


#### 2B (I) arrange and save uniform tau count
"""
uniform_srclist = src_scan[:-1] # last one is inserted non-uniformly
count_shape = next(iter(tau_count_track.values()))[0,0]['LIV'].shape[0]
dx = uniform_srclist[1][0] - uniform_srclist[0][0]
print(dx)
totNBSM_dict = {'uniformx': {}}
for alpha in range(3):
    for beta in range(alpha,3):
        temp_NBSM = np.zeros(count_shape)
        temp_N0 = 0
        for src_ratio in uniform_srclist:
            srctext = ''.join(f"{x}" for x in src_ratio)
            data = tau_count_track[srctext][alpha,beta]
            tau_vals = data['LIV'] 
            c_vals = tau_vals[:,0]
            NBSM_vals = tau_vals[:,1]
            # marginalizing/averaging the quantity NBSM across 0<x=f_e<1 at the source
            temp_NBSM += NBSM_vals*dx 
            temp_N0 += data['std'][:,1][0] * dx

        arrLIV = np.array([c_vals, temp_NBSM]).reshape(2, count_shape).T
        arrstd = np.array([c_vals, temp_N0*np.ones(count_shape)]).reshape(2, count_shape).T
        totNBSM_dict['uniformx'][alpha,beta] = {'LIV':arrLIV, 
                                                'std':arrstd }

with open(f'data/tau_count/uniformx/flux{fluxtype}_{exp}_dim{d}.pkl', 'wb') as f:
    pickle.dump(totNBSM_dict, f)
"""

#### 2B (II) compute LIVparam for uniformx src
"""
LIVparam = {d:{}}

with open(f'data/tau_count/uniformx/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    totNBSM_dict = pickle.load(f)

LIVparam[d]['uniformx'] = {}
# print(totNBSM_dict)
# for alpha in range(1):
#     for beta in range(1):
for alpha in range(3):
    for beta in range(alpha,3):

        data = totNBSM_dict['uniformx'][alpha,beta]
        tau_vals = data['LIV']   # shape (N,2)
        c_vals = tau_vals[:,0]
        NBSM_vals = tau_vals[:,1]
        N0 = data['std'][:,1][0]
        # c_vals = totNBSM_dict['uniformx'][alpha,beta][0]
        # NBSM_vals = totNBSM_dict['uniformx'][alpha,beta][1]
        # N0 = totNBSM_dict[alpha,beta][2]
        # print(alpha,beta,c_vals,NBSM_vals,N0,'===================')
        # build interpolator on filtered data
        NBSM_function = interp1d(c_vals, NBSM_vals, 
                             kind='linear', fill_value="extrapolate")
        
        def makezero(cexp):
            c = 10**cexp
            target = chi2.ppf(0.95, 1)
            # return chisq(NBSM_function(c), N0) - target        
            if NBSM_function(c)>=0:
                return chisq(NBSM_function(c), N0) - target    
            else:
                return None    

        ### brentq
        lower_bound_exp = -120 
        upper_bound_exp = -20
        # print(np.log10(min(c_vals)), np.log10(max(c_vals)))
        # ---- Find a valid bracket automatically ----
        exp_grid = np.linspace(- (40 + 10*(d-3)), - (20 + 10*(d-3)), 1000+100*(d-3))
        # exp_grid = np.linspace(np.log10(min(c_vals)), np.log10(max(c_vals)), 1000+100*(d-3))
        fvals = np.array([makezero(e) for e in exp_grid]) 
        # print(np.shape(fvals),fvals[0])       

        # print(exp_grid[0],exp_grid[1],exp_grid[2])
        # print(exp_grid[-1],exp_grid[-2],exp_grid[-3])
        bracket_found = False
        for i in range(len(exp_grid)-1):
            if fvals[i] * fvals[i+1] < 0:
                lower_bound_exp = exp_grid[i]
                upper_bound_exp = exp_grid[i+1]
                bracket_found = True
                break

        if not bracket_found:
            exp_grid = np.linspace(- (40 + 10*(d-3)), - (20 + 10*(d-3)), 1000+100*(d-3))
            # exp_grid = np.linspace(np.log10(min(c_vals)), np.log10(max(c_vals)), 100*(1000+100*(d-3)))
            fvals = np.array([makezero(e) for e in exp_grid])

            # print(exp_grid[0],exp_grid[1],exp_grid[2])
            # print(exp_grid[-1],exp_grid[-2],exp_grid[-3])

            bracket_found = False
            for i in range(len(exp_grid)-1):
                if fvals[i] * fvals[i+1] < 0:
                    lower_bound_exp = exp_grid[i]
                    upper_bound_exp = exp_grid[i+1]
                    bracket_found = True
                    break

            if not bracket_found:
                print("No sign change found in scan range.")

        # print(lower_bound_exp, upper_bound_exp)

        try:
            # brentq takes the function and the two ends of the bracket
            cexp_sol = brentq(makezero, lower_bound_exp, upper_bound_exp, xtol=1e-12)
            
            c_solution = 10**cexp_sol
            # solution is within the guess range. This removes absurd extrapolated c_solution. 
            if not (min(c_vals) <= c_solution <= max(c_vals)):
                c_solution = 1e100 # arbitrary large value, not sensitive
            
            # print(f"Root found at c = {c_solution:.2e}, Verification (should be ~0): {makezero(cexp_sol)}")

        except ValueError as e:
            print(f"Root find failed: {e}")
            print("This usually means makezero(a) and makezero(b) have the same sign.")
            print("Check if your target is within the range of your chisq values.")
        if not bracket_found:
            c_solution=1e100 # arbitrary large value, not sensitive

        print(alpha,beta,c_solution,
            np.round([NBSM_function(c_solution),N0, 
                chisq(NBSM_function(c_solution),N0)],2))
        LIVparam[d]['uniformx'][f"{alpha,beta}"] = c_solution
print(f"time processed:{np.round(timeit.default_timer(),2)} s")

with open(f'data/param/uniformx/flux{fluxtype}_{exp}_dim{d}.pkl', 'wb') as f:
    pickle.dump(LIVparam, f)

with open(f'data/param/uniformx/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    LIVparam = pickle.load(f)

# """
