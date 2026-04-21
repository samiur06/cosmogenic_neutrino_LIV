from LIV_fraction import * 
# from LIV_tau_count import * 

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

def makezero(x):
    target = chi2.ppf(0.95, 1)
        return chisq(NBSM_function(c), x) - target    
    else:
        return None    

fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)

xvals= np.linspace(2,1000,20)
mylst = []
for x in xvals: 
	mylst.append(chisq(2*y, y))

ax.plot(
    x, mylst,
    linestyle=':',
    color='blue',
    linewidth=2
)

fsolve(chisq(unknown, x)-chi2.ppf(0.95, 1), 2*x, xtol=1e-12)[0]

plt.tight_layout()
plt.show()