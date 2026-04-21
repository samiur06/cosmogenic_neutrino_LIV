from LIV_fraction import * 
from LIV_tau_count import * 
exp=input("Enter experiment (poemma/ICgen2radio/grand200k): ")
d=int(input("Enter dimension: "))
print((fluxtype, exp, d))
with open(f'data/param/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
    LIVparam = pickle.load(f)
# src_ratio_all = [[1,0,0],[0,1,0],[1,2,0]]
# srctexts = ["100", "010", "120"]
src_ratio_all = [[1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [1.0,2.0,0.0]]
srctexts = ['1.00.00.0', '0.01.00.0', '1.02.00.0']
flux_list = ['pess','mod','opt']
flux_label = ['no evolution', 'SFR evolution', 'AGN evolution']
src_list = LIVparam[d].keys()
# get alpha,beta keys from one entry
alphabeta_keys = list(LIVparam[d][srctexts[0]].keys())  # 6 of them
# prepare data matrix: rows = alpha,beta, cols = srctext
vals = np.zeros((len(alphabeta_keys), len(srctexts)))
for i, ab in enumerate(alphabeta_keys):
    for j, sr in enumerate(srctexts):
        vals[i, j] = LIVparam[d][sr][ab]
# bar positions
x = np.arange(len(alphabeta_keys))
width = 0.20
flavlabel = [r'$e$',r'$\mu$',r'$\tau$']
# print('vals=========',LIVparam[d])

### dimensions and 3 srccodes
"""
plt.figure(figsize=(8,5))
y_top = 1e-1
n_j = len(srctexts)
cmap = plt.get_cmap("tab10")  # very distinguishable
# colors = [cmap(j) for j in range(n_j)]
# cmap = plt.get_cmap("viridis")
drange = np.arange(3,9)
# to simplify making the legend for dimensions
for d in drange:
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    # some lines that doesn't appear in the plot
    plt.hlines(
        10, 
        x[0],   # left end
        x[1],   # right end
        color=base_color,
        linewidth=2,
        linestyle='-'
    )
for d in drange:
    print(d)
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    colors = [base_color[:3] + (0.4 + 0.6*j/len(srctexts),) 
              for j in range(len(srctexts))]
    # print(colors)
    with open(f'data/param/flux{fluxtype}_{exp}_dim{d}.pkl', 'rb') as f:
        LIVparam = pickle.load(f)
    # get alpha,beta keys from one entry
    alphabeta_keys = list(LIVparam[d][srctexts[0]].keys())  # 6 of them
    # prepare data matrix: rows = alpha,beta, cols = srctext
    vals = np.zeros((len(alphabeta_keys), len(srctexts)))
    for i, ab in enumerate(alphabeta_keys):
        for j, sr in enumerate(srctexts):
            vals[i, j] = LIVparam[d][sr][ab]

    for j in range(len(srctexts)):
        print(len(np.log10(vals[:, j])), np.shape(np.log10(vals[:, j])))
        plt.bar(x + j*width, np.log10(vals[:, j]) - np.log10(y_top),
                color='none', 
                edgecolor=base_color,#colors[j],
                linewidth=2,
                width=width)
        # plt.bar(x, (np.log10(vals[:, j]) - np.log10(y_top))/10,
        #         color=colors[j], width=0.5*width)
# x-ticks: alpha,beta labels
plt.xticks(x + width, 
          [f"{flavlabel[k[0]]}{flavlabel[k[1]]}"
           for k in (ast.literal_eval(x) for x in alphabeta_keys)],
          fontsize=14)
# plt.yscale("log")
# plt.ylim(np.log10(1e-40), -70)
plt.ylim(np.log10(1e-30), -180)
# plt.ylim(np.log10(1e-40)/10, -120/10)
plt.gca().invert_yaxis()
plt.ylabel(r"$\log{\kappa_{\alpha\beta}/\,\mathrm{GeV}^{4-d}}$", fontsize=14)
plt.xlabel(r"$(\alpha,\beta)$", fontsize=14)
plt.title(f"varying source ratio, {[x for x in srctexts]}", fontsize=14)

plt.legend([f'd={x}' for x in drange], 
    frameon=False, fontsize=14, ncol=3, loc='lower left')
plt.tick_params(top=False, bottom=False, right=True,
                direction="in", which="both")
plt.minorticks_on()
plt.tight_layout()
plt.savefig(f"figures/param_{fluxtype}_{exp}.pdf", dpi=300)
plt.close()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

### dimensions and exp with fixed_srctext
"""
plt.figure(figsize=(8,5))
y_top = 1e-1
n_j = len(srctexts)
cmap = plt.get_cmap("tab10")  # very distinguishable
# colors = [cmap(j) for j in range(n_j)]
# cmap = plt.get_cmap("viridis")

fixed_id_src = 2
src_ratio=src_ratio_all[fixed_id_src]
fixed_srctext = srctexts[fixed_id_src] #'1.00.00.0'#'0.01.00.0'#'1.02.00.0'
# line_flux = ['-','--',':']
line_flux = ['--',(0,(1,0.5)),(0,(1,2))]
drange = np.arange(3,9)
# to simplify making the legend for dimensions
for d in drange:
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    # some lines that doesn't appear in the plot
    plt.hlines(
        10, 
        x[0],   # left end
        x[1],   # right end
        color=base_color,
        linewidth=2,
        linestyle='-'
    )
# to simplify making the legend for flux types
for idx_ls, ls in enumerate(line_flux):
    plt.hlines(
        10*idx_ls, 
        x[0],   # left end
        x[1],   # right end
        color='black',
        linewidth=2,
        linestyle=ls
    )
for d in drange:
    print(d)
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    colors = [base_color[:3] + (0.4 + 0.6*j/len(srctexts),) 
              for j in range(len(srctexts))]
    
    for iflux, flux_var in enumerate(['pess','mod','opt']):
        with open(f'data/param/flux{flux_var}_{exp}_dim{d}.pkl', 'rb') as f:
            LIVparam = pickle.load(f)
        # get alpha,beta keys from one entry
        alphabeta_keys = list(LIVparam[d][fixed_srctext].keys())  # 6 of them
        # prepare data matrix: rows = alpha,beta, cols = srctext
        vals = np.zeros((len(alphabeta_keys), 1))
        for i, ab in enumerate(alphabeta_keys):
            for j, sr in enumerate([fixed_srctext]):
                vals[i, j] = LIVparam[d][sr][ab]
        y = np.log10(vals[:, 0]) - np.log10(y_top)
        for xi, yi in zip(x + iflux*width, y):
            plt.hlines(
                yi,
                xi - width/2,   # left end
                xi + width/2,   # right end
                color=base_color,
                linewidth=2,
                linestyle=line_flux[iflux]
            )
            
            # print(yi)
            # Vertical line (like a bar)
            for xedge in [xi - width/2,xi + width/2]:
                plt.vlines(
                    xedge,
                    yi,   # bottom
                    -20,   # top
                    color='black',#base_color,
                    linewidth=1,
                    linestyle='solid'#line_flux[iflux]
                )

# x-ticks: alpha,beta labels
plt.xticks(x + width, 
          [f"{flavlabel[k[0]]}{flavlabel[k[1]]}"
           for k in (ast.literal_eval(x) for x in alphabeta_keys)],
          fontsize=14)
# plt.yscale("log")
# plt.ylim(np.log10(1e-40), -70)
plt.ylim(np.log10(1e-30), -180)
# plt.ylim(np.log10(1e-40)/10, -120/10)
plt.gca().invert_yaxis()
plt.ylabel(r"$\log{\kappa_{\alpha\beta}/\,\mathrm{GeV}^{4-d}}$", fontsize=14)
plt.xlabel(r"$(\alpha,\beta)$", fontsize=14)


legend_tags = [f'd={x}' for x in drange] + flux_label
plt.legend(legend_tags, 
    frameon=False, fontsize=14,
    ncol=3,
    loc='lower left', 
    bbox_to_anchor=(0.0, 0.0))

total_ratio = int(sum(src_ratio))
s = ",".join(
    f"{int(v)}/{total_ratio}" if (v and total_ratio != 1) 
    else str(int(v))
    for v in src_ratio
    )
label = rf"$f_S=({s})$"
plt.text(0.97, 0.07, label, transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=14)

plt.tick_params(top=False, bottom=False, right=True,
                direction="in", which="both")
plt.minorticks_on()
plt.tight_layout()
# plt.savefig(f"figures/param_{exp}_src{fixed_srctext}.pdf", dpi=300)
plt.close()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
"""

###  dimensions and exp || uniform x result
# """
plt.figure(figsize=(8,5))
y_top = 1e-1
n_j = len(srctexts)
cmap = plt.get_cmap("tab10")  # very distinguishable
# colors = [cmap(j) for j in range(n_j)]
# cmap = plt.get_cmap("viridis")
fixed_srctext = '0.01.00.0'#'1.00.00.0'#'0.01.00.0'#'1.02.00.0'
# line_flux = ['-','--',':']
line_flux = ['--',(0,(1,0.5)),(0,(1,2))]
drange = np.arange(3,9)
# to simplify making the legend for dimensions
for d in drange:
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    # some lines that doesn't appear in the plot
    plt.hlines(
        10, 
        x[0],   # left end
        x[1],   # right end
        color=base_color,
        linewidth=2,
        linestyle='-'
    )
# to simplify making the legend for flux types
for idx_ls, ls in enumerate(line_flux):
    plt.hlines(
        10*idx_ls, 
        x[0],   # left end
        x[1],   # right end
        color='black',
        linewidth=2,
        linestyle=ls
    )
  
for d in drange:
    print(d)
    base_color = cmap((d-3)/5)  # normalize d in [0,1]
    colors = [base_color[:3] + (0.4 + 0.6*j/len(srctexts),) 
              for j in range(len(srctexts))]
    for iflux, flux_var in enumerate(['pess','mod','opt']):
        fname = f'data/param/uniformx/flux{flux_var}_{exp}_dim{d}.pkl'
        if not os.path.exists(fname):
            print(f"Warning: missing {fname}, skipping")
            continue   # leave empty, move to next flux

        with open(fname, 'rb') as f:
            LIVparam = pickle.load(f)

        # get alpha,beta keys from one entry
        alphabeta_keys = list(LIVparam[d]['uniformx'].keys())  # 6 of them
        # prepare data matrix: rows = alpha,beta, cols = srctext
        vals = np.zeros((len(alphabeta_keys), 1))
        for i, ab in enumerate(alphabeta_keys):
            vals[i, 0] = LIVparam[d]['uniformx'][ab]

        # print(d, flux_var, alphabeta_keys, vals)
        # n_pairs = len(alphabeta_keys)
        # n_flux = 3  # number of histograms per pair

        # # spacing parameters
        # bar_width = 0.15      # width of each histogram
        # intra_gap = 0.0     # gap between histograms within a pair
        # inter_gap = 0.4      # gap between consecutive pairs

        # # x positions for each pair
        # x_base = np.arange(n_pairs) * (n_flux*bar_width + inter_gap)

        # # values
        # vals = np.array([LIVparam[d]['uniformx'][ab] for ab in alphabeta_keys])
        # y = np.log10(vals) - np.log10(y_top)

        # # loop over pairs
        # for i, (xi_pair, yi_val) in enumerate(zip(x_base, y)):
        #     # loop over histograms in the pair
        #     for iflux in range(n_flux):
        #         xi = xi_pair + iflux*(bar_width + intra_gap)  # intra-pair shift
        #         yi = yi_val #yi_val[iflux] if yi_val.ndim>1 else yi_val  # if multiple fluxes
                
        #         # horizontal line
        #         plt.hlines(
        #             yi,
        #             xi - bar_width/2,
        #             xi + bar_width/2,
        #             color=base_color,
        #             linewidth=2,
        #             linestyle=line_flux[iflux]
        #         )
                
        #         # ensure negative placeholder if no result
        #         if yi > 0:
        #             yi = -130
                
        #         # vertical edges
        #         for xedge in [xi - bar_width/2, xi + bar_width/2]:
        #             plt.vlines(
        #                 xedge,
        #                 yi,
        #                 -20,
        #                 color='black',
        #                 linewidth=1,
        #                 linestyle='solid'
        #             )
  
        y = np.log10(vals[:, 0]) - np.log10(y_top)
        for xi, yi in zip(x + iflux*width, y):
            plt.hlines(
                yi,
                xi - width/2,   # left end
                xi + width/2,   # right end
                color=base_color,
                linewidth=2,
                linestyle=line_flux[iflux]
            )
            
            # if exp=='poemma' and fluxtype=='pess' or 'mod':
            #     print(fluxtype, d, yi)

            if yi>0: # no result
                yi = -75
            # Vertical line (like a bar)
            for xedge in [xi - width/2,xi + width/2]:
                plt.vlines(
                    xedge,
                    yi,   # bottom
                    -20,   # top
                    color='black',#base_color,
                    linewidth=1,
                    linestyle='solid'#line_flux[iflux]
                )


# x-ticks: alpha,beta labels
tick_labels = [rf"$\mathring{{\kappa}} ^{{(d)}}_{{{flavlabel[k[0]].strip('$')}{flavlabel[k[1]].strip('$')}}}$"
               for k in (ast.literal_eval(x) for x in alphabeta_keys)]
plt.xticks(x + width, tick_labels, fontsize=14)
# plt.xticks(x + width, 
#           [rf"$\mathring{{\kappa}} ^{{(d)}} _{flavlabel[k[0]]}{flavlabel[k[1]]}$"
#           # [f"{flavlabel[k[0]]}{flavlabel[k[1]]}"
#            for k in (ast.literal_eval(x) for x in alphabeta_keys)],
#           fontsize=14)
# plt.yscale("log")
# plt.ylim(np.log10(1e-40), -70)
plt.ylim(np.log10(1e-20), -100)
# plt.ylim(np.log10(1e-40)/10, -120/10)
plt.gca().invert_yaxis()
plt.ylabel(r"$\log \left( {\mathring{\kappa}^{{(d)}}_{\alpha\beta}/\,\mathrm{GeV}^{4-d}} \right) $", fontsize=14)
# plt.xlabel(r"$(\alpha,\beta)$", fontsize=14)

legend_tags = [f'd={x}' for x in drange] + flux_label
plt.legend(legend_tags, 
    frameon=False, fontsize=14,
    ncol=3,
    loc='lower left', 
    bbox_to_anchor=(0.0, 0.0))

label = r"$f_S=(x,1-x,0)$"
plt.text(0.97, 0.07, label, transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=14)

plt.tick_params(top=False, bottom=False, right=True,
                direction="in", which="both")
plt.minorticks_on()
plt.tight_layout()
plt.savefig(f"figures/param_{exp}_src_uniformx.pdf", dpi=300)
plt.show()
print(f"time processed:{np.round(timeit.default_timer(),2)} s")
# """
