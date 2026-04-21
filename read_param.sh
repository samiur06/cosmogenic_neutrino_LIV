for f in data/param/flux*.pkl; do
    # parse filename: flux{fluxtype}_{exp}_dim{d}.pkl
    base=$(basename $f .pkl)           # fluxSFR_grand200k_dim3
    fluxtype=$(echo $base | sed 's/flux\(.*\)_.*_dim.*/\1/')
    exp=$(echo $base     | sed 's/flux.*_\(.*\)_dim.*/\1/')
    d=$(echo $base       | sed 's/.*_dim\(.*\)/\1/')

    echo "─── fluxtype=$fluxtype  exp=$exp  dim=$d ───"
    python3 -c "import pickle; print(pickle.load(open('$f','rb')))"
done | tee data/param/summary.log