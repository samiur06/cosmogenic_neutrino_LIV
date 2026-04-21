#!/bin/bash
NJOBS=5

# python save_neutrino_mc.py 

python compute_taucount.py

python flux2taucount.py


#mkdir -p logs

# ## to do 1D counts/constraints
# fluxtypes=("SFR" "no")
# exps=("grand200k" "poemma")

# for exp in "${exps[@]}"; do
#   for fluxtype in "${fluxtypes[@]}"; do
#     for d in {3..8}; do
#       echo "Running: fluxtype=$fluxtype, exp=$exp, d=$d"
#       printf "%s\n%s\n%s\n" "$fluxtype" "$exp" "$d" \
#         | python3 flux2taucount.py \
#         > logs/param_${fluxtype}_${exp}_d${d}.log 2>&1 &

#       while [ $(jobs -r | wc -l) -ge $NJOBS ]; do
#         sleep 1
#       done

#     done
#   done
# done


# ### to do 2D counts/constraints
# fluxtype="SFR"
# exp="grand200k"
# d=6

# for pair_id in {1..15}; do
#   echo "Running: fluxtype=$fluxtype, exp=$exp, d=$d, pair=$pair_id"
#   printf "%s\n%s\n%s\n%s\n" "$fluxtype" "$exp" "$d" "$pair_id" \
#     | python3 flux2taucount.py \
#     > logs/param_${fluxtype}_${exp}_d${d}_pair${pair_id}.log 2>&1 &
#   while [ $(jobs -r | wc -l) -ge $NJOBS ]; do
#     sleep 1
#   done
# done

# wait
# echo "All done."
