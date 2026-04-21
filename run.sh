#!/bin/bash

fluxtypes=("SFR" "no")
exps=("grand200k" "poemma")

for exp in "${exps[@]}"; do
  for fluxtype in "${fluxtypes[@]}"; do
    for d in {3..8}; do 

      echo "Running: fluxtype=$fluxtype, exp=$exp, d=$d"
      printf "%s\n%s\n%s\n" "$fluxtype" "$exp" "$d" | python3 flux2taucount.py
      
    done
  done
done
