#!/bin/bash

dG1=21.5
dG2=9.3
k1list="8.0e-5"
k2list="2.0e-5"


for k1 in $k1list; do
  python ./ldchan.py --channel-k=5.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=16.0 \
      --dg-barrier=$dG1 --barrier-width=5.8 --constr-k=$k1 \
      --constr-z=20000 --dt=0.01 --num-steps=800000 --record-steps=5 --vac-steps=500 --gamma=8  \
      --name=implicit_data/dg_long_${dG1}_k_${k1}
done


for k2 in $k2list; do
  python ./ldchan.py  --channel-k=5.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=16.0 \
      --dg-barrier=$dG2 --barrier-width=5.5 --constr-k=$k2 \
      --constr-z=20000 --dt=0.01 --num-steps=800000 --record-steps=5 --vac-steps=500 --gamma=8 \
      --name=implicit_data/dg_long_${dG2}_k_${k2}
done