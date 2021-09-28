#!/bin/bash

dG1=21.5
dG2=9.3
k1list="9.0e-5 1.0e-4 1.1e-4"
k1list2="8.0e-5"
k2list="3.0e-5 4.0e-5 5.0e-5"
k2list2="2.0e-5"

# autocorrelation collection
python ./ldchan.py --channel-k=5.0 --wall-k=2.5 --ensemble-size=20 --sample-size=10 --channel-length=16.0 \
    --dg-barrier=21.5 --barrier-width=5.8 --constr-k=1.0e-4 \
    --constr-z=20000 --dt=0.005 --num-steps=200000 --record-steps=10 --vac-steps=500 --gamma=8 \
    --name=implicit_data/dg_${dG1}_k1.0e-4_AC

for k1 in $k1list; do
  python ./ldchan.py --channel-k=5.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=16.0 \
      --dg-barrier=$dG1 --barrier-width=5.8 --constr-k=$k1 \
      --constr-z=20000 --dt=0.005 --num-steps=400000 --record-steps=10 --vac-steps=500 --gamma=8  \
      --name=implicit_data/dg_${dG1}_k_${k1}
done

for k1 in $k1list2; do
  python ./ldchan.py --channel-k=5.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=16.0 \
      --dg-barrier=$dG1 --barrier-width=5.8 --constr-k=$k1 \
      --constr-z=20000 --dt=0.005 --num-steps=800000 --record-steps=10 --vac-steps=500 --gamma=8  \
      --name=implicit_data/dg_${dG1}_k_${k1}
done

for k2 in $k2list; do
  python ./ldchan.py  --channel-k=5.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=16.0 \
      --dg-barrier=$dG2 --barrier-width=5.5 --constr-k=$k2 \
      --constr-z=20000 --dt=0.005 --num-steps=400000 --record-steps=10 --vac-steps=500 --gamma=8 \
      --name=implicit_data/dg_${dG2}_k_${k2}
done

for k2 in $k2list2; do
  python ./ldchan.py  --channel-k=5.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=16.0 \
      --dg-barrier=$dG2 --barrier-width=5.5 --constr-k=$k2 \
      --constr-z=20000 --dt=0.005 --num-steps=800000 --record-steps=10 --vac-steps=500 --gamma=8 \
      --name=implicit_data/dg_${dG2}_k_${k2}
done