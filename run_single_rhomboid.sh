#!/bin/bash

k1list="0.00010 0.00013 0.00015 0.00017 0.00020"
dG1=21.4 # Barrier height
bw=5.0  # Barrier width
g=9.0  # gamma

outdir="implicit_rhomboid"

mkdir $outdir

for k1 in $k1list; do
  echo "k=${k1}..."
  fname="${outdir}/bw_${bw}_dg_${dG1}_g_${g}_k_${k1}"
  logfile="${fname}.log"
  python ./ldchan.py --channel-k=1.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=12.0 \
      --dg-barrier=$dG1 --barrier-width=${bw} --constr-k=$k1 \
      --mass=16.0 --barrier-shift=-1.5 \
      --constr-z=20000.0 --dt=0.005 --num-steps=50000 --record-steps=10 --vac-steps=500 --gamma=${g} \
      --plot \
      --name="$fname" > $logfile
done