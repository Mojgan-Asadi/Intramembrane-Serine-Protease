#!/bin/bash

k1list="0.00018 0.00020 0.00022"
dG1=13.0 # Barrier height
bw=0.3  # Barrier width
g=12.0  # gamma
bz0="-3.0"

outdir="implicit_substrate"

mkdir $outdir

for k1 in $k1list; do
  echo "k=${k1}..."
  fname="${outdir}/z0_${bz0}_bw_${bw}_dg_${dG1}_g_${g}_k_${k1}"
  logfile="${fname}.log"
  python ./ldchan.py --channel-k=1.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=12.0 \
      --dg-barrier=$dG1 --barrier-width=${bw} --constr-k=$k1 \
      --mass=32.0 --barrier-shift=${bz0} \
      --constr-z=20000.0 --dt=0.005 --num-steps=40000 --record-steps=10 --vac-steps=500 --gamma=${g} \
      --plot \
      --name="$fname" > $logfile
done