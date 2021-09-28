#!/bin/bash

dGlist="14.0"
gammalist="2.0 4.0 5.0 7.0 10.0"
k1list="0.00018 0.00020 0.00022"
bwidthlist="0.2 0.3 0.4 0.5"
bz0="-3.5"
outdir="implicit_substrate_z0_${bz0}"
mkdir "${outdir}"

for dG1 in $dGlist; do
  echo "dG = ${dG1}"
  for bw in $bwidthlist; do
    echo "Barrier Width: ${bw} A"
    mkdir "$outdir/bw_${bw}_dg_${dG1}"
    for g in $gammalist; do
      echo "g=${g}"
      for k1 in $k1list; do
        fname="bw_${bw}_dg_${dG1}/g_${g}_k_${k1}"
        logfile="${outdir}/${fname}.log"
        echo "  k = ${k1}"
        if [ -f "$logfile" ]; then
          echo "skipping $logfile "
          continue
        fi
        python ./ldchan.py --channel-k=1.0 --wall-k=2.5 --ensemble-size=40 --sample-size=20 --channel-length=12.0 \
          --dg-barrier=$dG1 --barrier-width=${bw} --constr-k=$k1 \
          --mass=32.0 --barrier-shift=${bz0} \
          --constr-z=20000.0 --dt=0.005 --num-steps=20000 --record-steps=10 --vac-steps=500 --gamma=${g} \
          --plot \
          --name="${outdir}/${fname}" > "$logfile" &
        sleep 1
      done
      echo "  Waiting..."
      wait
    done
  done
done
