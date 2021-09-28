## 1D Channel Langevin Dynamics

```shell
python ldchan.py  --channel-k=5.0 --wall-k=2.5 --ensemble-size=20 --sample-size=10 --channel-length=16.0 \
                  --dg-barrier=21.5 --barrier-width=5.8 --constr-k=0.0001 --constr-z=20000 --dt=0.005 \
                  --num-steps=200000 --record-steps=10 --vac-steps=500 --gamma=8
```

##  2D EVB Surface Langevin Dynamics

Example run

```shell
python lan2d.py --delta 2.5 5.0  --minima 0.0 2.0 3.0 4.0 --freqs 1.0 0.1 --jq1 0.05 0.05 --jq2 0.05 0.05 \
             --temp 300 --gamma 1.0 1.0 --mass 4.0 4.0 --init-q -1.0 -1.0 \
             --q-constr 1.0 -1.0 --k-constr 0.2 \
             --num-steps 20000 --ensemble-size 20 --plot-surface
```

Full options explanation can be found by running `python lan2d --help`