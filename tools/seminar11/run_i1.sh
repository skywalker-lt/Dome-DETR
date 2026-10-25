#!/usr/bin/env bash
# Seminar 11 inference job I1 on pod 3 (batch-1 / batch-16 test-set inference only; no training).
#  a  E / E2 / EP / E2P (d1 rewrites, d2 frozen pruning) on N1-E + DeFE fixes s1 and N1-E s0, epoch-159 EMA
#  b  E-pad check: N1-E + DeFE fixes s1 evaluated at batch 16 (in-training evaluator) must equal its batch-1 final
#  c  checkpoint0131 of the six FP-1 runs, batch-1 full test (132-epoch schedule question)
# usage (from this branch's checkout): bash tools/seminar11/run_i1.sh  -> /root/i1/{a,b,c}
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); PY=/root/venv/bin/python; O=/root/i1; mkdir -p $O/a $O/b $O/c
IMG=/root/aitod/images_test_link; GT=/root/aitod/annotations/aitodv2_test.json; RS=/root/rescore; RUNS=/root/runs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
echo "$(date -u +%FT%TZ) I1 start $(git -C $R rev-parse --short HEAD)"
for n in n1e_defefix_fp1_s1 n1e_fp1_s0; do
  $PY $R/tools/seminar11/fulltest_stack.py --repo $R --d1 $R/tools/seminar11 --ckpt $RS/${n}_ep159.pth --name $n \
    --img-dir $IMG --gt $GT --out $O/a --count-every 10 --device cuda --threads 4 > $O/a/$n.log 2>&1
  echo "$(date -u +%FT%TZ) a $n rc=$? $(tail -1 $O/a/$n.log)"
done
cd $R && $PY train.py -c configs/dome/pod/Dome-S-n1e-defefix-fp1-AITOD-pod.yml --test-only -r $RS/n1e_defefix_fp1_s1_ep159.pth \
  -u output_dir=$O/b > $O/b/test16.log 2>&1
echo "$(date -u +%FT%TZ) b rc=$? $(grep -E 'AP|current_stat|Average Precision' $O/b/test16.log | head -2 | tr '\n' ' ')"
declare -A CFG=([n1e_fp1_s0]=Dome-S-n1e-fp1 [n1e_fp1_s1]=Dome-S-n1e-fp1-s1 [thin_fp1_s1]=Dome-S-thin-fp1
                [thin_scaleaug_fp1_s1]=Dome-S-thin-scaleaug-fp1 [n1e_distill_fp1_s1]=Dome-S-n1e-distill-fp1 [n1e_defefix_fp1_s1]=Dome-S-n1e-defefix-fp1)
for n in "${!CFG[@]}"; do
  $PY tools/pod/run_eval.py configs/dome/pod/${CFG[$n]}-AITOD-pod.yml $RUNS/dome_s_$n/checkpoint0131.pth $IMG $GT $O/c/${n}_ep131 \
    --repo $R --device cuda --threads 4 --no-eval --no-maps > $O/c/${n}_ep131.log 2>&1
  echo "$(date -u +%FT%TZ) c $n rc=$? $(grep -E 'loaded|inference done' $O/c/${n}_ep131.log | tr '\n' ' ')"
done
rm -f /root/queue/STOP   # the user put job I1 before the queued training runs: release the queue now
echo "$(date -u +%FT%TZ) I1 DONE (queue released)"
