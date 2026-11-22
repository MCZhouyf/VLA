children=()
for seed in 2026112100 2026112101 2026112102 2026112103 2026112104; do
    for group in 100 60 10,30; do
        IFS=, read -a distances <<< "$group"
        run="$result/pilot-$seed-$group"
        bash code/minemem/t1_exec.sh protected-pilot "$run" --fixtures "$verified" \
            --seeds-file "$frozen" --seed "$seed" --models random steve1 vpt \
            --distances "${distances[@]}" >"$run.log" 2>&1 &
        children+=("$!")
        sleep 0.15
    done
done
status=0
for child in "${children[@]}"; do wait "$child" || status=1; done
test "$status" = 0
envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_respawn_report.py pilot \
    "$result/pilot-report" --seeds-file "$frozen" --inputs "$result"/pilot-*/cliff_T1.csv
