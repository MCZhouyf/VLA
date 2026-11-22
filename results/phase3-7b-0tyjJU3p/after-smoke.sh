#!/usr/bin/env bash
set -euo pipefail
run="$1"
cd "$WORK"
public="$run/public"
variants=(no_memory long spatial)
trap 'printf "%s\n" "$?" >"$run/after-smoke.status"' EXIT
for variant in "${variants[@]}"; do
    while [[ ! -s $run/smoke-$variant.status ]]; do sleep 35; done
    [[ "$(cat "$run/smoke-$variant.status")" == 0 ]]
done
python3 - "$run" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

root=Path(sys.argv[1])
report={}
for name in ('no_memory','long','spatial'):
    train=Path((root/f'smoke-{name}.training-run').read_text().strip())
    loss=[float(row['loss']) for row in csv.DictReader((train/'loss.csv').open())]
    width=min(20,len(loss)//2)
    start=statistics.fmean(loss[:width]) if width else None
    end=statistics.fmean(loss[-width:]) if width else None
    episode,=csv.DictReader((root/f'smoke-{name}-closed-loop/episodes.csv').open())
    config=json.loads((train/'training_config.json').read_text())
    checks=dict(loss_declined=width>0 and all(map(math.isfinite,loss)) and end<start,
                valid_decoded_steps=int(episode['decoded_steps']), steps=int(episode['steps']),
                deaths=int(episode['deaths']), pose_jump=episode['pose_jump']=='True')
    checks['passed']=(checks['loss_declined'] and 0<checks['valid_decoded_steps']<=checks['steps']
                      and checks['deaths']==0 and not checks['pose_jump'])
    report[name]=dict(early_mean_loss=start, late_mean_loss=end, window=width,
                      quantized=config['quantized'], training=json.loads((train/'summary.json').read_text()),
                      episode=episode,checks=checks)
(root/'smoke-summary.json').write_text(json.dumps(report,indent=2)+'\n')
if not all(r['checks']['passed'] for r in report.values()):
    raise SystemExit('Smoke checks require review')
PY
printf '%s Smoke checks passed\n' "$(date -Is)"
mkdir -p "$public"
gates="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
seeds="$WORK/results/phase1_t1_seeds.json"
service_pids=()
clean_services() {
    for pid in "${service_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${service_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    service_pids=()
}
status_and_cleanup() {
    local status=$?
    clean_services
    printf '%s\n' "$status" >"$run/after-smoke.status"
}
trap status_and_cleanup EXIT
trap 'exit 143' TERM INT
for gpu in 6 7; do
    envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$gpu"
done
bash code/minemem/phase1_jarvis.sh pair 6 18716 18726 "$public/service-6" >"$public/service-6.log" 2>&1 &
service_pids+=("$!")
bash code/minemem/phase1_jarvis.sh pair 7 18717 18727 "$public/service-7" >"$public/service-7.log" 2>&1 &
service_pids+=("$!")
ports=(18716 18726 18717 18727)
deadline=$((SECONDS + 1200))
for port in "${ports[@]}"; do
    until curl -fs --max-time 2 "http://127.0.0.1:$port/health"; do
        for pid in "${service_pids[@]}"; do kill -0 "$pid"; done
        if ((SECONDS >= deadline)); then exit 1; fi
        sleep 4
    done
done
pilot() {
    local seed="$1" port="$2" result="$public/seed-$1"
    bash code/minemem/t1_exec.sh protected-pilot "$result" --fixtures "$gates" \
        --seeds-file "$seeds" --seed "$seed" --models jarvis7b \
        --base-url "http://127.0.0.1:$port/v1" >"$result.log" 2>&1
}
children=()
declare -A worker_ports
for i in 0 1 2 3; do
    pilot "$((2026112100+i))" "${ports[i]}" &
    children+=("$!")
    worker_ports[$!]="${ports[i]}"
done
finished=
status=0
wait -n -p finished "${children[@]}" || status=1
[[ -n $finished ]]
pilot 2026112104 "${worker_ports[$finished]}" || status=1
for pid in "${children[@]}"; do
    if [[ $pid != "$finished" ]]; then wait "$pid" || status=1; fi
done
clean_services
[[ $status == 0 ]]
export CUDA_VISIBLE_DEVICES= MPLCONFIGDIR="$WORK/.cache/matplotlib"
envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_respawn_report.py pilot "$public/report" \
    --seeds-file "$seeds" --models random steve1 vpt jarvis7b \
    --inputs "$WORK"/results/t1-respawn-tTWwXxTb/pilot-*/cliff_T1.csv "$public"/seed-*/cliff_T1.csv
printf '%s Public pilot finished\n' "$(date -Is)"
model="$(cat "$run/smoke-no_memory.training-run")"
bash code/minemem/t1_exec.sh eval 6 "$run/navigation-gate" \
    --mode nav --training-run "$model" >"$run/navigation-gate.log" 2>&1
python3 - "$run" <<'PY'
from pathlib import Path
import csv,json,sys
root=Path(sys.argv[1])
rows=list(csv.DictReader((root/'navigation-gate/episodes.csv').open()))
assert len(rows)==10 and {int(r['seed']) for r in rows}==set(range(2030990000,2030990010))
assert all(r['mode']=='nav' and r['model']=='no_memory' and int(r['d'])==5 for r in rows)
successes=sum(r['success']=='True' for r in rows)
result=dict(successes=successes,n=len(rows),required_successes=8,passed=successes>=8)
(root/'navigation-gate/summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not result['passed']:
    raise SystemExit('No-memory navigation gate did not pass')
PY
