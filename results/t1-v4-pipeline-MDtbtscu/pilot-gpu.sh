#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
p="$1"
acceptance="$2"
pairs=()
cleanup() {
    local status=$?
    for pid in "${pairs[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    printf '%s\n' "$status" >"$p/pilot_gpu_status"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
until [[ -s $p/shortcut_recheck_status ]]; do sleep 30; done
[[ "$(cat "$p/shortcut_recheck_status")" == 0 ]]
idle() {
    envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$1" >/dev/null
}
until idle 6 && idle 7; do
    printf '%s waiting for both GPUs\n' "$(date -Is)"
    sleep 60
done
ports=(18716 18726 18717 18727)
frozen="$WORK/results/phase1_t1_seeds.json"
bash code/minemem/phase1_jarvis.sh pair 6 18716 18726 "$p/jarvis-service-gpu6" >"$p/jarvis-service-gpu6.log" 2>&1 &
pairs+=("$!")
bash code/minemem/phase1_jarvis.sh pair 7 18717 18727 "$p/jarvis-service-gpu7" >"$p/jarvis-service-gpu7.log" 2>&1 &
pairs+=("$!")
deadline=$((SECONDS + 1200))
for port in "${ports[@]}"; do
    until curl -fs --max-time 2 "http://127.0.0.1:$port/health"; do
        for pid in "${pairs[@]}"; do kill -0 "$pid"; done
        if ((SECONDS >= deadline)); then exit 1; fi
        sleep 5
    done
done
pilot() {
    local seed="$1" port="$2" path="$p/jarvis-$1"
    bash code/minemem/t1_exec.sh pilot "$path" --acceptance "$acceptance" \
        --seeds-file "$frozen" --seed "$seed" --models jarvis7b \
        --base-url "http://127.0.0.1:$port/v1" >"$path.log" 2>&1
}
children=()
for ((i=0; i<4; i++)); do
    pilot "$((2026112100 + i))" "${ports[i]}" &
    children+=("$!")
done
status=0
wait "${children[0]}" || status=1
pilot 2026112104 "${ports[0]}" || status=1
for pid in "${children[@]:1}"; do wait "$pid" || status=1; done
exit "$status"
