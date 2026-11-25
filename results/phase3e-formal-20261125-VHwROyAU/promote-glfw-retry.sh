#!/usr/bin/env bash
set -euo pipefail
run="$1"
export WORK="${run%/results/*}"
while [[ ! -s "$run/glfw-retry.status" ]]; do sleep 5; done
[[ $(cat "$run/glfw-retry.status") == 0 ]]
[[ ! -e "$run/acceptance" ]]
python3 - "$run" <<'PY'
import csv, os, pathlib, signal, sys
run=pathlib.Path(sys.argv[1]); active=run/'acceptance-revised'
rows=[]
for i in range(30):
    path=active/f'worker-{i:03d}'/'acceptance.csv'
    with path.open(newline='') as source:
        items=list(csv.DictReader(source))
    if len(items)!=24 or any(row['error'] for row in items):
        raise SystemExit(f'Incomplete original seed output in {path}')
    rows.extend(items)
assert len(rows)==720
if not (active/'worker-025').is_symlink() or not (run/'glfw-retry.status').read_text().strip()=='0':
    raise SystemExit('The successful retry did not replace the failed raw shard')
if (run/'stage-a.progress').read_text().strip()!='cpu_acceptance_revised_assignments':
    raise SystemExit('The earlier launcher advanced, leaving it in control')
root=int((run/'stage-a.pid').read_text())
expected=[b'bash',os.fsencode(run/'assignment-revision.sh'),os.fsencode(run)]
cmd=pathlib.Path(f'/proc/{root}/cmdline')
if not cmd.exists() or cmd.read_bytes().split(b'\0')[:3]!=expected:
    raise SystemExit('Unexpected earlier CPU retry launcher identity')
parents={}
for stat in pathlib.Path('/proc').glob('[0-9]*/stat'):
    try:
        parents[int(stat.parent.name)]=int(stat.read_text().rsplit(')',1)[1].split()[1])
    except (OSError,ValueError,IndexError):
        pass
children=set(); wave={root}
while wave:
    wave={pid for pid,parent in parents.items() if parent in wave}
    children.update(wave)
if not any(os.fsencode(run/'acceptance-retry') in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
           for pid in children if pathlib.Path(f'/proc/{pid}/cmdline').is_file()):
    raise SystemExit('No redundant retry found, leaving the earlier launcher in control')
# Pause the old coordinator until all of its duplicate retry descendants are signalled.
os.kill(root,signal.SIGSTOP)
try:
    for pid in children:
        try: os.kill(pid,signal.SIGTERM)
        except ProcessLookupError: pass
    os.kill(root,signal.SIGTERM)
finally:
    os.kill(root,signal.SIGCONT)
print('Complete validated 720-row acceptance reused the earlier successful retry',flush=True)
PY
mkdir "$run/acceptance"
for original in "$run/acceptance-revised"/worker-*; do
    ln -s "$original" "$run/acceptance/${original##*/}"
done
printf '0\n' > "$run/acceptance/exit_status"
printf '%s\n' "$$" > "$run/stage-a.pid"
bash "$run/snapshot/code/minemem/t1_phase3e_stage_a.sh" "$run" --reuse-acceptance
