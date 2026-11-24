import argparse,csv,json,math
from collections import Counter
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("directory",type=Path)
parser.add_argument("fixtures",type=Path)
parser.add_argument("--missing",action="store_true")
parser.add_argument("--smoke",action="store_true")
args=parser.parse_args()
frozen=json.loads(args.fixtures.read_text())["formal"]
if args.smoke:
    frozen=frozen[:1]
    folders=[args.directory/"smoke"]
else:
    folders=[args.directory/"full",args.directory/"retry"]
rows={}
for directory in folders:
    for path in sorted(directory.glob("worker-*/pose-controls.csv")):
        with path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                if not row["error"]:
                    rows.setdefault((int(row["seed"]),int(row["d"]),row["control"]),row)
modes=("start_pose","reverse_poses")
distances=(30,100)
needed={(seed,distance,mode) for seed in frozen for distance in distances for mode in modes}
missing=needed-rows.keys()
if args.missing:
    print(*(i for i,seed in enumerate(frozen) if any(item[0]==seed for item in missing)))
    raise SystemExit(0)
if missing:
    raise SystemExit(f"Missing valid cases: {sorted(missing)}")
counts=Counter()
for (seed,d,mode),row in rows.items():
    counts[d,mode]+=row["success"]=="True"
result={str(d):{mode:dict(successes=counts[d,mode],episodes=len(frozen)) for mode in modes} for d in distances}
if not args.smoke:
    first=next(iter(rows.values()))
    with (args.directory/"episodes.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(first))
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(needed))
name="smoke-summary.json" if args.smoke else "summary.json"
with (args.directory/name).open("x") as f:
    json.dump(result,f,indent=2);f.write("\n")
print(json.dumps(result))
