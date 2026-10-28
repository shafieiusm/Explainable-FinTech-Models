#!/usr/bin/env bash
# Reproduce every table and figure of the paper end-to-end.
set -euo pipefail
CFG=${CFG:-configs/default.yaml}
RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['output_dir'])")

python scripts/train.py        --config "$CFG"            # test metrics + fairness
python scripts/explain.py      --run "$RUN"               # Figures 4–6
python scripts/lending_demo.py --run "$RUN"               # Figures 7–8
python scripts/run_ablation.py --config "$CFG"            # Table 2
python scripts/run_cv.py       --config "$CFG"            # Table 3
python scripts/make_figures.py --results "$RUN/results.json" --data "$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['data']['path'])")" --out "$RUN/figures"
echo "done → $RUN/figures"
