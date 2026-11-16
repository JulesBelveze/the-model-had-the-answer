#!/bin/sh
# Fetch the files from the authors' repo that these scripts expect. They are not redistributed here.
set -e
git clone --depth 1 --filter=blob:none --sparse https://github.com/RikiyaT/lm_understand-logs upstream
git -C upstream sparse-checkout set exp8_math/code exp8_math/eval/p50c exp8_math/runs/p50c/edit exp8_math/runs/p50c/requests
mkdir -p their_code data logs edit
cp upstream/exp8_math/code/mathqa.py their_code/
printf 'def loop_signals(text):\n    return {"repetitive_end": False}\n' > their_code/mcqa.py  # stub for an import mathqa.py makes
cp -r upstream/exp8_math/eval/p50c logs/
cp upstream/exp8_math/runs/p50c/edit/edit.npz upstream/exp8_math/runs/p50c/edit/manifest.json edit/
cp upstream/exp8_math/runs/p50c/requests/fwd_4.json.claimed data/fwd_4.json
