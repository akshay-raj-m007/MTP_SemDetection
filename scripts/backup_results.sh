#!/usr/bin/env bash
# Collect every result of this machine into one archive, and optionally push it to the gpu-results branch.
#
#   bash scripts/backup_results.sh           # makes ~/ssl30_results_<time>.tgz and shows its size, pushes nothing
#   bash scripts/backup_results.sh --push    # the same, then commits the files to gpu-results and pushes
#
# Kept: configs, logs, metrics, summaries, bootstrap files, out-of-fold and test predictions, review pictures.
# Left out (large or rebuildable): model weights, per-epoch prediction dumps, prepared datasets, Python environments.
# gpu-results keeps the layout used before: gpuresults/outputs/... and gpuresults/*.log
# No file over 90 MB is pushed (GitHub refuses 100 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
ARCHIVE="$HOME/ssl30_results_$(date +%Y%m%d_%H%M).tgz"

tar -czf "$ARCHIVE" \
  --exclude='epoch_predictions' --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' --exclude='*.ckpt' \
  --exclude='prepared' --exclude='__pycache__' --exclude='env' --exclude='.venv' --exclude='venv' \
  outputs $(ls ./*.log 2>/dev/null | sed 's#^\./##' || true)

echo "Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1), $(tar -tzf "$ARCHIVE" | wc -l) entries)"
echo "Largest files:"; tar -tvzf "$ARCHIVE" | sort -k3 -n -r | head -8 | awk '{printf "  %8.1f MB  %s\n", $3/1e6, $6}'

if [ "$PUSH" -eq 0 ]; then
  echo "Nothing pushed. Copy it to your PC with:  scp <user>@<this machine>:$ARCHIVE .   or rerun with --push."
  exit 0
fi

WT="$(mktemp -d)"
git fetch origin gpu-results
git worktree add -f "$WT" -B gpu-results-backup origin/gpu-results
mkdir -p "$WT/gpuresults"
tar -xzf "$ARCHIVE" -C "$WT/gpuresults"
BIG="$(find "$WT/gpuresults" -type f -size +90M)"
if [ -n "$BIG" ]; then echo "Refusing to push, files over 90 MB:"; echo "$BIG"; exit 1; fi
cd "$WT"
git add -f gpuresults
git commit -m "Add ssl30 results: RetinaNet and SSD weighted cross-validation, final tests, bootstraps"
git push origin HEAD:gpu-results
cd - >/dev/null
git worktree remove --force "$WT"
echo "BACKUP PUSHED to origin/gpu-results"
