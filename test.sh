#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

datasets=(mnist fashion_mnist cifar10)
models=(linear mlp cnn gru transformer)

# Reuse the environment that evaluated LSTM; refuse CPU fallback.
uv run --no-sync python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Use the NVIDIA environment used for LSTM.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
PY

# Check every required file before replacing any evaluation results.
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for required in \
            "configs/image_classification/$dataset/$model.yaml" \
            "checkpoints/${dataset}_${model}_seed42/best.pt" \
            "results/a1/${dataset}_${model}_seed42/training_summary.json"; do
            if [[ ! -f "$required" ]]; then
                printf "Missing file: %s\n" "$required" >&2
                exit 1
            fi
        done
    done
done

completed=0
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        completed=$((completed + 1))
        printf "\n===== TEST %d/15: %s / %s =====\n" "$completed" "$dataset" "$model"
        uv run --no-sync python -m scripts.evaluation.evaluate \
            --data "$dataset" \
            --config "configs/image_classification/$dataset/$model.yaml"
    done
done

printf "\nCompleted all 15 evaluations. Results: results/a1/<run-name>/\n"
