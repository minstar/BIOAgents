#!/bin/bash
# Watch for ablation checkpoint-50 saves and trigger merge+eval
# Each ablation trains on a specific GPU; we reuse that GPU for eval since
# the training continues past checkpoint-50 and we can't double-load.
# Instead, we'll queue evals and run them on whichever GPU finishes eval first.

cd /data/project/private/minstar/workspace/BIOAgents

declare -A ABLATIONS
ABLATIONS[grpo_baseline]="checkpoints/grpo_baseline_lingshu7b/checkpoint-50"
ABLATIONS[dapo_only]="checkpoints/dapo_only_lingshu7b/checkpoint-50"
ABLATIONS[gspo_only]="checkpoints/gspo_only_lingshu7b/checkpoint-50"
ABLATIONS[drgrpo]="checkpoints/drgrpo_lingshu7b/checkpoint-50"

declare -A DONE

echo "[$(date)] Watching for ablation checkpoint-50 saves..."

while true; do
    ALL_DONE=true
    for name in "${!ABLATIONS[@]}"; do
        ckpt="${ABLATIONS[$name]}"
        if [ -z "${DONE[$name]}" ] && [ -d "$ckpt" ]; then
            # Check if adapter_model.safetensors exists (save complete)
            if [ -f "$ckpt/adapter_model.safetensors" ]; then
                echo "[$(date)] FOUND: $name checkpoint-50 saved!"
                DONE[$name]=1
                
                # Determine merged dir
                merged="${ckpt}-merged"
                results="results/algorithm_comparison/${name}"
                
                # Merge LoRA (use CPU to avoid GPU contention)
                if [ ! -d "$merged" ]; then
                    echo "[$(date)] Merging $name..."
                    .venv/bin/python scripts/merge_lora.py \
                        --base-model checkpoints/sft_warmup_lingshu7b_v2_merged/merged \
                        --lora-path "$ckpt" \
                        --output-dir "$merged" 2>&1 | tail -3
                    cp checkpoints/sft_warmup_lingshu7b_v2_merged/merged/preprocessor_config.json "$merged/" 2>/dev/null || true
                    echo "[$(date)] Merged: $merged"
                fi
            fi
        fi
        [ -z "${DONE[$name]}" ] && ALL_DONE=false
    done
    
    $ALL_DONE && break
    sleep 30
done

echo "[$(date)] All 4 ablation checkpoint-50s merged!"
echo "Ready for evaluation. Merged dirs:"
for name in "${!ABLATIONS[@]}"; do
    echo "  - ${ABLATIONS[$name]}-merged"
done
