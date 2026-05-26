#!/bin/bash
#SBATCH --job-name=ner_es           
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1         
#SBATCH --gpus-per-task=l40s:1      
#SBATCH --cpus-per-task=8           
#SBATCH --mem-per-gpu=64G           
#SBATCH --output=logs/%x_%A.out  

source ../.venv/bin/activate

# MODELS=(
#     "lcampillos/roberta-es-clinical-trials-ner" #0.79925
#     "PlanTL-GOB-ES/bsc-bio-ehr-es"              #0.79368
#     "dccuchile/bert-base-spanish-wwm-uncased"   #0.76727
# )

# EPOCHS=30
# BATCH=64
# CONTEXT=64
# HEAD_LAYERS=3
# AUGMENTATION="random"
# AUG_PROB=0.5
# PERC_TAGS=0.25
# LANG="es"

# for ckpt in "${MODELS[@]}"; do
#     echo "=========================================================="
#     echo "🚀 STARTING TRAINING RUN"
#     echo "🌍 Language:   ${LANG^^}"
#     echo "🧠 Checkpoint: $ckpt"
#     echo "=========================================================="
    
#     python hf_trainer.py "$ckpt" \
#         --language "$LANG" \
#         --val \
#         --augmentation $AUGMENTATION \
#         --number_of_layer_per_head $HEAD_LAYERS \
#         --context $CONTEXT \
#         --epochs $EPOCHS \
#         --batch $BATCH \
#         --percentage_tags $PERC_TAGS \
#         --aug_prob $AUG_PROB \
#         --classes SYMPTOM PROCEDURE DISEASE

#     echo "✅ Finished training for $LANG: $ckpt"
#     echo ""
# done
# echo "🎉 ALL ES MODELS COMPLETED!"


# MODELS=(
#     "lcampillos/roberta-es-clinical-trials-ner" 
#     "PlanTL-GOB-ES/bsc-bio-ehr-es"
# )
# CONTEXTS=(32 64)
# LAYERS=(1 3)

# EPOCHS=60; BATCH=64; AUGMENTATION="random"; AUG_PROB=0.5; PERC_TAGS=0.25; LANG="es"

# for ckpt in "${MODELS[@]}"; do
#     for ctx in "${CONTEXTS[@]}"; do
#         for hl in "${LAYERS[@]}"; do
#             echo "=========================================================="
#             echo "🚀 Training ${LANG^^} | Model: $ckpt | Ctx: $ctx | Layers: $hl"
#             python hf_trainer.py "$ckpt" --language "$LANG" --val --augmentation $AUGMENTATION --number_of_layer_per_head $hl --context $ctx --epochs $EPOCHS --batch $BATCH --percentage_tags $PERC_TAGS --aug_prob $AUG_PROB --classes SYMPTOM PROCEDURE DISEASE
#         done
#     done
# done

# EPOCHS=30; BATCH=64; LANG="es"; CTX=64; HL=3
# MODELS=("lcampillos/roberta-es-clinical-trials-ner" "PlanTL-GOB-ES/bsc-bio-ehr-es")

# AUGS=("none" "random" "ukn")
# PROBS=(0.2 0.5)
# PERCS=(0.1 0.25)

# for MODEL in "${MODELS[@]}"; do
#     for aug in "${AUGS[@]}"; do
#         if [ "$aug" == "none" ]; then
#             echo "=========================================================="
#             echo "🚀 ES | $MODEL | Aug: NONE"
#             python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation none --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#             continue
#         fi
#         for prob in "${PROBS[@]}"; do
#             for perc in "${PERCS[@]}"; do
#                 echo "=========================================================="
#                 echo "🚀 ES | $MODEL | Aug: $aug | Prob: $prob | Perc: $perc"
#                 python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation "$aug" --aug_prob $prob --percentage_tags $perc --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#             done
#         done
#     done
# done


# LANG="es"
# EPOCHS=60
# BATCH=64

# # Top 10 configurations for ES
# CONFIGS=(
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|random|0.5|0.25"
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.5|0.1"
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|random|0.2|0.1"
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|random|0.5|0.1"
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.5|0.25"
#     "lcampillos/roberta-es-clinical-trials-ner|64|1|random|0.5|0.25"
#     "lcampillos/roberta-es-clinical-trials-ner|32|1|random|0.5|0.25"
#     "lcampillos/roberta-es-clinical-trials-ner|32|3|random|0.5|0.25"
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.2|0.1"
#     "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.2|0.25"
# )

# echo "=========================================================="
# echo "🚀 TRAINING TOP 10 ENSEMBLE MODELS FOR: ${LANG^^} (60 EPOCHS)"
# echo "=========================================================="

# for item in "${CONFIGS[@]}"; do
#     MODEL=$(echo "$item" | cut -d'|' -f1)
#     CTX=$(echo "$item" | cut -d'|' -f2)
#     HL=$(echo "$item" | cut -d'|' -f3)
#     AUG=$(echo "$item" | cut -d'|' -f4)
#     PROB=$(echo "$item" | cut -d'|' -f5)
#     PERC=$(echo "$item" | cut -d'|' -f6)

#     echo "----------------------------------------------------------"
#     if [ "$AUG" == "none" ]; then
#         echo "⏳ $MODEL | Ctx:$CTX | Lyr:$HL | Aug:NONE"
#         python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation none --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#     else
#         echo "⏳ $MODEL | Ctx:$CTX | Lyr:$HL | Aug:$AUG | P:$PROB | %:$PERC"
#         python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation "$AUG" --aug_prob $PROB --percentage_tags $PERC --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#     fi
# done



LANG="es"
EPOCHS=60
BATCH=32

# The 4 random seeds to generate your 20 models
SEEDS=(42 123 456 999)

# Paste your Top 5 configs for the language here (ES Example)
CONFIGS=(
    "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.2|0.25"
    "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.5|0.1"
    "lcampillos/roberta-es-clinical-trials-ner|64|3|random|0.2|0.1"
    "lcampillos/roberta-es-clinical-trials-ner|64|3|ukn|0.5|0.25"
    "lcampillos/roberta-es-clinical-trials-ner|64|3|random|0.5|0.1"
)

echo "=========================================================="
echo "🚀 FULL DATASET TRAINING: ${LANG^^} (Top 5 Configs x 4 Seeds)"
echo "=========================================================="

for seed in "${SEEDS[@]}"; do
    for item in "${CONFIGS[@]}"; do
        MODEL=$(echo "$item" | cut -d'|' -f1)
        CTX=$(echo "$item" | cut -d'|' -f2)
        HL=$(echo "$item" | cut -d'|' -f3)
        AUG=$(echo "$item" | cut -d'|' -f4)
        PROB=$(echo "$item" | cut -d'|' -f5)
        PERC=$(echo "$item" | cut -d'|' -f6)

        echo "----------------------------------------------------------"
        if [ "$AUG" == "none" ]; then
            echo "⏳ $MODEL | Seed:$seed | Ctx:$CTX | Lyr:$HL | Aug:NONE"
            python hf_trainer.py "$MODEL" --language "$LANG" --random_seed $seed --augmentation none --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
        else
            echo "⏳ $MODEL | Seed:$seed | Ctx:$CTX | Lyr:$HL | Aug:$AUG | P:$PROB | %:$PERC"
            python hf_trainer.py "$MODEL" --language "$LANG" --random_seed $seed --augmentation "$AUG" --aug_prob $PROB --percentage_tags $PERC --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
        fi
    done
done


