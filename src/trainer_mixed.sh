#!/bin/bash
#SBATCH --job-name=ner_mixed           
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1         
#SBATCH --gpus-per-task=a40:1      
#SBATCH --cpus-per-task=8           
#SBATCH --mem-per-gpu=64G           
#SBATCH --output=logs/%x_%A.out  

source ../.venv/bin/activate

# MODELS=(
#     "xlm-roberta-base"                     #0.72196
#     "bert-base-multilingual-cased"         
#     "distilbert-base-multilingual-cased"   
# )

# EPOCHS=30
# BATCH=64
# CONTEXT=64
# HEAD_LAYERS=3
# AUGMENTATION="random"
# AUG_PROB=0.5
# PERC_TAGS=0.25
# LANG="mixed"

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
# echo "🎉 ALL MIXED MODELS COMPLETED!"


# EPOCHS=30; BATCH=64; LANG="mixed"
# MODEL="xlm-roberta-base"; CTX=64; HL=3

# AUGS=("none" "random" "ukn")
# PROBS=(0.2 0.5)
# PERCS=(0.1 0.25)

# for aug in "${AUGS[@]}"; do
#     if [ "$aug" == "none" ]; then
#         echo "=========================================================="
#         echo "🚀 MIXED | $MODEL | Aug: NONE"
#         python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation none --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#         continue
#     fi
#     for prob in "${PROBS[@]}"; do
#         for perc in "${PERCS[@]}"; do
#             echo "=========================================================="
#             echo "🚀 MIXED | $MODEL | Aug: $aug | Prob: $prob | Perc: $perc"
#             python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation "$aug" --aug_prob $prob --percentage_tags $perc --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#         done
#     done
# done

# LANG="mixed"
# EPOCHS=10
# BATCH=64

# # Top 9 unique configurations tested for MIXED
# CONFIGS=(
#     "xlm-roberta-base|64|3|random|0.5|0.25"
#     "xlm-roberta-base|64|3|random|0.2|0.25"
#     "xlm-roberta-base|64|3|ukn|0.5|0.25"
#     "xlm-roberta-base|64|3|ukn|0.5|0.1"
#     "xlm-roberta-base|64|3|random|0.2|0.1"
#     "xlm-roberta-base|64|3|ukn|0.2|0.1"
#     "xlm-roberta-base|64|3|none|0.5|0.2"
#     "xlm-roberta-base|64|3|random|0.5|0.1"
#     "xlm-roberta-base|64|3|ukn|0.2|0.25"
# )

# echo "=========================================================="
# echo "🚀 TRAINING ENSEMBLE MODELS FOR: ${LANG^^} (10 EPOCHS)"
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





# LANG="mixed"
# EPOCHS=10
# BATCH=64

# # The 4 random seeds to generate your 20 models
# SEEDS=(42 123 456 999)

# # Paste your Top 5 configs for the language here (ES Example)
# CONFIGS=(
#     "xlm-roberta-base|64|3|ukn|0.2|0.25"
#     "xlm-roberta-base|64|3|random|0.2|0.25"
#     "xlm-roberta-base|64|3|ukn|0.5|0.25"
#     "xlm-roberta-base|64|3|ukn|0.2|0.1"
#     # "xlm-roberta-large|64|3|ukn|0.2|0.25"  # 🔥 Added the Large variant using your best hyperparams!
# )

# echo "=========================================================="
# echo "🚀 FULL DATASET TRAINING: ${LANG^^} (Top 5 Configs x 4 Seeds)"
# echo "=========================================================="

# for seed in "${SEEDS[@]}"; do
#     for item in "${CONFIGS[@]}"; do
#         MODEL=$(echo "$item" | cut -d'|' -f1)
#         CTX=$(echo "$item" | cut -d'|' -f2)
#         HL=$(echo "$item" | cut -d'|' -f3)
#         AUG=$(echo "$item" | cut -d'|' -f4)
#         PROB=$(echo "$item" | cut -d'|' -f5)
#         PERC=$(echo "$item" | cut -d'|' -f6)

#         echo "----------------------------------------------------------"
#         if [ "$AUG" == "none" ]; then
#             echo "⏳ $MODEL | Seed:$seed | Ctx:$CTX | Lyr:$HL | Aug:NONE"
#             python hf_trainer.py "$MODEL" --language "$LANG" --random_seed $seed --augmentation none --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#         else
#             echo "⏳ $MODEL | Seed:$seed | Ctx:$CTX | Lyr:$HL | Aug:$AUG | P:$PROB | %:$PERC"
#             python hf_trainer.py "$MODEL" --language "$LANG" --random_seed $seed --augmentation "$AUG" --aug_prob $PROB --percentage_tags $PERC --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#         fi
#     done
# done


LANG="mixed"
EPOCHS=3
BATCH=16

# The 4 random seeds to generate your 20 models
SEEDS=(42 123 456 999)

# Paste your Top 5 configs for the language here (ES Example)
CONFIGS=(
    "FacebookAI/xlm-roberta-large|64|3|ukn|0.2|0.25"
    "FacebookAI/xlm-roberta-large|64|3|random|0.2|0.25"
    "FacebookAI/xlm-roberta-large|64|3|ukn|0.5|0.25"
    "FacebookAI/xlm-roberta-large|64|3|ukn|0.2|0.1"
    # "xlm-roberta-large|64|3|ukn|0.2|0.25"  # 🔥 Added the Large variant using your best hyperparams!
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


# LANG="mixed"
# EPOCHS=10
# BATCH=16

# # Top 9 unique configurations tested for MIXED
# CONFIGS=(
#     "FacebookAI/xlm-roberta-large|64|3|random|0.5|0.25"
#     "FacebookAI/xlm-roberta-large|64|3|random|0.2|0.25"
#     "FacebookAI/xlm-roberta-large|64|3|ukn|0.5|0.25"
#     "FacebookAI/xlm-roberta-large|64|3|ukn|0.5|0.1"
#     "FacebookAI/xlm-roberta-large|64|3|random|0.2|0.1"
#     # "xlm-roberta-base|64|3|ukn|0.2|0.1"
#     # "xlm-roberta-base|64|3|none|0.5|0.2"
#     # "xlm-roberta-base|64|3|random|0.5|0.1"
#     # "xlm-roberta-base|64|3|ukn|0.2|0.25"
# )

# echo "=========================================================="
# echo "🚀 TRAINING ENSEMBLE MODELS FOR: ${LANG^^} (10 EPOCHS)"
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

