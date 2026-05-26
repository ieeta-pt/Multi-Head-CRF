#!/bin/bash
#SBATCH --job-name=ner_en           
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1         
#SBATCH --gpus-per-task=l40s:1      
#SBATCH --cpus-per-task=8           
#SBATCH --mem-per-gpu=64G           
#SBATCH --output=logs/%x_%A.out  

source ../.venv/bin/activate

# MODELS=(
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" #0.73
#     "emilyalsentzer/Bio_ClinicalBERT"                      #0.7181
#     "dmis-lab/biobert-base-cased-v1.2"                     #0.7151
# )

# EPOCHS=30; BATCH=64; CONTEXT=64; HEAD_LAYERS=3; AUGMENTATION="random"; AUG_PROB=0.5; PERC_TAGS=0.25; LANG="en"

# for ckpt in "${MODELS[@]}"; do
#     echo "=========================================================="
#     echo "🚀 Training ${LANG^^} with $ckpt"
#     python hf_trainer.py "$ckpt" --language "$LANG" --val --augmentation $AUGMENTATION --number_of_layer_per_head $HEAD_LAYERS --context $CONTEXT --epochs $EPOCHS --batch $BATCH --percentage_tags $PERC_TAGS --aug_prob $AUG_PROB --classes SYMPTOM PROCEDURE DISEASE
# done

# MODELS=(
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" 
#     "emilyalsentzer/Bio_ClinicalBERT"
# )
# CONTEXTS=(32 64)
# LAYERS=(1 3)

# EPOCHS=60; BATCH=64; AUGMENTATION="random"; AUG_PROB=0.5; PERC_TAGS=0.25; LANG="en"

# for ckpt in "${MODELS[@]}"; do
#     for ctx in "${CONTEXTS[@]}"; do
#         for hl in "${LAYERS[@]}"; do
#             echo "=========================================================="
#             echo "🚀 Training ${LANG^^} | Model: $ckpt | Ctx: $ctx | Layers: $hl"
#             python hf_trainer.py "$ckpt" --language "$LANG" --val --augmentation $AUGMENTATION --number_of_layer_per_head $hl --context $ctx --epochs $EPOCHS --batch $BATCH --percentage_tags $PERC_TAGS --aug_prob $AUG_PROB --classes SYMPTOM PROCEDURE DISEASE
#         done
#     done
# done

# EPOCHS=30; BATCH=64; LANG="en"; CTX=64
# AUGS=("none" "random" "ukn")
# PROBS=(0.2 0.5)
# PERCS=(0.1 0.25)

# # Array format: "Model|HeadLayers"
# MODELS=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|1" "emilyalsentzer/Bio_ClinicalBERT|3")

# for item in "${MODELS[@]}"; do
#     MODEL="${item%%|*}"
#     HL="${item##*|}"
    
#     for aug in "${AUGS[@]}"; do
#         if [ "$aug" == "none" ]; then
#             echo "=========================================================="
#             echo "🚀 EN | $MODEL | Aug: NONE"
#             python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation none --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#             continue
#         fi
#         for prob in "${PROBS[@]}"; do
#             for perc in "${PERCS[@]}"; do
#                 echo "=========================================================="
#                 echo "🚀 EN | $MODEL | Aug: $aug | Prob: $prob | Perc: $perc"
#                 python hf_trainer.py "$MODEL" --language "$LANG" --val --augmentation "$aug" --aug_prob $prob --percentage_tags $perc --number_of_layer_per_head $HL --context $CTX --epochs $EPOCHS --batch $BATCH --classes SYMPTOM PROCEDURE DISEASE
#             done
#         done
#     done
# done


# LANG="en"
# EPOCHS=60
# BATCH=64

# # Top 10 configurations for EN (PubMedBERT swept the board!)
# CONFIGS=(
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|random|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|3|random|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.5|0.1"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|random|0.2|0.1"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|random|0.5|0.1"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|32|3|random|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.2|0.1"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.2|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|none|0.5|0.2"
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

# LANG="en"
# EPOCHS=30
# BATCH=64

# # The 4 random seeds to generate your 20 models
# SEEDS=(42 123 456 999)

# # Paste your Top 5 configs for the language here (ES Example)
# CONFIGS=(
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|3|ukn|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|3|random|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|3|random|0.5|0.1"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|3|none|0.5|0.2"
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|3|ukn|0.2|0.25"
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



# LANG="en"
# EPOCHS=30
# BATCH=32

# # Top 10 configurations for EN (PubMedBERT swept the board!)
# CONFIGS=(
#     "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|1|random|0.5|0.25"
#     "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|3|random|0.5|0.25"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.5|0.1"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.5|0.25"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|random|0.2|0.1"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|random|0.5|0.1"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|32|3|random|0.5|0.25"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.2|0.1"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|ukn|0.2|0.25"
# #     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract|64|1|none|0.5|0.2"
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


LANG="en"
EPOCHS=30
BATCH=32

# The 4 random seeds to generate your 20 models
SEEDS=(42 123 456 999)

# Paste your Top 5 configs for the language here (ES Example)
CONFIGS=(
    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|3|ukn|0.5|0.25"
    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|3|random|0.5|0.25"
    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|3|random|0.5|0.1"
    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|3|none|0.5|0.2"
    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract|64|3|ukn|0.2|0.25"
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
