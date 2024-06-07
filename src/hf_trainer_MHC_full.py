import random


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig
import json

from data import load_train_test, load_train_val, SelectModelInputs, BIOTagger, RandomlyUKNTokens, EvaluationDataCollator, RandomlyReplaceTokens, TrainDataCollator
from trainer import NERTrainer
from model import MultiHeadCRF

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from utils import setup_wandb, create_config

import os
import argparse

from metrics import NERMetrics


from time import sleep

# sleep(180)

parser = argparse.ArgumentParser(description="")
parser.add_argument("checkpoint", type=str)
parser.add_argument("--number_of_layer_per_head", type=int, default=1)
parser.add_argument("--percentage_tags", type=float, default=0.2)
parser.add_argument("--augmentation", type=str, default=None)
parser.add_argument("--aug_prob", type=float, default=0.5)
parser.add_argument("--context", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument('--val', action='store_true')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--classes", nargs='+', default=['SYMPTOM', 'PROCEDURE', 'DISEASE', 'PROTEIN', 'CHEMICAL'])


args = parser.parse_args()

model_checkpoint = args.checkpoint#"pubmed_bert_classifier_V2_synthetic/checkpoint-29268"

name = model_checkpoint.split("/")[0]

if args.val:
    val="val"
else:
    val="full"

if args.augmentation is not None:
    dir_name = f"trained-models/{name}-C{args.context}-H{args.number_of_layer_per_head}-E{args.epochs}-A{args.augmentation}-%{args.percentage_tags}-P{args.aug_prob}-{args.random_seed}-{val}"
else:
    dir_name = f"trained-models/{name}-C{args.context}-H{args.number_of_layer_per_head}-E{args.epochs}-{args.random_seed}-{val}"


classes = args.classes

# setup_wandb(dir_name, f"BioCreative-SYMPTEMIST-{len(classes)}-class-test")
training_args = create_config("roberta_trainer_config.yaml", 
                              output_dir=dir_name,
                              num_train_epochs=args.epochs,
                              dataloader_num_workers=4,
                              per_device_train_batch_size=args.batch,
                              evaluation_strategy="steps",
                            #   eval_steps=2,
                            #   eval_steps=None,
                              logging_steps=10,
                              logging_strategy="steps",
                              #gradient_accumulation_steps= 2, # batch 16 - 32 -64
                              per_device_eval_batch_size= args.batch*2,
                              prediction_loss_only=False,
                              seed=args.random_seed,
                              data_seed=args.random_seed)

#Best_model:
#    metric_for_best_model: eval_macroF1
#    greater_is_better: True

#model_checkpoint = "pubmed_bert_classifier_V2_synthetic/checkpoint-32490"
#model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

random.seed(args.random_seed)

CONTEXT_SIZE = args.context

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.model_max_length = 512

transforms = [BIOTagger(), SelectModelInputs()]

train_augmentation = None
if args.augmentation:
    if args.augmentation=="unk": 
        print("Note: The trainer will use RandomlyUKNTokens augmentation")
        train_augmentation = [RandomlyUKNTokens(tokenizer=tokenizer, 
                            context_size=CONTEXT_SIZE,
                            prob_change=args.aug_prob, 
                            percentage_changed_tags=args.percentage_tags)]
    elif args.augmentation=="random":
        print("Note: The trainer will use RandomlyReplaceTokens augmentation")
        train_augmentation = [RandomlyReplaceTokens(tokenizer=tokenizer, 
                            context_size=CONTEXT_SIZE,
                            prob_change=args.aug_prob, 
                            percentage_changed_tags=args.percentage_tags)]
    

if args.val:
    train_ds, test_ds = load_train_val(tokenizer=tokenizer,
                                          context_size=CONTEXT_SIZE,
                                          test_split_percentage=0.33,
                                          train_transformations=transforms,
                                          train_augmentations=train_augmentation,
                                          test_transformations=None)
else:
    train_ds, test_ds = load_train_test(tokenizer=tokenizer,
                                          context_size=CONTEXT_SIZE,
                                          train_transformations=transforms,
                                          train_augmentations=train_augmentation,
                                          test_transformations=None,
                                          entity = classes)

id2label = {0:"O", 1:"B", 2:"I"}
label2id = {v:k for k,v in id2label.items()}
config = AutoConfig.from_pretrained(model_checkpoint)

config.classes = classes
config.id2label = id2label
config.label2id = label2id
config.vocab_size = tokenizer.vocab_size
config.number_of_layer_per_head = args.number_of_layer_per_head
config.args_random_seed = args.random_seed

config.augmentation = args.augmentation
config.context_size = args.context
config.percentage_tags = args.percentage_tags
config.aug_prob = args.aug_prob


config.freeze = False
config.crf_reduction = "mean"

# def model_init():
#     return MultiHeadCRF(config=config)

training_args.eval_steps = len(train_ds)//training_args.per_device_train_batch_size*training_args.num_train_epochs//5
print("STEPS", training_args.eval_steps)

#training_args.save_steps = training_args.eval_steps

trainer = NERTrainer(
    model=MultiHeadCRF(config=config),
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=TrainDataCollator(tokenizer=tokenizer, 
                                                     padding="longest",
                                                     label_pad_token_id=tokenizer.pad_token_id),
    eval_data_collator=EvaluationDataCollator(tokenizer=tokenizer, 
                                              padding=True,
                                              label_pad_token_id=tokenizer.pad_token_id),
    compute_metrics=NERMetrics(context_size=CONTEXT_SIZE)
    
)
# input_ids, attention_mask, labels

# decode(labels) -> true spans
# decode(predicted) -> 


trainer.train()
