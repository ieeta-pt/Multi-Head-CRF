import random


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig
import json
from data import Spanish_Biomedical_NER_Corpus, CorpusTokenizer,CorpusDataset, CorpusPreProcessor ,BIOTagger, SelectModelInputs,RandomlyUKNTokens, EvaluationDataCollator, RandomlyReplaceTokens, TrainDataCollator
from trainer import NERTrainer

from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel
from model.configuration_multiheadcrf import MultiHeadCRFConfig

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from utils import setup_wandb, create_config

import os
import argparse

from metrics import NERMetrics


from time import sleep

# sleep(180)
if __name__ == "__main__":
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

    training_args = TrainingArguments(output_dir=dir_name,
                                        num_train_epochs=args.epochs,
                                        dataloader_num_workers=1,
                                        dataloader_pin_memory=True,
                                        per_device_train_batch_size=args.batch,
                                        #gradient_accumulation_steps= 2, # batch 16 - 32 -64
                                        per_device_eval_batch_size= args.batch*2,
                                        prediction_loss_only=False,
                                        logging_steps = 10,
                                        logging_first_step = True,
                                        logging_strategy = "steps",
                                        seed=args.random_seed,
                                        data_seed=args.random_seed,
                                        eval_steps=None,#0.1, # 
                                        save_steps=99999, # this is changed latter in the code below
                                        save_strategy="steps",
                                        save_total_limit=1,
                                        evaluation_strategy="steps",
                                        warmup_ratio = 0.1,
                                        learning_rate=2e-5,
                                        weight_decay=0.01,
                                        push_to_hub=False,
                                        report_to="none",
                                        fp16=True,
                                        fp16_full_eval=False)

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
        
        # First create a generic Corpus
        spanishCorpus = Spanish_Biomedical_NER_Corpus("../dataset/merged_data_subtask1_train.tsv","../dataset/documents" )
        #  create a Corpus PreProcessor, which handles certain preprocessing : merge_annoatation, filter_labels, split_data
        spanishCorpusProcessor = CorpusPreProcessor(spanishCorpus)
        spanishCorpusProcessor.merge_annoatation()
        spanishCorpusProcessor.filter_labels(classes)
        train_corpus, test_corpus = spanishCorpusProcessor.split_data(0.33)
        #  create a CorpusTokenizer, using the CorpusProcessor, which internally tokenizes the dataset and splits documents
        tokenized_train_corpus = CorpusTokenizer(train_corpus, tokenizer, CONTEXT_SIZE)
        # finally create the dataset by applying transformations, of which order is important (BioTagging should be run first)
        train_ds = CorpusDataset(tokenized_corpus=tokenized_train_corpus, transforms=transforms, augmentation=train_augmentation)

        tokenized_test_corpus = CorpusTokenizer(test_corpus, tokenizer, CONTEXT_SIZE)
        test_ds = CorpusDataset(tokenized_corpus=tokenized_test_corpus)
        
        

    else:
        trainSpanishCorpus = Spanish_Biomedical_NER_Corpus("../dataset/merged_data_subtask1_train.tsv","../dataset/documents" )
        trainSpanishCorpusProcessor = CorpusPreProcessor(trainSpanishCorpus)
        trainSpanishCorpusProcessor.merge_annoatation()
        trainSpanishCorpusProcessor.filter_labels(classes)
        tokenized_train_corpus = CorpusTokenizer(trainSpanishCorpusProcessor, tokenizer, CONTEXT_SIZE)
        train_ds = CorpusDataset(tokenized_corpus=tokenized_train_corpus, transforms=transforms, augmentations=train_augmentation)

        testSpanishCorpus = Spanish_Biomedical_NER_Corpus("../dataset/merged_data_subtask1_test.tsv","../dataset/documents" )
        testSpanishCorpusProcessor = CorpusPreProcessor(testSpanishCorpus)
        testSpanishCorpusProcessor.merge_annoatation()
        testSpanishCorpusProcessor.filter_labels(classes)
        tokenized_test_corpus = CorpusTokenizer(testSpanishCorpusProcessor, tokenizer, CONTEXT_SIZE)
        test_ds = CorpusDataset(tokenized_corpus=tokenized_test_corpus)


    id2label = {0:"O", 1:"B", 2:"I"}
    label2id = {v:k for k,v in id2label.items()}
    config = MultiHeadCRFConfig.from_pretrained(model_checkpoint,
                                                classes = args.classes,
                                                number_of_layer_per_head = args.number_of_layer_per_head,
                                                id2label = id2label,
                                                label2id = label2id,
                                                augmentation = args.augmentation,
                                                context_size = args.context,
                                                percentage_tags = args.percentage_tags,
                                                aug_prob = args.aug_prob,
                                                freeze = False,
                                                crf_reduction = "mean",
                                                )

    model = RobertaMultiHeadCRFModel.from_pretrained(model_checkpoint, config=config)
    model.training_mode() # fix a stupid bug regarding weight inits

    # def model_init():
    #     return MultiHeadCRF(config=config)

    training_args.eval_steps = len(train_ds)//training_args.per_device_train_batch_size*training_args.num_train_epochs//5
    print("STEPS", training_args.eval_steps)

    training_args.save_steps = training_args.eval_steps

    trainer = NERTrainer(
        model=model,
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
