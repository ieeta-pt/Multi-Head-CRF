import random
import os
import argparse

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig
import json
from data import CorpusTokenizer, CorpusDataset, CorpusPreProcessor, BIOTagger, SelectModelInputs, RandomlyUKNTokens, EvaluationDataCollator, RandomlyReplaceTokens, TrainDataCollator
from corpus import Spanish_Biomedical_NER_Corpus # Assuming this works for all our standard TSVs!
from trainer import NERTrainer

from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel
from model.configuration_multiheadcrf import MultiHeadCRFConfig

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from utils import setup_wandb, create_config
from metrics import NERMetrics
from time import sleep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Head CRF NER Trainer")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--language", type=str, required=True, help="Target language (e.g., es, nl, cz, mixed)")
    parser.add_argument("--number_of_layer_per_head", type=int, default=1)
    parser.add_argument("--percentage_tags", type=float, default=0.2)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument('--val', action='store_true', help="If set, uses train/val split and computes metrics. If omitted, trains on FULL dataset without validation.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--classes", nargs='+', default=['SYMPTOM', 'PROCEDURE', 'DISEASE'])

    args = parser.parse_args()

    model_checkpoint = args.checkpoint
    name = model_checkpoint.split("/")[0] +"-"+model_checkpoint.split("/")[1]
    lang = args.language

    # WandB Naming
    val_str = "val" if args.val else "full"
    
    if args.augmentation is not None:
        run_name = f"{name}-C{args.context}-H{args.number_of_layer_per_head}-E{args.epochs}-A{args.augmentation}-%{args.percentage_tags}-P{args.aug_prob}-{args.random_seed}"
    else:
        run_name = f"{name}-C{args.context}-H{args.number_of_layer_per_head}-E{args.epochs}-{args.random_seed}"
        
    dir_name = f"trained-models/{lang.upper()}/{val_str}/{run_name}"
    setup_wandb(name=run_name, project=f"MultiClinAI-{lang.upper()}-{val_str}")

    classes = args.classes

    # Dynamic Evaluation Strategy
    eval_strategy = "steps" if args.val else "no"

    training_args = TrainingArguments(
        output_dir=dir_name,
        num_train_epochs=args.epochs,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        prediction_loss_only=False,
        logging_steps=10,
        logging_first_step=True,
        logging_strategy="steps",
        seed=args.random_seed,
        data_seed=args.random_seed,
        eval_steps=None, # Will be calculated dynamically below if args.val is True
        save_steps=99999, # Will be updated dynamically below
        save_strategy="steps",
        save_total_limit=1,
        evaluation_strategy=eval_strategy,
        warmup_ratio=0.1,
        learning_rate=2e-5, # 4e-5
        weight_decay=0.01,
        push_to_hub=False,
        report_to="wandb",
        fp16=True,
        fp16_full_eval=False
    )

    random.seed(args.random_seed)
    CONTEXT_SIZE = args.context

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_length = 512

    transforms = [BIOTagger(), SelectModelInputs()]

    train_augmentation = None
    if args.augmentation:
        if args.augmentation == "unk": 
            print("Note: The trainer will use RandomlyUKNTokens augmentation")
            train_augmentation = [RandomlyUKNTokens(tokenizer=tokenizer, context_size=CONTEXT_SIZE, prob_change=args.aug_prob, percentage_changed_tags=args.percentage_tags)]
        elif args.augmentation == "random":
            print("Note: The trainer will use RandomlyReplaceTokens augmentation")
            train_augmentation = [RandomlyReplaceTokens(tokenizer=tokenizer, context_size=CONTEXT_SIZE, prob_change=args.aug_prob, percentage_changed_tags=args.percentage_tags)]

    # ==========================================
    # DATASET LOADING (Language-Aware)
    # ==========================================
    # Pointing to the new processed_multilingual directory
    base_data_dir = f"../dataset/processed_multilingual/{lang}"
    docs_dir = f"../dataset/processed_multilingual/documents-{lang}"
    
    if lang == "mixed":
        docs_dir = "../dataset/processed_multilingual/documents-all"

    if args.val:
        print(f"📊 Mode: VALIDATION. Training on train_{lang}.tsv and evaluating on val_{lang}.tsv")
        train_tsv = os.path.join(base_data_dir, f"train_{lang}.tsv")
        val_tsv = os.path.join(base_data_dir, f"val_{lang}.tsv")

        # Load Train
        trainCorpus = Spanish_Biomedical_NER_Corpus(train_tsv, docs_dir)
        trainProcessor = CorpusPreProcessor(trainCorpus)
        trainProcessor.merge_annoatation()
        trainProcessor.filter_labels(classes)
        tokenized_train = CorpusTokenizer(trainProcessor, tokenizer, CONTEXT_SIZE)
        train_ds = CorpusDataset(tokenized_corpus=tokenized_train, transforms=transforms, augmentations=train_augmentation)

        # Load Val
        testCorpus = Spanish_Biomedical_NER_Corpus(val_tsv, docs_dir)
        testProcessor = CorpusPreProcessor(testCorpus)
        testProcessor.merge_annoatation()
        testProcessor.filter_labels(classes)
        tokenized_test = CorpusTokenizer(testProcessor, tokenizer, CONTEXT_SIZE)
        test_ds = CorpusDataset(tokenized_corpus=tokenized_test)

    else:
        print(f"🚀 Mode: FULL PRODUCTION. Training on full_{lang}.tsv. Validation and metrics are DISABLED.")
        full_tsv = os.path.join(base_data_dir, f"full_{lang}.tsv")

        trainCorpus = Spanish_Biomedical_NER_Corpus(full_tsv, docs_dir)
        trainProcessor = CorpusPreProcessor(trainCorpus)
        trainProcessor.merge_annoatation()
        trainProcessor.filter_labels(classes)
        tokenized_train = CorpusTokenizer(trainProcessor, tokenizer, CONTEXT_SIZE)
        train_ds = CorpusDataset(tokenized_corpus=tokenized_train, transforms=transforms, augmentations=train_augmentation)

        test_ds = None # No validation dataset

    # ==========================================
    # MODEL SETUP
    # ==========================================
    id2label = {0:"O", 1:"B", 2:"I"}
    label2id = {v:k for k,v in id2label.items()}
    config = MultiHeadCRFConfig.from_pretrained(model_checkpoint,
                                                classes=args.classes,
                                                number_of_layer_per_head=args.number_of_layer_per_head,
                                                id2label=id2label,
                                                label2id=label2id,
                                                augmentation=args.augmentation,
                                                context_size=args.context,
                                                percentage_tags=args.percentage_tags,
                                                aug_prob=args.aug_prob,
                                                freeze=False,
                                                crf_reduction="mean")

    base_hf_config = AutoConfig.from_pretrained(model_checkpoint)
    
    # 2. Load the correct MultiHeadCRF wrapper
    if "roberta" in base_hf_config.model_type:
        print(f"🧠 Detected {base_hf_config.model_type} architecture. Loading RobertaMultiHeadCRFModel...")
        model = RobertaMultiHeadCRFModel.from_pretrained(model_checkpoint, config=config)
        # Resize on the backbone directly
        if hasattr(model, 'roberta'):
            model.roberta.resize_token_embeddings(len(tokenizer))
            
    elif "bert" in base_hf_config.model_type:
        print(f"🧠 Detected {base_hf_config.model_type} architecture. Loading BertMultiHeadCRFModel...")
        model = BertMultiHeadCRFModel.from_pretrained(model_checkpoint, config=config)
        # Resize on the backbone directly
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model type: {base_hf_config.model_type}. Please use a BERT or RoBERTa based model.")

    
    model.training_mode()

    # Dynamic Steps Calculation
    steps_per_epoch = len(train_ds) // training_args.per_device_train_batch_size
    eval_and_save_steps = max(1, (steps_per_epoch * training_args.num_train_epochs) // 5)
    
    if args.val:
        training_args.eval_steps = eval_and_save_steps
        print("STEPS (Eval/Save):", training_args.eval_steps)
    else:
        print("STEPS (Save Only):", eval_and_save_steps)
        
    training_args.save_steps = eval_and_save_steps

    # ==========================================
    # TRAINER INITIALIZATION
    # ==========================================
    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds if args.val else None,
        tokenizer=tokenizer,
        data_collator=TrainDataCollator(tokenizer=tokenizer, padding="longest", label_pad_token_id=tokenizer.pad_token_id),
        
        # Only attach evaluation collator and metrics if we are actually evaluating
        eval_data_collator=EvaluationDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=tokenizer.pad_token_id) if args.val else None,
        compute_metrics=NERMetrics(context_size=CONTEXT_SIZE) if args.val else None
    )

    trainer.train()