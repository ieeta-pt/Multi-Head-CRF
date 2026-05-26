import os
import click
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from transformers import AutoTokenizer
from data import CorpusTokenizer, CorpusDataset, CorpusPreProcessor, BIOTagger, SelectModelInputs, EvaluationDataCollator
from corpus import Spanish_Biomedical_NER_Corpus
from torch.utils.data import DataLoader
from decoder import decoder
from model.configuration_multiheadcrf import MultiHeadCRFConfig

# Import your dynamic model classes
from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel

def f1PR(tp, fn, fp):
    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = 0 if tp == 0 else tp / (tp + fn)
    f1 = 0 if precision == 0 and recall == 0 else (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

@click.command()
@click.argument("checkpoint")
@click.option("--val_tsv", default="../dataset/processed_multilingual/mixed/val_mixed.tsv")
@click.option("--docs_dir", default="../dataset/processed_multilingual/documents-all")
def main(checkpoint, val_tsv, docs_dir):
    print(f"{'='*70}\n🌍 MULTILINGUAL PER-LANGUAGE EVALUATION\n{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = 512

    config = MultiHeadCRFConfig.from_pretrained(checkpoint)
    base_name = getattr(config, "_name_or_path", checkpoint).lower()

    if "roberta" in base_name or "roberta" in checkpoint.lower():
        model = RobertaMultiHeadCRFModel.from_pretrained(checkpoint, config=config)
    else:
        model = BertMultiHeadCRFModel.from_pretrained(checkpoint, config=config)
    
    model = model.to(device)
    model.eval()
    
    # We must use the exact order of entities as defined in the model config
    entities = sorted(config.classes)
    context_size = config.context_size

    print("Loading Validation Dataset...")
    valCorpus = Spanish_Biomedical_NER_Corpus(val_tsv, docs_dir)
    valProcessor = CorpusPreProcessor(valCorpus)
    valProcessor.merge_annoatation()
    valProcessor.filter_labels(entities)
    
    transforms = [BIOTagger(), SelectModelInputs()]
    tokenized_val = CorpusTokenizer(valProcessor, tokenizer, context_size)
    val_ds = CorpusDataset(tokenized_corpus=tokenized_val, transforms=transforms)   
    
    # val_ds = [val_ds[i] for i in range(1000)]
    # tokenized_val = [tokenized_val[i] for i in range(1000)]
    
    # shuffle=False is critical so DataLoader order matches tokenized_val exactly
    eval_datacollator = EvaluationDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=tokenizer.pad_token_id)
    dl = DataLoader(val_ds, batch_size=32, collate_fn=eval_datacollator, shuffle=False)

    print("Running Inference...")
    all_preds = {ent: [] for ent in entities}
    
    for eval_batch in tqdm(dl, desc="Predicting"):
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in eval_batch["inputs"].items()}
            _output = model(**inputs)

            for i, ent_label in enumerate(entities):
                # NO ARGMAX! Pass the raw arrays exactly as the model outputs them.
                all_preds[ent_label].extend(_output[i].cpu().numpy())

    print("Decoding Spans and Calculating Metrics...")
    documents = defaultdict(dict)
    doc_gs = defaultdict(lambda: defaultdict(set))
    
    for i in range(len(tokenized_val)):
        raw_item = tokenized_val[i]
        
        doc_id = raw_item["doc_id"]
        seq_id = raw_item["sequence_id"]
        offsets = raw_item["offsets"]
        annots = raw_item.get("list_annotations", {})

        for ent in entities:
            if ent in annots:
                for ann in annots[ent]:
                    doc_gs[doc_id][ent].add((ann['start_span'], ann['end_span']))
                        
        documents[doc_id][seq_id] = {'offsets': offsets}
        for ent in entities:
            # NO SLICING! Pass the full padded arrays exactly as NERMetrics does.
            documents[doc_id][seq_id][ent] = all_preds[ent][i]

    lang_metrics = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    
    for doc_id in documents.keys():
        lang = str(doc_id).split('-')[0].upper()
        
        for ent in entities:
            current_doc = [documents[doc_id][seq][ent] for seq in sorted(documents[doc_id].keys())]
            current_offsets = [documents[doc_id][seq]['offsets'] for seq in sorted(documents[doc_id].keys())]
            
            # Exact 1-to-1 match to your NERMetrics call
            decoded = decoder(current_doc, current_offsets, padding=context_size)
            
            # Safely extract span predictions
            pred_spans = set(decoded["span"]) if (isinstance(decoded, dict) and "span" in decoded) else set()
            true_spans = doc_gs[doc_id][ent]
            
            tp = len(true_spans.intersection(pred_spans))
            fn = len(true_spans.difference(pred_spans))
            fp = len(pred_spans.difference(true_spans))
            
            lang_metrics[lang][ent]["tp"] += tp
            lang_metrics[lang][ent]["fp"] += fp
            lang_metrics[lang][ent]["fn"] += fn

    print(f"\n{'='*70}\n📊 PER-LANGUAGE EVALUATION RESULTS\n{'='*70}")
    
    for lang in sorted(lang_metrics.keys()):
        print(f"\n🌍 Language: {lang}")
        print("-" * 50)
        print(f"{'Entity':<15} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
        print("-" * 50)
        
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for ent in entities:
            tp = lang_metrics[lang][ent]["tp"]
            fp = lang_metrics[lang][ent]["fp"]
            fn = lang_metrics[lang][ent]["fn"]
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            f1, p, r = f1PR(tp, fn, fp)
            print(f"{ent:<15} | {f1:.4f}     | {p:.4f}     | {r:.4f}")
            
        overall_f1, overall_p, overall_r = f1PR(total_tp, total_fn, total_fp)
        print("-" * 50)
        print(f"{'OVERALL':<15} | {overall_f1:.4f}     | {overall_p:.4f}     | {overall_r:.4f}\n")

if __name__ == '__main__':
    main()