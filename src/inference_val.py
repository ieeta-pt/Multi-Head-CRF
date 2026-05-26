import click
import os
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer

from corpus import Spanish_Biomedical_NER_Corpus
from data import CorpusTokenizer, CorpusDataset, CorpusPreProcessor, BIOTagger, SelectModelInputs, EvaluationDataCollator
from model.configuration_multiheadcrf import MultiHeadCRFConfig
from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel
from decoder import decoder
from torch.utils.data import DataLoader

@click.command()
@click.argument("checkpoint")
@click.option("--val_tsv", default="../dataset/processed_multilingual/mixed/val_mixed.tsv")
@click.option("--docs_dir", default="../dataset/processed_multilingual/documents-all")
@click.option("--out_folder", default="predictions_val")
def main(checkpoint, val_tsv, docs_dir, out_folder):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = 512

    config = MultiHeadCRFConfig.from_pretrained(checkpoint)
    
    # 🚨 FOOLPROOF ARCHITECTURE DETECTION 🚨
    checkpoint_lower = str(checkpoint).lower()
    roberta_keywords = ["roberta", "ufal", "pdelobelle", "cltl", "vesteinn", "xlm", "campillos", "plantl"]
    is_roberta = any(keyword in checkpoint_lower for keyword in roberta_keywords)
    
    if is_roberta:
        print(f"🧠 Detected RoBERTa architecture for: {checkpoint}")
        model = RobertaMultiHeadCRFModel.from_pretrained(checkpoint, config=config)
    else:
        print(f"🧠 Detected BERT architecture for: {checkpoint}")
        model = BertMultiHeadCRFModel.from_pretrained(checkpoint, config=config)

    model = model.to(device)
    model.eval()
   
    _entities = sorted(config.classes) 
    
    print("Loading Validation Dataset...")
    valCorpus = Spanish_Biomedical_NER_Corpus(val_tsv, docs_dir)
    valProcessor = CorpusPreProcessor(valCorpus)
    valProcessor.merge_annoatation()
    valProcessor.filter_labels(_entities)
    
    transforms = [BIOTagger(), SelectModelInputs()]
    tokenized_val = CorpusTokenizer(valProcessor, tokenizer, config.context_size)
    val_ds = CorpusDataset(tokenized_corpus=tokenized_val, transforms=transforms)   
    
    eval_datacollator = EvaluationDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=tokenizer.pad_token_id)
    dl = DataLoader(val_ds, batch_size=32, collate_fn=eval_datacollator, shuffle=False)

    print("Running Inference...")
    all_preds = {ent: [] for ent in _entities}

    for eval_batch in tqdm(dl, desc="Predicting"):
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in eval_batch["inputs"].items()}
            _output = model(**inputs)

            for i, ent_label in enumerate(_entities):
                all_preds[ent_label].extend(_output[i].cpu().numpy())

    print("Reconstructing Documents...")
    documents = defaultdict(dict)

    for i in range(len(tokenized_val)):
        raw_item = tokenized_val[i]
        
        doc_id = raw_item["doc_id"]
        seq_id = raw_item["sequence_id"]
                        
        if seq_id not in documents[doc_id]:
            documents[doc_id][seq_id] = {}
            
        documents[doc_id][seq_id]['offsets'] = raw_item["offsets"]
        
        for ent in _entities:
            documents[doc_id][seq_id][ent] = all_preds[ent][i]

    print("Decoding Samples...")
    data = []
    os.makedirs(out_folder, exist_ok=True)
    
    for doc_id in documents.keys():
        
        # 🚨 ALWAYS LOAD FULL TEXT FROM DOCUMENT SOURCE ON DISK 🚨
        path1 = os.path.join(docs_dir, f"{doc_id}.txt")
        clean_id = str(doc_id).split('-')[-1] if '-' in str(doc_id) else str(doc_id)
        path2 = os.path.join(docs_dir, f"{clean_id}.txt")
        
        if os.path.exists(path1):
            with open(path1, "r", encoding="utf-8") as f:
                doc_text = f.read()
        elif os.path.exists(path2):
            with open(path2, "r", encoding="utf-8") as f:
                doc_text = f.read()
        else:
            # If it fails, scream loudly in the terminal so we know exactly which file is missing
            print(f"⚠️ WARNING: Could not find text file for {doc_id}! Looked for {path1} and {path2}.")
            doc_text = " " * 500000 
            
        for ent in _entities:
            current_doc = [documents[doc_id][seq][ent] for seq in sorted(documents[doc_id].keys())]
            current_offsets = [documents[doc_id][seq]['offsets'] for seq in sorted(documents[doc_id].keys())]
            
            decoded = decoder(current_doc, current_offsets, padding=config.context_size, text=doc_text)
            
            if isinstance(decoded, dict) and "span" in decoded:
                for start_span, end_span in decoded["span"]:
                    data.append({
                        "filename": doc_id,
                        "label": ent,
                        "strat_span": start_span,
                        "end_span": end_span,
                        "text": doc_text[start_span:end_span],
                    })
            
    out_df = pd.DataFrame(data)
    
    # Clean filename saving (grabbing just the model folder name)
    fOut_name = checkpoint.strip("/").split("/")[-2]
    out_path = os.path.join(out_folder, f"{fOut_name}.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"✅ Saved perfectly formatted predictions to {out_path}")

if __name__ == '__main__':
    main()