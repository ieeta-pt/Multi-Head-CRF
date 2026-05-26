
import os
import click
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import load_model, load_model_and_tokenizer, load_model_local
from corpus import Spanish_Biomedical_NER_Corpus_Inference
from data import CorpusTokenizer, CorpusDataset, CorpusPreProcessor, SelectModelInputs, EvaluationDataCollator
from model.configuration_multiheadcrf import MultiHeadCRFConfig
from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel
from decoder import decoder

@click.command()
@click.argument("checkpoint")
@click.option("--docs_dir", default="../dataset/documents")
@click.option("--out_folder", default="predictions")
def main(checkpoint, docs_dir, out_folder):
    print(f"🚀 Initializing Inference for: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dynamically Load the Model and Tokenizer
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

    entities = config.classes
    context_size = config.context_size

    # 2. Prepare the Dataset
    print("Loading Inference Dataset...")
    testCorpus = Spanish_Biomedical_NER_Corpus_Inference(docs_dir, entities=entities)
    testProcessor = CorpusPreProcessor(testCorpus)
    
    transforms = [SelectModelInputs()] # No BIOTagger needed for inference!
    tokenized_test = CorpusTokenizer(testProcessor, tokenizer, context_size)
    test_ds = CorpusDataset(tokenized_corpus=tokenized_test, transforms=transforms)   
    
    eval_datacollator = EvaluationDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=tokenizer.pad_token_id)
    # shuffle=False is critical so DataLoader order matches tokenized_test exactly
    dl = DataLoader(test_ds, batch_size=32, collate_fn=eval_datacollator, shuffle=False)

    # 3. Fast GPU Inference
    print("Running Inference...")
    all_preds = {ent: [] for ent in entities}

    for eval_batch in tqdm(dl, desc="Predicting"):
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in eval_batch["inputs"].items()}
            _output = model(**inputs)

            for i, ent_label in enumerate(entities):
                # Detach from GPU to prevent OOM errors!
                all_preds[ent_label].extend(_output[i].cpu().numpy())

    # 4. Reconstruct Documents safely bypassing the DataCollator
    print("Decoding Spans...")
    documents = defaultdict(dict)

    for i in range(len(tokenized_test)):
        raw_item = tokenized_test[i]
        
        doc_id = raw_item["doc_id"].split('/')[-1]
        seq_id = raw_item["sequence_id"]
        offsets = raw_item["offsets"]
        text = raw_item["text"]
                        
        documents[doc_id][seq_id] = {'offsets': offsets, 'text': text}
        for ent in entities:
            documents[doc_id][seq_id][ent] = all_preds[ent][i]

    # 5. Format and Save Output TSV
    print("Formatting Output...")
    os.makedirs(out_folder, exist_ok=True)
    data = []
    
    for doc_id in documents.keys():
        # Get full text string from the document
        doc_text = documents[doc_id][0]["text"]
        
        for ent in entities:
            current_doc = [documents[doc_id][seq][ent] for seq in sorted(documents[doc_id].keys())]
            current_offsets = [documents[doc_id][seq]['offsets'] for seq in sorted(documents[doc_id].keys())]
            
            decoded = decoder(current_doc, current_offsets, padding=context_size, text=doc_text)
            
            if "span" in decoded:
                for start_span, end_span in decoded["span"]:
                    data.append({
                        "filename": os.path.splitext(doc_id)[0],
                        "strat_span": start_span,
                        "end_span": end_span,
                        "label": ent,
                        "code": "-",
                        "text": doc_text[start_span:end_span],
                    })
            
    out_df = pd.DataFrame(data)
    
    # Save with the name of the checkpoint folder
    fOut_name = checkpoint.strip("/").split("/")[-1]
    out_path = os.path.join(out_folder, f"{fOut_name}.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"✅ Saved {len(out_df)} entities to {out_path}\n")

if __name__ == '__main__':
    main()