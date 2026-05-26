import click
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Import your custom modules
from corpus import Spanish_Biomedical_NER_Corpus_Inference
from data import CorpusTokenizer, CorpusDataset, CorpusPreProcessor, EvaluationDataCollator
from model.configuration_multiheadcrf import MultiHeadCRFConfig
from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel
from decoder import decoder
from torch.utils.data import DataLoader

def decoder_from_samples(prediction_batch, context_size, entities):
    documents = {}
    padding = context_size

    for i in range(len(prediction_batch)):
        doc_id = prediction_batch[i]['doc_id'].split('/')[-1]
        if doc_id not in documents:
            documents[doc_id] = {}

        documents[doc_id][prediction_batch[i]['sequence_id']] = {
            **{"output_"+k: prediction_batch[i]["output_"+k] for k in entities},
            'offsets': prediction_batch[i]['offsets'],
            'text': prediction_batch[i]["text"]
        }

    predicted_entities = {}
    for doc in documents.keys():
        text = documents[doc][0]["text"]
        predicted_entities[doc] = {
            "document_text": text,
            "doc_id": doc
        }
        for entity in entities:
            current_doc = [documents[doc][seq]["output_"+entity] for seq in sorted(documents[doc].keys())]
            current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
            predicted_entities[doc][entity] = decoder(current_doc, current_offsets, padding=padding, text=text)
    return predicted_entities

def remove_txt(data):
    new_data = {}
    for k, v in data.items():
        new_k, _ = os.path.splitext(k)
        new_data[new_k] = v
    return new_data

@click.command()
@click.argument("checkpoint")
@click.option("--docs_dir", required=True, help="Path to the folder containing raw .txt documents")
@click.option("--out_folder", default="predictions_test")
def main(checkpoint, docs_dir, out_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 Enable cuDNN auto-tuner for dynamic L40S pathway optimization
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"⚙️ Loading tokenizer and config from {checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = 512
    config = MultiHeadCRFConfig.from_pretrained(checkpoint)

    # 🚨 FOOLPROOF ARCHITECTURE DETECTION 🚨
    checkpoint_lower = str(checkpoint).lower()
    roberta_keywords = ["roberta", "ufal", "pdelobelle", "cltl", "vesteinn", "xlm", "campillos", "plantl"]
    is_roberta = any(keyword in checkpoint_lower for keyword in roberta_keywords)

    if is_roberta:
        print(f"🧠 Detected RoBERTa architecture")
        model = RobertaMultiHeadCRFModel.from_pretrained(checkpoint, config=config)
    else:
        print(f"🧠 Detected BERT architecture")
        model = BertMultiHeadCRFModel.from_pretrained(checkpoint, config=config)

    model = model.to(device)
    model.eval()

    # # 🔥 NEW: Compile the model specifically for the L40S architecture
    # if hasattr(torch, "compile") and device == "cuda":
    #     print("⚡ Compiling model for L40S... (This takes ~1 minute upfront, but flies afterward)")
    #     model = torch.compile(model)

    _entities = sorted(config.classes)
    print(f"🏷️ Detecting entities: {_entities}")

    print(f"📂 Loading documents from {docs_dir}...")
    testSpanishCorpus = Spanish_Biomedical_NER_Corpus_Inference(docs_dir, entities=_entities)

    testSpanishCorpusProcessor = CorpusPreProcessor(testSpanishCorpus)
    tokenized_test_corpus = CorpusTokenizer(testSpanishCorpusProcessor, tokenizer, config.context_size)

    test_ds = CorpusDataset(tokenized_corpus=tokenized_test_corpus)

    eval_datacollator = EvaluationDataCollator(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id
    )

    dl = DataLoader(
        test_ds, 
        batch_size=128,      # 🔥 CRANKED UP from 64 to 256
        collate_fn=eval_datacollator, 
        shuffle=False,
        num_workers=12,      # Wakes up your CPU cores
        pin_memory=True      # Speeds up transfer to GPU
    )

    print("🚀 Running Inference...")
    outputs = []
    
    for eval_batch in tqdm(dl, desc="Predicting"):
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in eval_batch["inputs"].items()}
            
            # 🔥 Switched to bfloat16 (Native to L40S / Ada Lovelace architecture)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _output = model(**inputs)

            # Pre-transfer the outputs to the CPU as numpy arrays ONCE per batch
            batch_outputs = {}
            for i, ent_label in enumerate(_entities):
                # If compiled, sometimes outputs are nested or slightly different, ensuring smooth CPU transfer
                batch_outputs[f"output_{ent_label}"] = _output[i].cpu().numpy()

        # 🔥 Fast extraction loop bypassing slow torch.is_tensor type-checks
        for i in range(len(eval_batch["doc_id"])):
            item = {
                "doc_id": eval_batch["doc_id"][i],
                "sequence_id": eval_batch["sequence_id"][i],
                "offsets": eval_batch["offsets"][i],
                "text": eval_batch["text"][i]
            }
            
            for ent_label in _entities:
                item[f"output_{ent_label}"] = batch_outputs[f"output_{ent_label}"][i]
                
            outputs.append(item)

    print("🧩 Reconstructing and Decoding Documents...")
    docs = decoder_from_samples(outputs, context_size=config.context_size, entities=_entities)
    docs = remove_txt(docs)

    print("💾 Saving Predictions...")
    fOut_name = os.path.basename(os.path.normpath(checkpoint))
    os.makedirs(out_folder, exist_ok=True)

    data = []
    for doc_id, doc in docs.items():
        for entity_type in _entities:
            # Safely get span to avoid KeyErrors if the model predicted nothing for a document
            for span in doc[entity_type].get("span", []):
                data.append({
                    "filename": doc_id,
                    "label": entity_type,
                    "strat_span": span[0], 
                    "end_span": span[1],
                    "text": doc["document_text"][span[0]:span[1]],
                })

    out_df = pd.DataFrame(data)
    out_path = os.path.join(out_folder, f"{fOut_name}.tsv")

    # Handle empty predictions gracefully
    if out_df.empty:
        print("⚠️ WARNING: No entities predicted! Saving empty TSV with headers.")
        out_df = pd.DataFrame(columns=["filename", "label", "strat_span", "end_span", "code", "text"])

    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"✅ Saved predictions to {out_path}")

if __name__ == '__main__':
    main()