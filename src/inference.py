import click

from utils import load_model, load_model_and_tokenizer, load_model_local
from data import load_train_test, BIOTagger, SelectModelInputs, EvaluationDataCollator, DocumentReaderDatasetForTraining
from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForMaskedLM,AutoModel
import transformers
from decoder import decoder
from torch.utils.data import DataLoader
import torch
import pandas as pd
from collections import defaultdict 

import os
import json




def decoder_from_samples(prediction_batch, context_size, entities):
    documents = {}
    padding = context_size

    # reconsturct the document in the correct order
    for i in range(len(prediction_batch)):
        doc_id = prediction_batch[i]['doc_id'].split('/')[-1]
        
        if doc_id not in documents.keys():
            documents[doc_id] = {}
            
            # run 1 time is enough for this stuff

        documents[doc_id][prediction_batch[i]['sequence_id']] = {
            **{"output_"+k: prediction_batch[i]["output_"+k] for k in entities},
            'offsets': prediction_batch[i]['offsets'],
            'text':prediction_batch[i]["text"]}
        
    print("DOCUMENTS:", len(documents))
            
    predicted_entities = {}
    # decode each set of labels and store the offsets
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
    for k,v in data.items():
        new_k, _ = os.path.splitext(k)
        new_data[new_k]=v
        
    return new_data

@click.command()
@click.option("--checkpoint")
@click.option("--out_folder", default="runs")
def main(checkpoint, out_folder):
    
    
    if torch.cuda.is_available():
        #single GPU bc CRF
        assert torch.cuda.device_count()==1
    
    model, tokenizer, config = load_model_local(checkpoint)
    model = model.to(f"cuda")
    tokenizer.model_max_length = 512
    
    _entities = sorted(['SINTOMA' , 'PROCEDIMIENTO', 'ENFERMEDAD', 'PROTEINAS', 'CHEMICAL'])
    
    
    _, test_ds = load_train_test(tokenizer=tokenizer,
                                          context_size=config.context_size,
                                          train_transformations=None,
                                          train_augmentations=None,
                                          test_transformations=None,
                                          entity = _entities)

    
    eval_datacollator = EvaluationDataCollator(tokenizer=tokenizer, 
                                            padding=True,
                                            label_pad_token_id=tokenizer.pad_token_id)

    dl = DataLoader(test_ds, batch_size=8, collate_fn=eval_datacollator)

    outputs = []
    for eval_batch in dl:
        with torch.no_grad():
            _output = model(**eval_batch["inputs"].to("cuda"))

            for i,ent_label in enumerate(_entities):
                eval_batch[f"output_{ent_label}"] = _output[i]
                
            eval_batch |= eval_batch["inputs"]
            del eval_batch["inputs"]
        keys = list(eval_batch.keys())
        
        for i in range(len(eval_batch["doc_id"])):
            outputs.append({k:eval_batch[k][i] for k in keys})


    docs = decoder_from_samples(outputs, context_size=config.context_size, entities=_entities)
    docs = remove_txt(docs)
    
    
    fOut_name = "-".join(checkpoint.split("/")[-2:])
    
    data = []
    for doc_id, doc in docs.items():
        for entity_type in _entities:
            for span in doc[entity_type]["span"]:
                data.append({
                    "filename": doc_id,
                    "strat_span": span[0],
                    "end_span": span[1],
                    "label": entity_type,
                    "code": "-",
                    "text": doc["document_text"][span[0]:span[1]],
                })
            
    out_df = pd.DataFrame(data)
    
    out_df.to_csv(os.path.join(out_folder,f"{fOut_name}.tsv"), sep="\t", index=False)
    
if __name__ == '__main__':
    main()