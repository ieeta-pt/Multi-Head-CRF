import click
import pandas as pd
from collections import defaultdict
import json
import os
from functools import partial, lru_cache
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel



@click.command()
@click.argument("input_run")
@click.option("--t", default=0.6)
@click.option("--use_gazetteer", is_flag=True)
@click.option("--output_folder", default="runs")
def main(input_run, t, use_gazetteer, output_folder):
    
    run = pd.read_csv(input_run, sep="\t")
    
    codes = defaultdict(set)

    with open("../embeddings/snomedCT.jsonl") as f:   
        for data in map(json.loads, f):
            codes[str(data["id"])].add(data["text"])
            
    USE_LOWER_CASE = True
    MIN_EMB_THREASHOLD = t
    if USE_LOWER_CASE:
    ## LOAD the exact match files
        def transform_entity(x):
            return x.lower()
    else:
        def transform_entity(x):
            return x

    
    def is_float(string):
        value = False
        try:
            float(string)
            value= True
        except ValueError:
            pass
        
        return value
        
    def load_data_train(path):
        
        text_to_codes = defaultdict(set)
        
        df = pd.read_csv(path, sep="\t")
        
        for i, row in df.iterrows():
            if row["label"] != "UNCLEAR":
                text_to_codes[transform_entity(row["text"])].add(row["code"])
            
        return {t:list(c) for t,c in text_to_codes.items()}

    def load_data_train_distemist(path):
        #filename	mark	label	off0	off1	span	code	semantic_rel
        text_to_codes = defaultdict(set)
        
        df = pd.read_csv(path, sep="\t")
        
        for i, row in df.iterrows():
            if row["label"] != "UNCLEAR":
                text_to_codes[transform_entity(row["span"])].add(row["code"])
            
        return {t:list(c) for t,c in text_to_codes.items()}

    def build_direct_lookup_function(data_dict):
        def _lookup(entity):
            entity = transform_entity(entity)
            return data_dict.get(entity, [])
        return _lookup
    
    medprocner_direct_match = load_data_train("../dataset/medprocner/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv")
    symptemist_direct_match = load_data_train("../dataset/symptemist/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete.tsv")
    distemist_direct_match = load_data_train_distemist("../dataset/distemist/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv")
    pharmaconer_direct_match = load_data_train("../dataset/pharmaconer/new_format/pharmaconer_task2_train.tsv")

    medprocner_training_lookup = build_direct_lookup_function(medprocner_direct_match)
    symptemist_training_lookup = build_direct_lookup_function(symptemist_direct_match)
    distemist_training_lookup = build_direct_lookup_function(distemist_direct_match)
    pharmaconer_training_lookup = build_direct_lookup_function(pharmaconer_direct_match)
    

    NORMALIZER_MODEL_MAPPINGS = {
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large": "sapBERT-multilanguage-large",
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "sapBERT-multilanguage",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token": "sapBERT-english",
    }

    NORMALIZER_MODEL_MAPPINGS_REVERSED = {v:k for k,v in NORMALIZER_MODEL_MAPPINGS.items()}

    device = "cuda"

    embeddings= {}
    embeddings_id = {}
    _path = "../embeddings"

    for file_p in filter(lambda x: x.endswith(".jsonl"), os.listdir(_path)):
        with open(os.path.join(_path, file_p),  encoding='utf-8') as f:
            embeddings_id[file_p.split(".")[0]] = [json.loads(x)["id"] for x in f if x != "\n"]

    for file_p in filter(lambda x:x.endswith(".npy"), os.listdir(_path)):
        _embeddings = torch.as_tensor(np.load(os.path.join(_path, file_p))).to(device)
        _embeddings = _embeddings/torch.linalg.norm(_embeddings, ord=2, axis=-1, keepdims=True)
        embeddings[file_p.split("_")[0]] = _embeddings
        

    checkpoint = NORMALIZER_MODEL_MAPPINGS_REVERSED[file_p.split("_")[-1][:-4]]
            
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    model = AutoModel.from_pretrained(checkpoint).to(device)

    @lru_cache(maxsize=1_000_000)
    def embedding_lookup_function(entity, prefix="",threshold=MIN_EMB_THREASHOLD):
   
        entity = transform_entity(entity)
        best_matches_per_embfile = []
        
        for dict_key in embeddings.keys():
            inputs = tokenizer(entity + prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state[:,0,:]#[0].mean(axis=1)
                embedding = embedding/torch.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
                scores = (embeddings[dict_key] @ embedding.T).squeeze()
                max_score, index_max = torch.max(scores, dim=-1)
                max_score = max_score.cpu().numpy().item()
                index_max = index_max.cpu().numpy().item()
                if max_score>threshold:
                    best_matches_per_embfile.append((embeddings_id[dict_key][index_max], max_score))
        
        if len(best_matches_per_embfile)>0:
            return [max(best_matches_per_embfile, key=lambda x:x[1])[0]]
        else:
            return []
    
    def build_embedding_lookup_function(folder_path, prefix="",threshold=MIN_EMB_THREASHOLD):
        
        local_embeddings_id = []
        local_embeddings = []
        
        for file_p in filter(lambda x: x.endswith(".jsonl"), os.listdir(folder_path)):
            with open(os.path.join(folder_path, file_p)) as f:
                local_embeddings_id = [x["id"] for x in map(json.loads, f)]

        for file_p in filter(lambda x:x.endswith(".npy"), os.listdir(folder_path)):
            _embeddings = torch.as_tensor(np.load(os.path.join(folder_path, file_p))).to(device)
            _embeddings = _embeddings/torch.linalg.norm(_embeddings, ord=2, axis=-1, keepdims=True)
            local_embeddings = _embeddings
        
        @lru_cache(maxsize=1_000_000)
        def _lookup(entity):
            entity = transform_entity(entity)

            inputs = tokenizer(entity + prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state[:,0,:]#[0].mean(axis=1)
                embedding = embedding/torch.linalg.norm(embedding, ord=2, axis=-1, keepdims=True)
                scores = (local_embeddings @ embedding.T).squeeze()
                max_score, index_max = torch.max(scores, dim=-1)
                max_score = max_score.cpu().numpy().item()
                index_max = index_max.cpu().numpy().item()
                
                if max_score>threshold:
                    return [local_embeddings_id[index_max]]
                else:
                    return []
        return _lookup
    
    
    medprocner_training_lookup = build_direct_lookup_function(medprocner_direct_match)
    symptemist_training_lookup = build_direct_lookup_function(symptemist_direct_match)
    distemist_training_lookup = build_direct_lookup_function(distemist_direct_match)
    pharmaconer_training_lookup = build_direct_lookup_function(pharmaconer_direct_match)
    
    
    medprocner_training_semantic_lookup = build_embedding_lookup_function("../dataset/medprocner/medprocner_gazetteer/", threshold=t)
    symptemist_training_semantic_lookup = build_embedding_lookup_function("../dataset/symptemist/symptemist_gazetteer/", threshold=t)
    distemist_training_semantic_lookup = build_embedding_lookup_function("../dataset/distemist/distemist_gazetteer/", threshold=t)
     
    
    if use_gazetteer:
        medprocner_cascade = [ medprocner_training_lookup, medprocner_training_semantic_lookup]
        symptemist_cascade = [ symptemist_training_lookup, symptemist_training_semantic_lookup]
        distemist_cascade = [ distemist_training_lookup, distemist_training_semantic_lookup]
    else:
        medprocner_cascade = [ medprocner_training_lookup, partial(embedding_lookup_function, prefix=" (procedimiento)", threshold=t)]
        symptemist_cascade = [ symptemist_training_lookup, partial(embedding_lookup_function, prefix=" (sintoma)", threshold=t)]
        distemist_cascade = [ distemist_training_lookup, partial(embedding_lookup_function, prefix=" (enfermedad)", threshold=t)]
        
    pharmaconer_cascade = [ pharmaconer_training_lookup, partial(embedding_lookup_function, prefix="", threshold=t)]
    
    _entities = sorted(['SYMPTOM' , 'PROCEDURE', 'DISEASE', 'PROTEIN', 'CHEMICAL'])
    
    document_annotations = defaultdict(list)
    for i, row in tqdm(run.iterrows()):

        prediction = []
        if row["label"]=="PROCEDURE":
            cascade_order = medprocner_cascade
        elif row["label"]=="SYMPTOM":
            cascade_order = symptemist_cascade
        elif row["label"]=="DISEASE":
            cascade_order = distemist_cascade
        elif row["label"]=="CHEMICAL" or row["label"]=="PROTEIN":
            cascade_order = pharmaconer_cascade
        else:
            raise ValueError
        
        for lookup_fn in cascade_order:
            if len(prediction) == 0:
                
                prediction.extend(lookup_fn(row["text"]))
            else:
                break
            
        if len(prediction) == 0:
            row['code'] = 'NO_CODE' 
        else:
            row['code'] = prediction
        
        document_annotations[row["filename"]].append(dict(row))
    
    
    # Desambiguation based on the document
    for doc_id, annotations in document_annotations.items():
        id_entities = defaultdict(list)
        for i, entity in enumerate(annotations):
            # assign id to entities
            entity["id"] = i
            if isinstance(entity["code"], list):
                for linked_id in entity["code"]:
                    id_entities[linked_id].append(entity["id"])
                        # print(id_entities)
                    
        # do majoraty voting (pick the id that has the longest list that each entity belongs too)
        for entity in annotations:
        
            if isinstance(entity["code"], list):
                
                most_freq_id, _ = max([(linked_id, len(id_entities[linked_id])) for linked_id in entity["code"]], key=lambda x:x[1])
                entity["code"] = most_freq_id

    data = [entity for annotations in document_annotations.values() for entity in annotations]
    
    df = pd.DataFrame(data)
    #df.columns = ["filename","strat_span","end_span","label","code", "text"]
    
    
    col_to_keep = ["filename","span_ini", "span_end","start_span","strat_span","end_span","label","code"]
    for _col in df.columns:
        if _col not in col_to_keep:
            df.drop(_col, inplace=True, axis=1)

    fname, _ = os.path.splitext(input_run)
    
    notes=""
    if use_gazetteer:
        notes="_wGazetteer"
    
    file_name = os.path.basename(fname)
    
    df.to_csv(f"{output_folder}/{file_name}_norm_{t}{notes}.tsv", sep="\t", index=False)
    
if __name__ == '__main__':
    main()