import click
import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
import os
import subprocess

NORMALIZER_MODEL_MAPPINGS = {
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large": "sapBERT-multilanguage-large",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "sapBERT-multilanguage",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token": "sapBERT-english",
}

NORMALIZER_MODEL_MAPPINGS_REVERSED = {v:k for k,v in NORMALIZER_MODEL_MAPPINGS.items()}


def normalize_emb(emb):
    return emb/torch.linalg.norm(emb,ord=2, axis=-1, keepdims=True)

def get_embeddings(text, tokenizer, model, device, normalize=False):
    
    with torch.no_grad():
        
        toks_cuda = tokenizer.batch_encode_plus(text, 
                                                padding="max_length", 
                                                max_length=25, 
                                                truncation=True,
                                                return_tensors="pt").to(device)

        # cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        cls_rep = model(**toks_cuda).last_hidden_state[:,0,:]#[0].mean(axis=1)#
        if normalize:
            cls_rep = normalize_emb(cls_rep).cpu().detach().numpy()
        return cls_rep
    #embeddings.append(cls_rep.cpu().detach().numpy())
    #embeddings = 
    #return embeddings

def batch_generator(file_path, batch_size):

    batch = []
    with open(file_path) as f:
        for data in map(json.loads, f):
            batch.append(data["text"]) 
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  
            yield batch



def count_lines_fast(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    return int(result.stdout.split()[0])

@click.command()
@click.argument("input_file")
@click.option("--checkpoint", default="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large")
@click.option("--batch_size", default=128)
@click.option("--device", default="cuda")
@click.option("--normalize", is_flag=True)
def main(input_file, checkpoint, batch_size, device, normalize):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    model = AutoModel.from_pretrained(checkpoint).to(device)
    
    embeddings = []
    for batch_text in tqdm(batch_generator(input_file, batch_size=batch_size), total=count_lines_fast(input_file)//batch_size):
        embeddings.append(get_embeddings(batch_text, tokenizer, model, device, normalize=normalize))
    embeddings = np.concatenate(embeddings, axis=0) 
    
    base_file, _ = os.path.splitext(input_file)
    model_base_name = NORMALIZER_MODEL_MAPPINGS[checkpoint]
    np.save(f"{base_file}_{normalize}_embeddings_{model_base_name}.npy", embeddings)
    
if __name__ == "__main__":
    main()
    
    