import pandas as pd
from collections import defaultdict
from data import CorpusAnnotated, Corpus
import os

class Spanish_Biomedical_NER_Corpus(CorpusAnnotated):
    def __init__(self, file_path, documents_folder):
        annotations = defaultdict(list)
        df = pd.read_csv(file_path, sep="\t")
        for _, row in df.iterrows():
            annotations[row["filename"]].append({k:row[k] for k in ["label", "start_span", "end_span"]})

        #load the documents
        document_text = {}
        for file in annotations.keys():
            with open(os.path.join(documents_folder,f"{file}.txt"), 'r') as f:
                document_text[file] = ''.join([line for line in f]).strip()

        data = [{"doc_id":k, "text":document_text[k], "annotations":v} for k,v in annotations.items()]   

        super().__init__(data)

class Spanish_Biomedical_NER_Corpus_Inference(Corpus):
    def __init__(self, documents_folder, entities:list):
        
        #load the documents
        data = []
        for file in os.listdir(documents_folder):
            with open(os.path.join(documents_folder,file), 'r') as f:
                doc_text = ''.join([line for line in f]).strip()
                data.append({"doc_id":file, "text":doc_text})

        super().__init__(data, entities=entities)