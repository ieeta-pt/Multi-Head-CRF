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
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from data import Corpus # Keeping your original import

# Helper function for the threads to use
def _read_single_document(file_info):
    folder, file_name = file_info
    with open(os.path.join(folder, file_name), 'r', encoding='utf-8') as f:
        return {"doc_id": file_name, "text": f.read().strip()}

class Spanish_Biomedical_NER_Corpus_Inference(Corpus):
    def __init__(self, documents_folder, entities:list):
        data = []
        files = os.listdir(documents_folder)
        
        print(f"📂 Found {len(files)} documents. Firing up 16 threads to load them...")
        
        # Create a list of tuples containing the folder and filename
        file_infos = [(documents_folder, f) for f in files]
        
        # Use ThreadPoolExecutor to read files in parallel (massive speedup for network drives)
        with ThreadPoolExecutor(max_workers=16) as executor:
            # map automatically keeps the results in order, and tqdm gives us a sweet progress bar
            results = list(tqdm(executor.map(_read_single_document, file_infos), total=len(files), desc="Reading TXT files", unit="docs"))
            
        data.extend(results)

        super().__init__(data, entities=entities)