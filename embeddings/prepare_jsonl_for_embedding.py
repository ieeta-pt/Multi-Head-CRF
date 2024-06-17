import os
import json
import pandas as pd


files = {
    "../dataset/medprocner/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv":"medprocner",
    "../dataset/symptemist/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv":"symptemist",
    "../dataset/distemist/distemist_gazetteer/distemist_gazetteer.tsv":"distemist",
    "../snomedct/releases/SnomedCT_SpanishRelease-es_PRODUCTION_20190430T120000Z/RF2Release/Snapshot/Terminology/sct2_Description_SpanishExtensionSnapshot-es_INT_20190430.txt":"snomedCT"
}

for file, name in files.items():
    data = pd.read_csv(file, sep='\t')
    data['id'] = data['code']
    data['text'] = data['term']
    data = data[['id','text']]
    with open(name+".jsonl", 'w') as f:
        print(data.to_json(orient='records', lines=True),file=f, flush=False, end="")
