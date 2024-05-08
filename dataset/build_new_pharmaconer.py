import pandas as pd
import os
from collections import defaultdict

file_mappings = []


symptemist_df_test = pd.read_csv('symptemist/symptemist_test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv', sep='\t')
symptemist_df_train = pd.read_csv('symptemist/symptemist_train/subtask1-ner/tsv/symptemist_tsv_train_subtask1.tsv', sep='\t')
file_mappings.extend(symptemist_df_test.filename)
file_mappings.extend(symptemist_df_train.filename)
file_mappings = set(file_mappings)


data = defaultdict(lambda: defaultdict(lambda: dict))

dirs = ['pharmaconer/test-set_1.1/test/', 'pharmaconer/dev-set_1.1/dev/', 'pharmaconer/train-set_1.1/train/']

true_codes = []

for dir in dirs:
    for file in os.listdir(dir+'subtrack1/'):
        if file.endswith('.ann'):
            with open(dir+'subtrack2/'+file[:-4]+".tsv") as f:
                for line in f:
                    expected_docid, true_code = line.strip().split('\t')
                    if true_code == '<null>':
                        true_code = 'NO_CODE'
                    true_codes.append(true_code)

            with open(dir+'subtrack1/'+file) as f:
                file = file[:-4]
                if file not in file_mappings:
                    file = 'es-'+str(file)
                for line in f:
                    line = line.rstrip()
                    if not line.startswith('#'):
                        ann_id, label_spans, text = line.split('\t')
                        label, start_span, end_span = label_spans.split()
                        start_span = int(start_span)
                        end_span = int(end_span)

                        if file == 'es-S0211-69952015000200015-1':
                            if (start_span, end_span) == (1376, 1387): # Bence-Jones
                                start_span, end_span = (1375, 1386)
                            if (start_span, end_span) == (1915, 1928): # catecolaminas
                                start_span, end_span = (1914, 1927)
                            if (start_span, end_span) == (1931, 1943): # metanefrinas
                                start_span, end_span = (1930, 1942)
                            if (start_span, end_span) == (2301, 2307): # calcio
                                start_span, end_span = (2299, 2305)
                            if (start_span, end_span) == (2565, 2575): # creatinina
                                start_span, end_span = (2563, 2573)
                       
                        data[file][ann_id] = {'label': label, 'start_span': start_span, 'end_span': end_span, 
                                             'text': text, 'code': "NO_CODE"}

                    else:
                        _, notes , code =  line.split('\t')
                        ann_id = notes.split()[1]

                        if code == '<null>':
                            code = "NO_CODE"
                        data[file][ann_id]['code'] =code
data_pd = []
for docid in  data.keys():
    for annid, new_data  in  data[docid].items():
        data_pd.append({'filename':docid,'ann_id':annid}|new_data)

df = pd.DataFrame(data_pd)


df_task2 = df[['filename', 'label', 'start_span', 'end_span', 'text', 'code']]
df_task1 = df[['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text']]


symptemist_df = pd.read_csv('symptemist/symptemist_test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv', sep='\t')
len(set(symptemist_df.filename))

test_ids = set()
for file in os.listdir('symptemist/symptemist_test/subtask1-ner/txt'):
    test_ids.add(file[:-4])
train_ids = set()
for file in os.listdir('symptemist/symptemist_train/subtask1-ner/txt'):
    train_ids.add(file[:-4])
if not os.path.exists("pharmaconer/new_format/"):
    os.mkdir("pharmaconer/new_format/")

df_task1[df.filename.isin(train_ids)].to_csv('pharmaconer/new_format/pharmaconer_task1_train.tsv', index=False, sep='\t')
df_task1[df.filename.isin(test_ids)].to_csv('pharmaconer/new_format/pharmaconer_task1_test.tsv', index=False, sep='\t')

df_task2[df.filename.isin(train_ids)].to_csv('pharmaconer/new_format/pharmaconer_task2_train.tsv', index=False, sep='\t')
df_task2[df.filename.isin(test_ids)].to_csv('pharmaconer/new_format/pharmaconer_task2_test.tsv', index=False, sep='\t')