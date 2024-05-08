from collections import defaultdict
import pandas as pd
import os


#for subtask1, prepare the dataset
train = ['symptemist/symptemist_train/subtask1-ner/tsv/symptemist_tsv_train_subtask1.tsv',
'medprocner/medprocner_train/tsv/medprocner_tsv_train_subtask1.tsv',
'distemist/training/subtrack1_entities/distemist_subtrack1_training_mentions.tsv',
'pharmaconer/new_format/pharmaconer_task1_train.tsv']


test = ['symptemist/symptemist_test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv',
'medprocner/medprocner_test/tsv/medprocner_tsv_test_subtask1.tsv',
'distemist/test_annotated/subtrack1_entities/distemist_subtrack1_test_mentions.tsv',
'pharmaconer/new_format/pharmaconer_task1_test.tsv',
]

docs = defaultdict(list)


for ds in ['train', 'test']:
    final_data = []
    for file in eval(ds):
        df = pd.read_csv(file, sep="\t", names=['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'], skiprows=1)
        #relabel the rows
        df['label'] = df['label'].str.replace('SINTOMA','SYMPTOM')
        df['label'] = df['label'].str.replace('PROCEDIMIENTO','PROCEDURE')
        df['label'] = df['label'].str.replace('ENFERMEDAD','DISEASE')
        df['label'] = df['label'].str.replace('PROTEINAS','PROTEIN')
        df['label'] = df['label'].str.replace('NO_NORMALIZABLES','CHEMICAL')
        df['label'] = df['label'].str.replace('NORMALIZABLES','CHEMICAL')
        #rmeove the unclear label since it is not mapped to an entity
        df = df[df['label'] != 'UNCLEAR']

        for i, row in df.iterrows():
            # docs[row["filename"]].append({k:row[k] for k in ["ann_id", "label", "start_span", "end_span", "text"]})
            final_data.append(row)
        

    data = [{"filename":os.path.join("documents",f"{k}.txt"), "annotations":v} for k,v in docs.items()]    
    df = pd.DataFrame(final_data).sort_values(by=['filename', 'start_span', 'end_span'])
    df['ann_id'] = df.groupby('filename').cumcount()
    # df.drop('ann_id', axis=1, inplace=True)
    df.to_csv("merged_data_subtask1_"+ds+".tsv", index=False, sep='\t')

    


# merge distemist files and store the merged result
distemist_part1 = pd.read_csv('distemist/training/subtrack2_linking/distemist_subtrack2_training1_linking.tsv', sep='\t')
distemist_part2 = pd.read_csv('distemist/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv', sep='\t')
distemist_final = pd.concat([distemist_part1, distemist_part2])
distemist_final.to_csv('distemist/training/subtrack2_linking/distemist_subtrack2_merged_linking.tsv', index=False, sep='\t')

#for subtask1, prepare the dataset
train = ['symptemist/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2.tsv',
'medprocner/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv',
'distemist/training/subtrack2_linking/distemist_subtrack2_merged_linking.tsv',
'pharmaconer/new_format/pharmaconer_task2_train.tsv']


test = ['symptemist/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv',
'medprocner/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv',
'distemist/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv',
'pharmaconer/new_format/pharmaconer_task2_test.tsv',
]

docs = defaultdict(list)

# filename, label, start_span, end_span, text

start_span_labels = ['start_span', 'span_ini', 'off0']
end_span_labels = ['end_span', 'span_end', 'off1']
text_labels =['span', 'text']

for ds in ['train', 'test']:
    final_data = []
    for file in eval(ds):
        # print(file)
        df = pd.read_csv(file, sep="\t")
        #relabel the rows
        df['label'] = df['label'].str.replace('SINTOMA','SYMPTOM')
        df['label'] = df['label'].str.replace('PROCEDIMIENTO','PROCEDURE')
        df['label'] = df['label'].str.replace('ENFERMEDAD','DISEASE')
        df['label'] = df['label'].str.replace('PROTEINAS','PROTEIN')
        df['label'] = df['label'].str.replace('NORMALIZABLES','CHEMICAL')
        df['label'] = df['label'].str.replace('NO_NORMALIZABLES','CHEMICAL')
        #rmeove the unclear label since it is not mapped to an entity
        df = df[df['label'] != 'UNCLEAR']

        for i, row in df.iterrows():
            # docs[row["filename"]].append({k:row[k] for k in ["ann_id", "label", "start_span", "end_span", "text"]})
            start_span = -1
            end_span = -1
            text=-1
            for k in start_span_labels:
                if k in dict(row).keys():
                    start_span = row[k]
            for k in end_span_labels:
                if k in dict(row).keys():
                    end_span = row[k]
            for k in text_labels:
                if k in dict(row).keys():
                    text = row[k]  
            final_data.append({
                'filename':row['filename'], 
                'label':row['label'], 
                'start_span':start_span, 
                'end_span':end_span, 
                'text':text,
                'code':row['code']
                })
        



    df = pd.DataFrame(final_data).sort_values(by=['filename', 'start_span', 'end_span'])

    df.to_csv("merged_data_subtask2_"+ds+".tsv", index=False, sep='\t')


    