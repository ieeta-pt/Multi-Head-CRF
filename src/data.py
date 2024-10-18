from typing import Any
import torch
import os
import pandas as pd
import random
import math

from utils import split_chunks, RangeDict
from collections import defaultdict

from transformers import DataCollatorForTokenClassification

from itertools import groupby
from operator import itemgetter

# import logging


# logger = logging.getLogger()
# if logger.hasHandlers():
#     logger.handlers.clear()
# logger.setLevel(logging.DEBUG)
# stream_handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)


class Corpus:
    #  A class that represent a corpus. If you want a specific dataset, you need to implement a function that returns a corpus
    def __init__(self, data: list ):
        
        assert set(data[0].keys()) == set(["doc_id", "text", "annotations"]) , "Every document needs to contain a field call 'doc_id', 'text' and 'annotation'"
        assert set(data[0]["annotations"][0].keys()) == set(['label', 'start_span', 'end_span']), "Every annotation needs to contain a 'start_span', 'end_span' and 'label'"
        
        #Data needs to be formated as follows {"doc_id":id, "text": doucment_text, "annotations":[{'label': LABEL, 'start_span': num, 'end_span': num}, ... ]  }   
        self.corpus = data
    def __len__(self):
        return len(self.corpus)
    
    def __iter__(self):
        return iter(self.corpus)
    
    def __getitem__(self, index):
        return self.corpus[index]
    
    
    def get_entities(self):
        entities = set()
        for document in self.corpus:
            for annotation in document['annotations']:
                entities.add(annotation['label'])
        return entities
        
    def split(self, split):
        split_index = int(len(self.corpus) * split)
        trainCorpus = Corpus(self.corpus[split_index:])
        testCorpus = Corpus(self.corpus[:split_index])
        return trainCorpus, testCorpus

class CorpusPreProcessor:  
        
    def __init__(self, corpus:Corpus):
        self.corpus = corpus
    
    def filter_labels(self, labels):
        filtered_data = []
        for document in self.corpus:
            filtered_annotations = [ann for ann in document['annotations'] if ann['label'] in labels]
            filtered_data.append({'doc_id':document['doc_id'], 'text': document['text'], 'annotations':filtered_annotations})
        self.corpus = Corpus(filtered_data)
        
    # def add_ids_to_annotations(self):    
    #     id =0    
    #     for document in self.data:
    #         # id = defaultdict(lambda: 0)
    #         for ann in document['annotations']:
    #             id +=1
    #             ann['ann_id'] = ann['label'][:4] +"_"+ str(id[ann['label']])
    
    def merge_annoatation(self):
        total_collisions=defaultdict(lambda: 0)
        entities  = self.corpus.get_entities()
        
        for doc in self.corpus:
            new_annotations = []

            sample_annotations = defaultdict(lambda: RangeDict())
            new_annotation_index = 0
            for annotation in doc["annotations"]:
                # checks for a collision and returns the spans of the new entity
                new_span = sample_annotations[annotation['label']].maybe_merge_annotations(annotation)
                if new_span:
                    new_annotation_index += 1
                    # lets create a new annotation bc collision
                    annotation = {"label": annotation['label'], "start_span": new_span[0], "end_span": new_span[1]}
                    total_collisions[annotation['label']]+=new_span[2]
                    
                sample_annotations[annotation['label']][(annotation["start_span"], annotation["end_span"])] = annotation
            for entity_type in entities:
                new_annotations.extend(sample_annotations[entity_type].get_all_annotations())
            doc['annotations'] = new_annotations
            
            
        if sum(total_collisions.values())>0:
            print(f"Warning, we found {total_collisions} collisions that were automaticly handle by merging strategy")
        
    def split_data(self, test_split_percentage):
        trainCorpus, testCorpus = self.corpus.split(test_split_percentage)
        return  CorpusPreProcessor(trainCorpus), CorpusPreProcessor(testCorpus)
    


def Spanish_Biomedical_NER_Corpus(file_path, documents_folder):
    # file_path: tsv with the columns "label", "start_span", "end_span", "filename"
    # document_path: dir containing the filenames
    
    # Load the data into a dictionary
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
    
    corpus = Corpus(data)
    
    return corpus
    
    
class CorpusTokenizer:
    def __init__(self, corpus: CorpusPreProcessor, tokenizer, context_size = 0):
        self.corpus  = corpus.corpus
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.dataset = []
        
        self.tokenizer_uses_special_tokens = (hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None) and (hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None)
        # if self.context_size == 0:
        self.__tokenize()
        
        self.context_size = context_size
        if self.tokenizer_uses_special_tokens and context_size!=0:
            self.context_size -= 1 # cls + sep
        self.center_tokens = tokenizer.model_max_length - 2*context_size
        self.__split()
            
 
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return iter(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __tokenize(self):
        for doc in self.corpus:
            encoding = self.tokenizer(doc['text'], add_special_tokens=False)[0]
            tokens = encoding.ids
            offsets = encoding.offsets
            attention_mask = [1] * len(tokens)
            
            sample = {
                        "doc_id": doc['doc_id'],
                        "text": doc['text'],
                        "input_ids": tokens,
                        "attention_mask": attention_mask,
                        "offsets": offsets,
                        "annotations": doc["annotations"]
                    }
            self.dataset.append(sample)



    def __split(self):
        entities = self.corpus.get_entities()
        split_data = []
        for doc in self.dataset:    #                            
            attention_mask = [0] * self.context_size + [1] * len(doc['input_ids'])
            tokens = [self.tokenizer.pad_token_id] * self.context_size + doc['input_ids']
            
            offsets = [None] * self.context_size + doc['offsets']

            
            for j,i in enumerate(range(self.context_size,len(tokens),self.center_tokens)):
                
                left_context_tokens = tokens[i-self.context_size:i]
                central_tokens = tokens[i:i+self.center_tokens]
                right_context_tokens = tokens[i+self.center_tokens:i+self.center_tokens+self.context_size]
                
                left_context_offsets = offsets[i-self.context_size:i]
                central_offsets = offsets[i:i+self.center_tokens]
                right_context_offsets = offsets[i+self.center_tokens:i+self.center_tokens+self.context_size]
                
                left_context_attention_mask = attention_mask[i-self.context_size:i]
                central_attention_mask = attention_mask[i:i+self.center_tokens]
                right_context_attention_mask = attention_mask[i+self.center_tokens:i+self.center_tokens+self.context_size]
                
                sample_tokens = left_context_tokens + central_tokens + right_context_tokens
                sample_offsets = left_context_offsets + central_offsets + right_context_offsets
                sample_attention_mask = left_context_attention_mask + central_attention_mask + right_context_attention_mask 
                
                if self.tokenizer_uses_special_tokens:
                
                    sample_tokens = [self.tokenizer.cls_token_id] + sample_tokens + [self.tokenizer.sep_token_id]
                    sample_offsets = [None] + sample_offsets + [None]
                    sample_attention_mask = [1] + sample_attention_mask + [1]
                
                assert len(sample_tokens)<=self.tokenizer.model_max_length 
                # and len(sample_offsets)<=self.tokenizer.model_max_length
                
                if j==0:
                    low_offset, high_offset = sample_offsets[self.context_size+1][0], sample_offsets[-2][1]
                else:
                    low_offset, high_offset = sample_offsets[1][0], sample_offsets[-2][1]

                
                sample_annotations = {k:RangeDict() for k in entities}

                for annotation in doc["annotations"]:
                    if annotation["start_span"] >= low_offset and annotation["start_span"]<=high_offset or annotation["end_span"] >= low_offset and annotation["end_span"]<=high_offset:         
                        sample_annotations[annotation['label']][(annotation["start_span"], annotation["end_span"])] = annotation
                        

                    
                sample = {
                    "doc_id": doc['doc_id'],
                    "text": doc['text'][low_offset: high_offset],
                    "sequence_id": j, 
                    "input_ids": sample_tokens,
                    "attention_mask": sample_attention_mask,
                    "offsets": sample_offsets,
                    # "view_offset": (low_offset, high_offset),
                    "annotations": doc["annotations"],
                    "label_range_dict": sample_annotations, 
                    "list_annotations": {k:v.get_all_annotations() for k,v in sample_annotations.items()},
                }
                
                assert len(sample["input_ids"])<=self.tokenizer.model_max_length
                assert len(sample["offsets"])<=self.tokenizer.model_max_length
                
                split_data.append(sample)
        self.dataset = split_data 
                
                
                
class CorpusDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 tokenized_corpus: CorpusTokenizer ,
                 transforms=None,
                 augmentations=None):
        super().__init__()
        
        self.dataset = tokenized_corpus.dataset
        self.transforms = transforms
        self.augmentations =augmentations
        self.__transform_and_augment()
        
    def __transform_and_augment(self):
        transformed_data= []
        for sample in self.dataset:
            
            if self.transforms is not None:
                for transform in self.transforms:
                    sample = transform(sample)
                    
            if self.augmentations is not None:
                for augmentation in self.augmentations:
                    sample = augmentation(sample)
            transformed_data.append(sample)
        self.dataset = transformed_data
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample




    
    
    
    
        
# class Conll_Dataset(Dataset):
    # assumes data is in the Conll format (tsv): word,fileID,span,tag, 
    # span is assumed to take the form start_end.
    # What	document1	163_166	O
    # document1.txt is assumed to exist within the documents folder
    
    # def __init__(self, path):
    #     self.load_data(path)
    #     # self.merge_bio_tags()
       
    
    # def load_data(self, path ):
    #     with open(path,'r') as file:
            
    #         lines = [x.lower() for x in file.readlines()]
    #         file_content = set(lines)
    #         print(len(lines), len(file_content))
    #         for line in file_content:
    #             line = line.rstrip('\n')
    #             if line != '': # checking empty string 
    #                 if len(line.split('\t')) !=4:
    #                     print(len(line.split('\t')))
    #                     logging.error("Error parsing the CoNLL file format for the line: "+line)
    #                 else:
    #                     word,file_name,span,tag = line.split('\t')                        
    #                     start_span, end_span = span.split('_')
    #                     # if tag != 'O' and {"start_span": int(start_span),"end_span":int(end_span), "label":tag, "text": word} not in self.data[file_name]:        
    #                     self.data[file_name].append({"start_span": int(start_span),"end_span":int(end_span), "label":tag, "text": word})
    #     for file, data in self.data.items():
    #         # print(data)
    #         tmp_strings = []
    #         for tmp in data:
    #             if tmp['label'] != 'o':
    #                 tmp_strings.append(f"{tmp['start_span']}_{tmp['end_span']}_{tmp['label']}")
            
    #         if (len(set(tmp_strings))!= len(tmp_strings)):
    #             print(data)
    #             print(sorted(tmp_strings))

    #             print(len(data), len(tmp_strings), len(set(tmp_strings)))
    
    
    # # def load_data(self, path ):
    # #     with open(path,'r') as file:
    # #         for line in file.readlines():
    # #             line = line.rstrip('\n')
    # #             if line != '': # checking empty string 
    # #                 if len(line.split('\t')) !=4:
    # #                     print(len(line.split('\t')))
    # #                     logging.error("Error parsing the CoNLL file format for the line: "+line)
    # #                 else:
    # #                     word,file_name,span,tag = line.split('\t')                        
    # #                     start_span, end_span = span.split('_')
    # #                     if tag != 'O' and {"start_span": int(start_span),"end_span":int(end_span), "label":tag, "text": word} not in self.data[file_name]:        
    # #                         self.data[file_name].append({"start_span": int(start_span),"end_span":int(end_span), "label":tag, "text": word})
                            
                            
    # def merge_bio_tags(self):
    #     # This function takes the B-Person I person, and merges into a single span person, assumes that the pairs are ordered
    #     # edit this is a bad practice
    #     new_data = defaultdict(list)
    #     for file, annotations in self.data.items():
    #         for annotation in sorted(annotations, key=lambda d: d['start_span']):
    #         # for annotation in annotations:

    #             if annotation['label'][0] == 'b':
    #                 annotation['label'] = annotation['label'][2:]
    #                 new_data[file].append(annotation)
    #             else:
    #                 new_data[file][-1]['end_span'] = annotation['end_span']
    #                 new_data[file][-1]['text'] += " " + annotation['text']
                
    #     self.data = new_data

                            # docs[row["filename"]].append({k:row[k] for k in ["ann_id", "label", "start_span", "end_span", "text"]})
    
        
    # data = [{"filename":os.path.join("../dataset/documents",f"{k}.txt"), "annotations":v} for k,v in docs.items()]



class SelectModelInputs():
    
    def __call__(self, sample) -> Any:
        return { k:sample[k] for k in ["input_ids", "attention_mask", "labels"]}

    
    
class BIOTagger():
    
    def __call__(self, sample) -> Any:
        label_dict ={}
        for ent in sample["label_range_dict"].keys():
        
            labels = [0]
            prev_annotation = None
            current_label = 0
            
            for offset in sample["offsets"][1:]:
                if offset is None:
                    current_label = 0
                else:
                    if offset in sample["label_range_dict"][ent]:
                        if prev_annotation != sample["label_range_dict"][ent][offset]:
                            current_label = 1
                            prev_annotation = sample["label_range_dict"][ent][offset]
                        else:
                            current_label = 2
                    else:
                        current_label = 0
                        prev_annotation = None
                    
                labels.append(current_label)
            
            #pad
            labels = labels + [0]*(len(sample["offsets"])-len(labels))
            label_dict[ent] = labels

        
        return sample | {"labels": label_dict}

#if prev_annotation is None:
#    current_label = 1
#    prev_annotation = sample["annotations"][offset]
#else:      

# O O O B I I O O O
# entity identification

# O O O O B O O

class RandomlyUKNTokens:

    def __init__(self, 
                 tokenizer, 
                 context_size,
                 prob_change=0.5, 
                 percentage_changed_tags=0.2):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.prob_change = prob_change
        self.percentage_changed_tags = percentage_changed_tags
    
    def pick_token(self):
        return self.tokenizer.unk_token_id
    
    def __call__(self, sample) -> Any:
        
        if torch.rand(1) < self.prob_change:
            bi_tags = []
            o_tags = []
            ent = random.choice(list(sample['labels'].keys()))
            #for ent in sample['labels'].keys():
            
                # pick tokens based on the tags, same amount of O and B/I
                
                # get total of BI and O tags
               
                # when the sample ends it may not have right context
                # print( len(sample["labels"][ent]))
            right_context = max(self.context_size - (self.tokenizer.model_max_length - len(sample["labels"][ent])), 0)
            
            if right_context == 0:
                labels = sample["labels"][ent][self.context_size:]
            else:
                labels = sample["labels"][ent][self.context_size:-right_context]

            for i,tag in enumerate(labels):
                if tag==0:
                    o_tags.append(i+self.context_size)
                else:
                    bi_tags.append(i+self.context_size)

            bi_tags = sorted((bi_tags))
            o_tags = sorted((o_tags))
            num_changes = int(self.percentage_changed_tags*len(bi_tags))
            if num_changes==0 and len(bi_tags)>0:
                num_changes=1
            
            bi_rand_indexes = torch.randperm(len(bi_tags))[:num_changes]
            o_rand_indexes = torch.randperm(len(o_tags))[:num_changes]
            for i in bi_rand_indexes:
                sample["input_ids"][bi_tags[i]] = self.pick_token()
                
            for i in o_rand_indexes:              
                sample["input_ids"][o_tags[i]] = self.pick_token()

        return sample
    
class RandomlyReplaceTokens(RandomlyUKNTokens):
    def pick_token(self):
        token_id = int(torch.rand(1)*self.tokenizer.vocab_size)

        while token_id in [self.tokenizer.unk_token_id,self.tokenizer.pad_token_id,self.tokenizer.sep_token_id,self.tokenizer.cls_token_id]:
            token_id = int(torch.rand(1)*self.tokenizer.vocab_size)
            
        return token_id
    
    
class EvaluationDataCollator(DataCollatorForTokenClassification):
    
    def torch_call(self, features):
        
        model_inputs = {"input_ids", "attention_mask"}
        
        reminder_columns = set(features[0].keys()) - model_inputs
        
        out = {k:[] for k in reminder_columns}
        inputs = [{k: feature[k] for k in model_inputs}
                  for feature in features ]
        
        for feature in features:
            for k in reminder_columns:
                out[k].append(feature[k])
        
        out["inputs"] = super().torch_call(inputs)
        
        return out



def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded



class TrainDataCollator(DataCollatorForTokenClassification):

    def torch_call(self, features): # labels:[{disease: [] }, {disease: []}], input
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None #v[{}, ] /Z diseases:[list]
        
        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side
        
        
        new_labels = defaultdict(lambda: list())
        entites = set()
        for sample in labels:
            for ent,l in sample.items():
                padded_lab = l + [0] * (sequence_length - len(l))                
                new_labels[ent].append(padded_lab) #{symptoms:[[] [] ]}

        for ent in new_labels.keys():
            new_labels[ent] = torch.tensor(new_labels[ent], dtype = torch.int64)
            
        batch[label_name] = dict(new_labels)
            
        return batch    