#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# How to use this file in another Python file:
# from load_datasets import datasets, data
#

from elements import Entity
from elements import EntitySet

from utils import get_filepaths
from utils import read_file


datasets = {
    'symptemist': {
        'train': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/symptemist/symptemist_train/subtask1-ner/tsv/symptemist_tsv_train_subtask1.tsv',
            'nen_tsv': ['../dataset/symptemist/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete+COMPOSITE.tsv'],
        },
        'test': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/symptemist/symptemist_test/subtask1-ner/tsv/symptemist_tsv_test_subtask1.tsv',
            'nen_tsv': ['../dataset/symptemist/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2+COMPOSITE.tsv'],
        },
    },
    'medprocner': {
        'train': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/medprocner/medprocner_train/tsv/medprocner_tsv_train_subtask1.tsv',
            'nen_tsv': ['../dataset/medprocner/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv'],
        },
        'test': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/medprocner/medprocner_test/tsv/medprocner_tsv_test_subtask1.tsv',
            'nen_tsv': ['../dataset/medprocner/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv'],
        },
    }, 
    'distemist': {
        'train': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/distemist/training/subtrack1_entities/distemist_subtrack1_training_mentions.tsv',
            'nen_tsv': ['../dataset/distemist/training/subtrack2_linking/distemist_subtrack2_training1_linking.tsv',
                        '../dataset/distemist/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv'],
        },
        'test': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/distemist/test_annotated/subtrack1_entities/distemist_subtrack1_test_mentions.tsv',
            'nen_tsv': ['../dataset/distemist/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv'],
        },
    },
    'pharmaconer': {
        'train': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/pharmaconer/new_format/pharmaconer_task1_train.tsv',
            'nen_tsv': ['../dataset/pharmaconer/new_format/pharmaconer_task2_train.tsv'],
        },
        'test': {
            'txt_dir': '../dataset/documents/',
            'ner_tsv': '../dataset/pharmaconer/new_format/pharmaconer_task1_test.tsv',
            'nen_tsv': ['../dataset/pharmaconer/new_format/pharmaconer_task2_test.tsv'],
        },
    },
}


data = {
    'symptemist': {
        'train': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
        'test': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
    },
    'medprocner': {
        'train': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
        'test': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
    },
    'distemist': {
        'train': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
        'test': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
    },
    'pharmaconer': {
        'train': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
        'test': {
            'docid2text': dict(),
            'docid2entities': dict(),
        },
    },
}


def contains_only_digits(s):
    assert isinstance(s, str)
    for c in s:
        if c not in '0123456789':
            return False
    return True


def valid_code(code):
    assert isinstance(code, str)
    #
    if '+' in code:
        codes = code.split('+')
    else:
        codes = [code]
    #
    for code in codes:
        if code == 'NO_CODE':
            continue
        if not contains_only_digits(code):
            return False
    #
    return True


#
# Let's convert the entity types from Spanish to English.
#
spanish_to_english = {
    'SINTOMA': 'SYMPTOM',
    'PROCEDIMIENTO': 'PROCEDURE',
    'ENFERMEDAD': 'DISEASE',
    'NORMALIZABLES': 'CHEMICAL',
    'NO_NORMALIZABLES': 'CHEMICAL',
    'PROTEINAS': 'PROTEIN',
    'UNCLEAR': 'UNCLEAR',
}


for dname, dataset in datasets.items():
    for subset in dataset:
        #
        # First, get docid2text.
        #
        for fpath, _, docid in zip(*get_filepaths(dataset[subset]['txt_dir'], '.txt')):
            data[dname][subset]['docid2text'][docid] = read_file(fpath).rstrip()
            data[dname][subset]['docid2entities'][docid] = EntitySet()
        #
        # Second, get docid2entities.
        #
        ner_tsv = read_file(dataset[subset]['ner_tsv'])
        ner_tsv_lines = ner_tsv.splitlines()
        #
        header = ner_tsv_lines[0]
        assert header.startswith('filename')
        #
        for line in ner_tsv_lines[1:]:
            docid, ann_id, label, start_span, end_span, text = line.strip().split('\t')
            label = spanish_to_english[label]
            s = int(start_span)
            e = int(end_span)
            span = (s, e)
            #
            # Extra step of processing to fix some entity mentions
            # that are incorrect.
            #
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
                text = text.replace('""', '"')
            assert data[dname][subset]['docid2text'][docid][s:e] == text
            #
            data[dname][subset]['docid2entities'][docid].add(Entity(text, span, label))
        #
        # Third, add the normalization identifiers (codes) into the respective entities.
        #
        for docid, entities in data[dname][subset]['docid2entities'].items():
            for e in entities:
                e.code = 'NO_CODE'
        #
        nen_tsv_lines = list()
        for i, nen_tsv_file in enumerate(dataset[subset]['nen_tsv']):
            lines = read_file(nen_tsv_file).splitlines()
            header = lines[0]
            assert header.startswith('filename')
            if i == 0:
                nen_tsv_lines += lines
            else:
                nen_tsv_lines += lines[1:]
        #
        for line in nen_tsv_lines[1:]:
            if dname in {'symptemist', 'medprocner'}:
                docid, label, start_span, end_span, text, code, sem_rel, is_abbrev, is_composite, need_context = line.strip().split('\t')
            elif dname in {'distemist'}:
                docid, ann_id, label, start_span, end_span, text, code, sem_rel = line.strip().split('\t')
            elif dname in {'pharmaconer'}:
                docid, label, start_span, end_span, text, code = line.strip().split('\t')
            else:
                assert False
            label = spanish_to_english[label]
            s = int(start_span)
            e = int(end_span)
            span = (s, e)
            #
            # Deal with incorrect identifiers.
            #
            if (dname in {'distemist'}) and (code == 'NOMAP'):
                code = 'NO_CODE'
            #
            INCORRECT_CODES = [
                '1.66753E+16',            # symptemist, train
                '1.59978E+16',            # symptemist, train
                'N+O',                    # symptemist, train
                '1.65511910001191E+016',  # medprocner, test
                '1.63195610001191E+016',  # medprocner, test
                '1.63185210001191E+016',  # medprocner, test
                '4.50229981000132E+017',  # medprocner, test
            ]
            if code not in INCORRECT_CODES:
                assert valid_code(code) or (dname == 'pharmaconer' and code.lower().startswith('chebi:'))
            #
            found = False
            for e in data[dname][subset]['docid2entities'][docid]:
                if e.span == span:
                    e.code = code
                    found = True
                    break
            assert found
