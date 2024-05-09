#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils import read_file
from utils import write_file


in_filepaths = [
    '../dataset/merged_data_subtask1_train.tsv',
    '../dataset/merged_data_subtask2_train.tsv',
    '../dataset/merged_data_subtask1_test.tsv',
    '../dataset/merged_data_subtask2_test.tsv',
]


out_filepaths = [
    'merged_data_subtask1_train_ready_for_evaluation.tsv',
    'merged_data_subtask2_train_ready_for_evaluation.tsv',
    'merged_data_subtask1_test_ready_for_evaluation.tsv',
    'merged_data_subtask2_test_ready_for_evaluation.tsv',
]


for in_fp, out_fp in zip(in_filepaths, out_filepaths):
    tsv_lines = read_file(in_fp).splitlines()[1:]
    #
    out_tsv = '{}\t{}\t{}\t{}\t{}\n'.format('doc_id', 'start_span', 'end_span', 'entity_type', 'code')
    #
    for line in tsv_lines:
        if 'subtask1' in in_fp:
            filename, ann_id, label, start_span, end_span, text = line.strip().split('\t')
            out_tsv += '{}\t{}\t{}\t{}\t{}\n'.format(filename, start_span, end_span, label, 'NO_CODE')
        else:
            assert 'subtask2' in in_fp
            filename, label, start_span, end_span, text, code = line.strip().split('\t')
            out_tsv += '{}\t{}\t{}\t{}\t{}\n'.format(filename, start_span, end_span, label, code)
    #
    write_file(out_fp, out_tsv)


print(
'''
Success!

The following files were created:
    merged_data_subtask1_train_ready_for_evaluation.tsv
    merged_data_subtask2_train_ready_for_evaluation.tsv
    merged_data_subtask1_test_ready_for_evaluation.tsv
    merged_data_subtask2_test_ready_for_evaluation.tsv

These gold standard files can be used in the evaluation.py script.
''')
