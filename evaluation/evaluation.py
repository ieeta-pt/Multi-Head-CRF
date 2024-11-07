#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys

args = sys.argv[1:]

if len(args) != 2:
    print()
    print('Usage:')
    print('  $  python3  evaluation.py  SUBSET  PREDICTIONS_FILE.tsv')
    print()
    print('Examples of usage:')
    print('  $  python3  evaluation.py  train   pred.tsv')
    print('  $  python3  evaluation.py  test    pred.tsv')
    print()
    print('----------------------------------------------------------------------------------------------------')
    print()
    print('NER (named entity recognition) and entity linking are simultaneously evaluated.')
    print()
    print('The subset (train or test) to be evaluated needs to be selected.')
    print('This will load the respective (gold standard) subset in the four datasets:')
    print('    SympTEMIST')
    print('    MedProcNER')
    print('    DisTEMIST')
    print('    PharmaCoNER (note that the train and test split was adjusted to match')
    print('                 the same from the other three datasets)')
    print()
    print('The input TSV file with NER and entity linking predictions should have exactly these five columns:')
    print('doc_id         start_span    end_span    entity_type    code')
    print('S0004-06142    1023          1044        PROCEDURE      449264008')
    print('S0004-06142    208           240         PROCEDURE      449263002+449264008')
    print()
    print('Note that composite mentions containing multiple codes are concatenated with a plus sign (+).')
    print()
    print('For an entity to be correct (in NER evaluation) its span and entity type must exactly match.')
    print()
    print('If an entity contains multiple wrong codes we count this as a single FP (False Positive) and a')
    print('single FN (False Negative).')
    print()
    print('This is the evaluation approach followed by the official MedProcNER evaluation script (May 2023):')
    print('    https://github.com/TeMU-BSC/medprocner_evaluation_library')
    print()
    print('Also for reference:')
    print()
    print('    DisTEMIST evaluation library (May 2022)')
    print('    https://github.com/TeMU-BSC/distemist_evaluation_library')
    print()
    print('    PharmaCoNER evaluation script (March 2019)')
    print('    https://github.com/PlanTL-GOB-ES/PharmaCoNER-Evaluation-Script')
    print()
    exit()


#
# Let's start the evaluation...
#
subset, pred_filepath = args

print('python3  evaluation.py  {}  {}'.format(subset, pred_filepath))
print()

print('Subset:')
print('    {}'.format(subset))
print()

print('Predictions file path:')
print('    {}'.format(repr(pred_filepath)))
print()


import numpy as np

from load_datasets import data, MISSING_CODE
from elements import EntitySet
from utils import read_file


pred = read_file(pred_filepath)


UNIQUE_ENTITY_TYPES = set()













#
# First, we need to get the gold standard "docid2entities".
# We will simplify how data is stored.
#
# For simplicity an entity will be represented as a dict():
# {'span_type': '5-10;SYMPTOM', 'code': '8943002'}
#

#
# Merge the data from the four datasets.
#
merged_data = {}

for dname, dataset in data.items():
    #
    for _subset, d in dataset.items():
        if _subset not in merged_data:
            merged_data[_subset] = {
                'docid2text': d['docid2text'],
                'docid2entities': {docid: EntitySet() for docid in d['docid2text']}
            }
        #
        for docid, entities in d['docid2entities'].items():
            for e in entities:
                if e.typ != 'UNCLEAR':
                    merged_data[_subset]['docid2entities'][docid].add(e)







data = merged_data
docids = set(data[subset]['docid2text'])






true_docid2entities = {docid: list() for docid in docids}

for docid in docids:
    true_entities = data[subset]['docid2entities'][docid]
    #
    # Make sure that we don't add repeated entities, that is,
    # entities with the same span and the same type.
    #
    unique_added_entities = set()
    simplified_true_entities = list()
    for e in true_entities:
        ste = dict()
        ste['span_type'] = '{}-{};{}'.format(e.begin, e.end, e.typ)
        ste['code'] = e.code
        assert ste['span_type'] not in unique_added_entities
        unique_added_entities.add(ste['span_type'])
        simplified_true_entities.append(ste)
        #
        UNIQUE_ENTITY_TYPES.add(e.typ)
    #
    true_docid2entities[docid] = simplified_true_entities









#
# For NER evaluation, the variable "true_docid2entities" is correct.
#
# However, for NEL evaluation we should only consider the entities that
# are present in the NEL TSV file, and therefore we need to remove the
# entities with missing codes (that is, entities that did not contain
# a gold standard NEL annotation).
#
true_nel_docid2entities = {docid: list() for docid in docids}

for docid in docids:
    true_entities = true_docid2entities[docid]
    #
    # Remove entities that are with missing NEL codes, because these
    # do not count for the entity linking evaluation.
    #
    true_nel_docid2entities[docid] = [e for e in true_entities if e['code'] != MISSING_CODE]




















pred = read_file(pred_filepath)
pred_lines = pred.splitlines()





















header = pred_lines[0].split('\t')
assert len(header) == 6















#
# When loading the predictions from the TSV file make sure that the
# entity types (labels, classes) are translated to English.
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
    #
    # Keep the new English labels equal.
    #
    'SYMPTOM': 'SYMPTOM',
    'PROCEDURE': 'PROCEDURE',
    'DISEASE': 'DISEASE',
    'CHEMICAL': 'CHEMICAL',
    'PROTEIN': 'PROTEIN',
}











#
# Each line in the TSV file corresponds to one entity to add
# in the respective DocID (file name).
#
# Make sure that we don't add repeated entities, that is,
# entities with the same span and the same type (in the
# same DocID).
#
unique_added_entities = set()
pred_docid2entities = {docid: list() for docid in docids}

for i, line in enumerate(pred_lines[1:]):
    docid, start, end, typ, code, _ = line.split('\t')
    typ = spanish_to_english[typ]
    #
    e = dict()
    e['span_type'] = '{}-{};{}'.format(start, end, typ)
    #
    if code.lower() in {'nomap', 'null', '<null>'}:
        code = 'NO_CODE'
    #
    e['code'] = code
    #
    unique_entity = '{};{}'.format(docid, e['span_type'])
    #
    if unique_entity in unique_added_entities:
        print('Repetition of {} (line {}, ignoring)'.format(repr(unique_entity), i))
    else:
        unique_added_entities.add(unique_entity)
        #
        UNIQUE_ENTITY_TYPES.add(typ)
        #
        pred_docid2entities[docid].append(e)

print('\n\n')





















if UNIQUE_ENTITY_TYPES == {'CHEMICAL', 'PROTEIN', 'DISEASE', 'PROCEDURE', 'SYMPTOM'}:
    #
    # Summarized results for pretty table.
    #
    summarized_results = {
        'ner': {
            'counts': 0,
            'entities': {
                'CHEMICAL':  {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'PROTEIN':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'DISEASE':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'PROCEDURE': {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'SYMPTOM':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
            },
            'micro': {'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
            'macro': {'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
        },
        'nel_including_composite_mentions': {
            'counts': 0,
            'entities': {
                'CHEMICAL':  {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'PROTEIN':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'DISEASE':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'PROCEDURE': {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'SYMPTOM':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
            },
            'micro': {'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
            'macro': {'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
        },
        'nel_excluding_composite_mentions': {
            'counts': 0,
            'entities': {
                'CHEMICAL':  {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'PROTEIN':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'DISEASE':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'PROCEDURE': {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
                'SYMPTOM':   {'counts': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
            },
            'micro': {'TP': 0, 'FP': 0, 'FN': 0, 'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
            'macro': {'p': 0.0, 'r': 0.0, 'f': 0.0, 'a': 0.0},
        },
    }



















def fpr(tp, fn, fp):
    if tp == 0:
        f = 0.0
        p = 0.0
        r = 0.0
        acc = 0.0
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r)
        #
        # Accuracy is the ratio between "Correct classifications" and
        # "All classifications".
        # https://en.wikipedia.org/wiki/Accuracy_and_precision
        #
        # "All classifications" is the sum of "Correct classifications"
        # and "Incorrect classifications".
        #
        # Regarding the evaluation of Entity Linking, note that Accuracy
        # only matches F1-score when gold standard NER entities are
        # employed. In this scenario, an "Incorrect classification" will
        # add a FN (False Negative) and a FP (False Positive).
        #
        acc = 2*tp / (2*tp + fp + fn)
    return f, p, r, acc

















#
# Now that we have the TRUE (gold standard) and PRED (predictions)
# entities loaded we can proceed with the NER and entity linking evaluation.
#
# Let's start with the NER evaluation.
#
print('NER evaluation')
print('==============')

total_tp = 0
total_fn = 0
total_fp = 0

tp_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}
fn_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}
fp_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}

f1_per_class        = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
precision_per_class = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
recall_per_class    = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
acc_per_class       = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}

for docid in docids:
    true_entities = true_docid2entities[docid]
    pred_entities = pred_docid2entities[docid]
    #
    # Micro-average.
    #
    true_entities_set = {e['span_type'] for e in true_entities}
    pred_entities_set = {e['span_type'] for e in pred_entities}
    #
    true_positives = true_entities_set.intersection(pred_entities_set)
    tp = len(true_positives)
    #
    false_negatives = true_entities_set.difference(pred_entities_set)
    fn = len(false_negatives)
    #
    false_positives = pred_entities_set.difference(true_entities_set)
    fp = len(false_positives)
    #
    total_tp += tp
    total_fn += fn
    total_fp += fp
    #
    # Macro-average.
    #
    true_entities_per_class = {c: list() for c in UNIQUE_ENTITY_TYPES}
    for e in true_entities:
        span, typ = e['span_type'].split(';')
        code = e['code']
        true_entities_per_class[typ].append({'span': span, 'code': code})
    #
    pred_entities_per_class = {c: list() for c in UNIQUE_ENTITY_TYPES}
    for e in pred_entities:
        span, typ = e['span_type'].split(';')
        code = e['code']
        pred_entities_per_class[typ].append({'span': span, 'code': code})
    #
    for c in UNIQUE_ENTITY_TYPES:
        true_entities = true_entities_per_class[c]
        pred_entities = pred_entities_per_class[c]
        #
        true_entities_set = {e['span'] for e in true_entities}
        pred_entities_set = {e['span'] for e in pred_entities}
        #
        true_positives = true_entities_set.intersection(pred_entities_set)
        tp = len(true_positives)
        #
        false_negatives = true_entities_set.difference(pred_entities_set)
        fn = len(false_negatives)
        #
        false_positives = pred_entities_set.difference(true_entities_set)
        fp = len(false_positives)
        #
        tp_per_class[c] += tp
        fn_per_class[c] += fn
        fp_per_class[c] += fp

micro_f1, micro_precision, micro_recall, micro_acc = fpr(total_tp, total_fn, total_fp)

for c in UNIQUE_ENTITY_TYPES:
    f, p, r, acc = fpr(tp_per_class[c], fn_per_class[c], fp_per_class[c])
    f1_per_class[c] = f
    precision_per_class[c] = p
    recall_per_class[c] = r
    acc_per_class[c] = acc

macro_f1 = np.average(list(f1_per_class.values()))
macro_precision = np.average(list(precision_per_class.values()))
macro_recall = np.average(list(recall_per_class.values()))
macro_acc = np.average(list(acc_per_class.values()))

print()
print('Micro-average F1-score:  {:.6f}'.format(micro_f1))
print('Micro-average Precision: {:.6f}'.format(micro_precision))
print('Micro-average Recall:    {:.6f}'.format(micro_recall))
print('Micro-average Accuracy:  {:.6f}'.format(micro_acc))
print('Total TP: {:>5d}'.format(total_tp))
print('Total FN: {:>5d}'.format(total_fn))
print('Total FP: {:>5d}'.format(total_fp))
print()
print('Macro-average F1-score:  {:.6f}'.format(macro_f1))
print('Macro-average Precision: {:.6f}'.format(macro_precision))
print('Macro-average Recall:    {:.6f}'.format(macro_recall))
print('Macro-average Accuracy:  {:.6f}'.format(macro_acc))
print()
print('  Evaluation per entity type (class)')
print('  ----------------------------------')
for c in sorted(UNIQUE_ENTITY_TYPES):
    print('    Entity type {}:'.format(repr(c)))
    print('      F1-score:  {:.6f}'.format(f1_per_class[c]))
    print('      Precision: {:.6f}'.format(precision_per_class[c]))
    print('      Recall:    {:.6f}'.format(recall_per_class[c]))
    print('      Accuracy:  {:.6f}'.format(acc_per_class[c]))
    print('      TP: {:>5d}'.format(tp_per_class[c]))
    print('      FN: {:>5d}'.format(fn_per_class[c]))
    print('      FP: {:>5d}'.format(fp_per_class[c]))
    print('    --------------------------------')

print('\n')







if UNIQUE_ENTITY_TYPES == {'CHEMICAL', 'PROTEIN', 'DISEASE', 'PROCEDURE', 'SYMPTOM'}:
    #
    task = 'ner'
    #
    summarized_results[task]['counts'] += total_tp + total_fn
    #
    for c in UNIQUE_ENTITY_TYPES:
        summarized_results[task]['entities'][c]['counts'] = tp_per_class[c] + fn_per_class[c]
        summarized_results[task]['entities'][c]['TP'] = tp_per_class[c]
        summarized_results[task]['entities'][c]['FP'] = fp_per_class[c]
        summarized_results[task]['entities'][c]['FN'] = fn_per_class[c]
        summarized_results[task]['entities'][c]['p'] = precision_per_class[c]
        summarized_results[task]['entities'][c]['r'] = recall_per_class[c]
        summarized_results[task]['entities'][c]['f'] = f1_per_class[c]
        summarized_results[task]['entities'][c]['a'] = acc_per_class[c]
    #
    summarized_results[task]['micro']['TP'] = total_tp
    summarized_results[task]['micro']['FP'] = total_fp
    summarized_results[task]['micro']['FN'] = total_fn
    summarized_results[task]['micro']['p'] = micro_precision
    summarized_results[task]['micro']['r'] = micro_recall
    summarized_results[task]['micro']['f'] = micro_f1
    summarized_results[task]['micro']['a'] = micro_acc
    #
    summarized_results[task]['macro']['p'] = macro_precision
    summarized_results[task]['macro']['r'] = macro_recall
    summarized_results[task]['macro']['f'] = macro_f1
    summarized_results[task]['macro']['a'] = macro_acc





















def codes_are_equal(code1, code2):
    #
    # Before comparing the codes let's convert the string to lowercase
    # because the PharmaCoNER uses the "chebi:" and "CHEBI:" prefix
    # identifiers interchangeably.
    #
    code1 = code1.lower()
    code2 = code2.lower()
    #
    code1_set = set(code1.split('+'))
    code2_set = set(code2.split('+'))
    return code1_set == code2_set


























#
# Let's do entity linking evaluation (considering composite mentions).
#
print('Entity linking (including composite mentions)')
print('=============================================')

total_tp = 0
total_fn = 0
total_fp = 0

tp_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}
fn_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}
fp_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}

f1_per_class        = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
precision_per_class = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
recall_per_class    = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
acc_per_class       = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}

for docid in docids:
    true_entities = true_nel_docid2entities[docid]
    pred_entities = pred_docid2entities[docid]
    #
    # Micro-average.
    #
    true_entity_to_code = {e['span_type']: e['code'] for e in true_entities}
    pred_entity_to_code = {e['span_type']: e['code'] for e in pred_entities}
    #
    for te in true_entity_to_code:
        if te in pred_entity_to_code:
            code1 = true_entity_to_code[te]
            code2 = pred_entity_to_code[te]
            if codes_are_equal(code1, code2):
                total_tp += 1
            else:
                total_fn += 1
                total_fp += 1
        else:
            total_fn += 1
    for pe in pred_entity_to_code:
        if pe not in true_entity_to_code:
            total_fp += 1
    #
    # Macro-average.
    #
    true_entities_per_class = {c: list() for c in UNIQUE_ENTITY_TYPES}
    for e in true_entities:
        span, typ = e['span_type'].split(';')
        code = e['code']
        true_entities_per_class[typ].append({'span': span, 'code': code})
    #
    pred_entities_per_class = {c: list() for c in UNIQUE_ENTITY_TYPES}
    for e in pred_entities:
        span, typ = e['span_type'].split(';')
        code = e['code']
        pred_entities_per_class[typ].append({'span': span, 'code': code})
    #
    for c in UNIQUE_ENTITY_TYPES:
        true_entities = true_entities_per_class[c]
        pred_entities = pred_entities_per_class[c]
        #
        true_entity_to_code = {e['span']: e['code'] for e in true_entities}
        pred_entity_to_code = {e['span']: e['code'] for e in pred_entities}
        #
        for te in true_entity_to_code:
            if te in pred_entity_to_code:
                code1 = true_entity_to_code[te]
                code2 = pred_entity_to_code[te]
                if codes_are_equal(code1, code2):
                    tp_per_class[c] += 1
                else:
                    fn_per_class[c] += 1
                    fp_per_class[c] += 1
            else:
                fn_per_class[c] += 1
        for pe in pred_entity_to_code:
            if pe not in true_entity_to_code:
                fp_per_class[c] += 1

micro_f1, micro_precision, micro_recall, micro_acc = fpr(total_tp, total_fn, total_fp)

for c in UNIQUE_ENTITY_TYPES:
    f, p, r, acc = fpr(tp_per_class[c], fn_per_class[c], fp_per_class[c])
    f1_per_class[c] = f
    precision_per_class[c] = p
    recall_per_class[c] = r
    acc_per_class[c] = acc

macro_f1 = np.average(list(f1_per_class.values()))
macro_precision = np.average(list(precision_per_class.values()))
macro_recall = np.average(list(recall_per_class.values()))
macro_acc = np.average(list(acc_per_class.values()))

print()
print('Micro-average F1-score:  {:.6f}'.format(micro_f1))
print('Micro-average Precision: {:.6f}'.format(micro_precision))
print('Micro-average Recall:    {:.6f}'.format(micro_recall))
print('Micro-average Accuracy:  {:.6f}'.format(micro_acc))
print('Total TP: {:>5d}'.format(total_tp))
print('Total FN: {:>5d}'.format(total_fn))
print('Total FP: {:>5d}'.format(total_fp))
print()
print('Macro-average F1-score:  {:.6f}'.format(macro_f1))
print('Macro-average Precision: {:.6f}'.format(macro_precision))
print('Macro-average Recall:    {:.6f}'.format(macro_recall))
print('Macro-average Accuracy:  {:.6f}'.format(macro_acc))
print()
print('  Evaluation per entity type (class)')
print('  ----------------------------------')
for c in sorted(UNIQUE_ENTITY_TYPES):
    print('    Entity type {}:'.format(repr(c)))
    print('      F1-score:  {:.6f}'.format(f1_per_class[c]))
    print('      Precision: {:.6f}'.format(precision_per_class[c]))
    print('      Recall:    {:.6f}'.format(recall_per_class[c]))
    print('      Accuracy:  {:.6f}'.format(acc_per_class[c]))
    print('      TP: {:>5d}'.format(tp_per_class[c]))
    print('      FN: {:>5d}'.format(fn_per_class[c]))
    print('      FP: {:>5d}'.format(fp_per_class[c]))
    print('    --------------------------------')

print('\n')




















if UNIQUE_ENTITY_TYPES == {'CHEMICAL', 'PROTEIN', 'DISEASE', 'PROCEDURE', 'SYMPTOM'}:
    #
    task = 'nel_including_composite_mentions'
    #
    summarized_results[task]['counts'] += total_tp + total_fn
    #
    for c in UNIQUE_ENTITY_TYPES:
        summarized_results[task]['entities'][c]['counts'] = tp_per_class[c] + fn_per_class[c]
        summarized_results[task]['entities'][c]['TP'] = tp_per_class[c]
        summarized_results[task]['entities'][c]['FP'] = fp_per_class[c]
        summarized_results[task]['entities'][c]['FN'] = fn_per_class[c]
        summarized_results[task]['entities'][c]['p'] = precision_per_class[c]
        summarized_results[task]['entities'][c]['r'] = recall_per_class[c]
        summarized_results[task]['entities'][c]['f'] = f1_per_class[c]
        summarized_results[task]['entities'][c]['a'] = acc_per_class[c]
    #
    summarized_results[task]['micro']['TP'] = total_tp
    summarized_results[task]['micro']['FP'] = total_fp
    summarized_results[task]['micro']['FN'] = total_fn
    summarized_results[task]['micro']['p'] = micro_precision
    summarized_results[task]['micro']['r'] = micro_recall
    summarized_results[task]['micro']['f'] = micro_f1
    summarized_results[task]['micro']['a'] = micro_acc
    #
    summarized_results[task]['macro']['p'] = macro_precision
    summarized_results[task]['macro']['r'] = macro_recall
    summarized_results[task]['macro']['f'] = macro_f1
    summarized_results[task]['macro']['a'] = macro_acc







































def is_composite_mention(code):
    if '+' in code:
        return True
    else:
        return False












#
# Let's do entity linking evaluation (excluding composite mentions).
#
print('Entity linking (excluding composite mentions)')
print('=============================================')

total_tp = 0
total_fn = 0
total_fp = 0

tp_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}
fn_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}
fp_per_class = {c: 0 for c in UNIQUE_ENTITY_TYPES}

f1_per_class        = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
precision_per_class = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
recall_per_class    = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}
acc_per_class       = {c: 0.0 for c in UNIQUE_ENTITY_TYPES}

for docid in docids:
    true_entities = true_nel_docid2entities[docid]
    pred_entities = pred_docid2entities[docid]
    #
    # Ignore composite mentions (according to the gold standard).
    #
    composite_mentions = {e['span_type'] for e in true_entities if is_composite_mention(e['code'])}
    true_entities_filtered = [e for e in true_entities if e['span_type'] not in composite_mentions]
    pred_entities_filtered = [e for e in pred_entities if e['span_type'] not in composite_mentions]
    #
    # Micro-average.
    #
    true_entity_to_code = {e['span_type']: e['code'] for e in true_entities_filtered}
    pred_entity_to_code = {e['span_type']: e['code'] for e in pred_entities_filtered}
    #
    for te in true_entity_to_code:
        if te in pred_entity_to_code:
            code1 = true_entity_to_code[te]
            code2 = pred_entity_to_code[te]
            if codes_are_equal(code1, code2):
                total_tp += 1
            else:
                total_fn += 1
                total_fp += 1
        else:
            total_fn += 1
    for pe in pred_entity_to_code:
        if pe not in true_entity_to_code:
            total_fp += 1
    #
    # Macro-average.
    #
    true_entities_per_class = {c: list() for c in UNIQUE_ENTITY_TYPES}
    for e in true_entities_filtered:
        span, typ = e['span_type'].split(';')
        code = e['code']
        true_entities_per_class[typ].append({'span': span, 'code': code})
    #
    pred_entities_per_class = {c: list() for c in UNIQUE_ENTITY_TYPES}
    for e in pred_entities_filtered:
        span, typ = e['span_type'].split(';')
        code = e['code']
        pred_entities_per_class[typ].append({'span': span, 'code': code})
    #
    for c in UNIQUE_ENTITY_TYPES:
        true_entities = true_entities_per_class[c]
        pred_entities = pred_entities_per_class[c]
        #
        true_entity_to_code = {e['span']: e['code'] for e in true_entities}
        pred_entity_to_code = {e['span']: e['code'] for e in pred_entities}
        #
        for te in true_entity_to_code:
            if te in pred_entity_to_code:
                code1 = true_entity_to_code[te]
                code2 = pred_entity_to_code[te]
                if codes_are_equal(code1, code2):
                    tp_per_class[c] += 1
                else:
                    fn_per_class[c] += 1
                    fp_per_class[c] += 1
            else:
                fn_per_class[c] += 1
        for pe in pred_entity_to_code:
            if pe not in true_entity_to_code:
                fp_per_class[c] += 1

micro_f1, micro_precision, micro_recall, micro_acc = fpr(total_tp, total_fn, total_fp)

for c in UNIQUE_ENTITY_TYPES:
    f, p, r, acc = fpr(tp_per_class[c], fn_per_class[c], fp_per_class[c])
    f1_per_class[c] = f
    precision_per_class[c] = p
    recall_per_class[c] = r
    acc_per_class[c] = acc

macro_f1 = np.average(list(f1_per_class.values()))
macro_precision = np.average(list(precision_per_class.values()))
macro_recall = np.average(list(recall_per_class.values()))
macro_acc = np.average(list(acc_per_class.values()))

print()
print('Micro-average F1-score:  {:.6f}'.format(micro_f1))
print('Micro-average Precision: {:.6f}'.format(micro_precision))
print('Micro-average Recall:    {:.6f}'.format(micro_recall))
print('Micro-average Accuracy:  {:.6f}'.format(micro_acc))
print('Total TP: {:>5d}'.format(total_tp))
print('Total FN: {:>5d}'.format(total_fn))
print('Total FP: {:>5d}'.format(total_fp))
print()
print('Macro-average F1-score:  {:.6f}'.format(macro_f1))
print('Macro-average Precision: {:.6f}'.format(macro_precision))
print('Macro-average Recall:    {:.6f}'.format(macro_recall))
print('Macro-average Accuracy:  {:.6f}'.format(macro_acc))
print()
print('  Evaluation per entity type (class)')
print('  ----------------------------------')
for c in sorted(UNIQUE_ENTITY_TYPES):
    print('    Entity type {}:'.format(repr(c)))
    print('      F1-score:  {:.6f}'.format(f1_per_class[c]))
    print('      Precision: {:.6f}'.format(precision_per_class[c]))
    print('      Recall:    {:.6f}'.format(recall_per_class[c]))
    print('      Accuracy:  {:.6f}'.format(acc_per_class[c]))
    print('      TP: {:>5d}'.format(tp_per_class[c]))
    print('      FN: {:>5d}'.format(fn_per_class[c]))
    print('      FP: {:>5d}'.format(fp_per_class[c]))
    print('    --------------------------------')

print('\n')























if UNIQUE_ENTITY_TYPES == {'CHEMICAL', 'PROTEIN', 'DISEASE', 'PROCEDURE', 'SYMPTOM'}:
    #
    task = 'nel_excluding_composite_mentions'
    #
    summarized_results[task]['counts'] += total_tp + total_fn
    #
    for c in UNIQUE_ENTITY_TYPES:
        summarized_results[task]['entities'][c]['counts'] = tp_per_class[c] + fn_per_class[c]
        summarized_results[task]['entities'][c]['TP'] = tp_per_class[c]
        summarized_results[task]['entities'][c]['FP'] = fp_per_class[c]
        summarized_results[task]['entities'][c]['FN'] = fn_per_class[c]
        summarized_results[task]['entities'][c]['p'] = precision_per_class[c]
        summarized_results[task]['entities'][c]['r'] = recall_per_class[c]
        summarized_results[task]['entities'][c]['f'] = f1_per_class[c]
        summarized_results[task]['entities'][c]['a'] = acc_per_class[c]
    #
    summarized_results[task]['micro']['TP'] = total_tp
    summarized_results[task]['micro']['FP'] = total_fp
    summarized_results[task]['micro']['FN'] = total_fn
    summarized_results[task]['micro']['p'] = micro_precision
    summarized_results[task]['micro']['r'] = micro_recall
    summarized_results[task]['micro']['f'] = micro_f1
    summarized_results[task]['micro']['a'] = micro_acc
    #
    summarized_results[task]['macro']['p'] = macro_precision
    summarized_results[task]['macro']['r'] = macro_recall
    summarized_results[task]['macro']['f'] = macro_f1
    summarized_results[task]['macro']['a'] = macro_acc



































#
# Finally, we present a summarized table with all the results.
#


if UNIQUE_ENTITY_TYPES == {'CHEMICAL', 'PROTEIN', 'DISEASE', 'PROCEDURE', 'SYMPTOM'}:
    #
    print('\n\n\n')
    #
    draw = (
        '  ================================================================================\n'
        '  === NER (named entity recognition) strict evaluation (exact match)           ===\n'
        '  ================================================================================\n'
        '  9999999  entities      TP     FP     FN  Precision    Recall  F1-score  Accuracy\n'
        '  -----------------  --------------------  -----------------------------  --------\n'
        '  9999999  CHEMICAL  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   PROTEIN  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   DISEASE  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999 PROCEDURE  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   SYMPTOM  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  -----------------  --------------------  -----------------------------  --------\n'
        '     micro-averaged  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '     macro-averaged                         0.999999  0.999999  0.999999  0.999999\n'
        '  ================================================================================\n'
        '\n'
        '\n'
        '  ================================================================================\n'
        '  === NEL (named entity linking) evaluation including composite mentions       ===\n'
        '  ================================================================================\n'
        '  9999999  entities      TP     FP     FN  Precision    Recall  F1-score  Accuracy\n'
        '  -----------------  --------------------  -----------------------------  --------\n'
        '  9999999  CHEMICAL  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   PROTEIN  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   DISEASE  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999 PROCEDURE  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   SYMPTOM  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  -----------------  --------------------  -----------------------------  --------\n'
        '     micro-averaged  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '     macro-averaged                         0.999999  0.999999  0.999999  0.999999\n'
        '  ================================================================================\n'
        '\n'
        '\n'
        '  ================================================================================\n'
        '  === NEL (named entity linking) evaluation excluding composite mentions       ===\n'
        '  ================================================================================\n'
        '  9999999  entities      TP     FP     FN  Precision    Recall  F1-score  Accuracy\n'
        '  -----------------  --------------------  -----------------------------  --------\n'
        '  9999999  CHEMICAL  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   PROTEIN  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   DISEASE  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999 PROCEDURE  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  9999999   SYMPTOM  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '  -----------------  --------------------  -----------------------------  --------\n'
        '     micro-averaged  999999 999999 999999   0.999999  0.999999  0.999999  0.999999\n'
        '     macro-averaged                         0.999999  0.999999  0.999999  0.999999\n'
        '  ================================================================================\n'
        '\n'
    )
    #
    draw = draw.replace('9999999', '{:7d}')
    draw = draw.replace('0.999999', '{:8.6f}')
    draw = draw.replace('999999', '{:6d}')
    #
    # Concatenate all the values into a list.
    #
    s = summarized_results
    #
    values = list()
    #
    for task in ['ner', 'nel_including_composite_mentions',
                        'nel_excluding_composite_mentions']:
        values.append(s[task]['counts'])
        for c in ['CHEMICAL', 'PROTEIN', 'DISEASE', 'PROCEDURE', 'SYMPTOM']:
            values.extend(s[task]['entities'][c].values())
        #
        values.extend(s[task]['micro'].values())
        values.extend(s[task]['macro'].values())
    #
    # Print final summarized table.
    #
    draw = draw.format(*values)
    #
    print(draw)
