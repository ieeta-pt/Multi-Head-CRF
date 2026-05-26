from collections import defaultdict
import torch

# helper function taken from chatgpt
def group_spans(lst):
    if not lst: return [] # <--- Add this safety check
    spans = []
    start = lst[0]
    end = lst[0]

    for num in lst[1:]:
        if num == end + 1:
            end = num
        else:
            spans.append((start, end))
            start = num
            end = num

    spans.append((start, end))
    return spans


def ensemble_token_level(tokens):
    # ensembles based on the BIO tags and makes a mjority vote on the tokens, very naive approach
    # assumed to be run before the decoder
    # tokens are assumed to be a list of tensors, where each item of the list corresponds to a different run
    number_of_runs = len(tokens)
    majority_vote = int(number_of_runs / 2) if number_of_runs % 2 == 0 else int(number_of_runs / 2) + 1
    # initialize array of dictionaries
    counter = [{0: 0, 1: 0, 2: 0} for i in range(len(tokens[0]))]
    # counter of tokens
    for model_output in tokens:
        for i, tok in enumerate(model_output):
            counter[i][tok] += 1
    # vote max
    outs = [max(i, key=i.get) if max(i.values()) >= majority_vote else 0 for i in counter]
    return torch.tensor(outs)


def ensemble_span_level(entities, text):
    # entiteies is a list of outputs from the decoder and text is the raw text
    # ensembles on a span level. If spans are (130, 140) and  (135, 145) then the ensemble will be (135, 140)
    # has a bad implementation, expands the spans and counts each item and converts back to spans
    number_of_runs = len(entities)
    majority_vote = int(number_of_runs / 2) if number_of_runs % 2 == 0 else int(number_of_runs / 2) + 1

    # expan the spans
    counter = defaultdict(int)
    for model_output in entities:
        for span in model_output['span']:
            for i in range(span[0], span[1]+1):
                counter[i] += 1

    outs = [i for i in sorted(counter.keys()) if counter[i] >= majority_vote]
    grouped_spans = group_spans(outs)

    final_output_span = []
    final_output_en = []
    # reconstruct entities
    for span in grouped_spans:
        final_output_span.append(span)
        final_output_en.append(text[span[0]:span[1]])
    return {"span": list(final_output_span), "text": list(final_output_en)}


def ensemble_entity_level(entities, text):
    # entiteies is a list of outputs from the decoder and text is the raw text
    # ensembles on an entity level. each entity needs to be exact matching and voting is done based on the span ranges.
    number_of_runs = len(entities)
    majority_vote = int(number_of_runs / 2) if number_of_runs % 2 == 0 else int(number_of_runs / 2) + 1
    counter = defaultdict(int)

    # use spans as entity keys
    for model_output in entities:
        
        if len(model_output['span'])>0 and isinstance(model_output['span'][0], list):
            model_output['span'] = list(map(tuple, model_output['span']))
        for span in set(model_output['span']):
            counter[span] += 1

    outs = [i for i in counter.keys() if counter[i] >= majority_vote]

    # enity level agreement
    final_output_span = []
    final_output_en = []
    # reconstruct entities
    for span in sorted(outs, key=lambda x: x[0]):
        final_output_span.append(span)
        final_output_en.append(text[span[0]:span[1]])
    return {"span": list(final_output_span), "text": list(final_output_en)}