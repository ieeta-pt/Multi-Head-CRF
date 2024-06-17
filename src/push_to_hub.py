from model.modeling_multiheadcrf import RobertaMultiHeadCRFModel, BertMultiHeadCRFModel
from model.configuration_multiheadcrf import MultiHeadCRFConfig
from transformers import AutoTokenizer
import argparse

tokenizer = AutoTokenizer.from_pretrained("lcampillos/roberta-es-clinical-trials-ner")

parser = argparse.ArgumentParser(description="")
parser.add_argument("token", type=str, default=None)

args = parser.parse_args()


name_pairs = [
    ("../models/lcampillos-None-C32-H1-E60-Arandom-%0.25-P0.5-42-checkpoint-1080/", "IEETA/RobertaMultiHeadCRF-C32-0"),
    ("../models/lcampillos-None-C32-H3-E60-ANone-%0.0-P0.0-42-checkpoint-1080/", "IEETA/RobertaMultiHeadCRF-C32-1"),
    ("../models/lcampillos-None-C32-H3-E60-Arandom-%0.5-P0.75-42-checkpoint-1080/", "IEETA/RobertaMultiHeadCRF-C32-2"),
    ("../models/lcampillos-None-C32-H3-E60-Arandom-%0.25-P0.5-42-checkpoint-1080/", "IEETA/RobertaMultiHeadCRF-C32-3"),
]

for path, rep_name in name_pairs:

    config = MultiHeadCRFConfig.from_json_file(f"{path}/config.json")
    config.architectures[0] = "RobertaMultiHeadCRFModel"

    model = RobertaMultiHeadCRFModel(config)


    from safetensors import safe_open
    import os
    state_dict = {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key.replace("bert.", "roberta.")] = f.get_tensor(key)
    print(model.load_state_dict(state_dict))

    config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")

    model.push_to_hub(rep_name, token=args.token)
    config.push_to_hub(rep_name, token=args.token)
    tokenizer.push_to_hub(rep_name, token=args.token)

