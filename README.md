# Multi-Head-CRF

This repository contains the implementation for the Multi-Head-CRF model as described in:

**Multi-head CRF classifier for biomedical multi-class Named Entity Recognition on Spanish clinical notes**

*Authors:*
- Richard A A Jonker ([ORCID: 0000-0002-3806-6940](https://orcid.org/0000-0002-3806-6940))
- Tiago Almeida ([ORCID: 0000-0002-4258-3350](https://orcid.org/0000-0002-4258-3350))
- Rui Antunes ([ORCID: 0000-0003-3533-8872](https://orcid.org/0000-0003-3533-8872))
- João R Almeida ([ORCID: 0000-0003-0729-2264](https://orcid.org/0000-0003-0729-2264))
- Sérgio Matos ([ORCID: 0000-0003-1941-3983](https://orcid.org/0000-0003-1941-3983))

## Overview

The identification of medical concepts from clinical narratives has a large interest in the biomedical scientific community due to its importance in treatment improvements or drug development research. Biomedical Named Entity Recognition (NER) in clinical texts is crucial for automated information extraction, facilitating patient record analysis, drug development, and medical research. Traditional approaches often focus on single-class NER tasks, yet recent advancements emphasize the necessity of addressing multi-class scenarios, particularly in complex biomedical domains. This paper proposes a strategy to integrate a Multi-Head Conditional Random Field (CRF) classifier for multi-class NER in Spanish clinical documents. Our methodology overcomes overlapping entity instances of different types, a common challenge in traditional NER methodologies, by using a multi-head CRF model. This architecture enhances computational efficiency and ensures scalability for multi-class NER tasks, maintaining high performance. By combining four diverse datasets, SympTEMIST, MedProcNER, DisTEMIST, and PharmaCoNER, we expand the scope of NER to encompass five classes: symptoms, procedures, diseases, chemicals, and proteins. To the best of our knowledge, these datasets combined create the largest Spanish multi-class dataset focusing on biomedical entity recognition and linking for clinical notes, which is important to train a biomedical model in Spanish. We also provide entity linking to the multi-lingual SNOMED CT vocabulary, with the eventual goal of performing biomedical relation extraction. Through experimentation and evaluation of Spanish clinical documents, our strategy provides competitive results against single-class NER models. For NER, our system achieves a combined micro-averaged F1-score of 78.73, with clinical mentions normalized to SNOMED CT with an end-to-end F1-score of 54.51.

## Setup

Create a python environment.
```
python -m venv venv
PIP=venv/bin/pip
$PIP install --upgrade pip
$PIP install -r requirements.txt
```

## Dataset

The dataset used in this work merges four separate datasets:
- SymptEMIST: [Zenodo](https://zenodo.org/records/10635215)
- DisTEMIST: [Zenodo](https://zenodo.org/records/7614764)
- MedProcNER: [Zenodo](https://zenodo.org/records/8224056)
- PharmaCoNER: [Zenodo](https://zenodo.org/records/4270158)

All datasets are licensed under CC4.

To set up the dataset, a script is provided (`dataset/download_dataset.sh`) that downloads these datasets, prepares them in the correct format, and merges them to create a unified dataset.

Alternatively, the dataset is available on:
- [HuggingFace](https://huggingface.co/datasets/IEETA/SPACCC-Spanish-NER)
- [Zenodo](https://zenodo.org/records/11174163)

This step is required if you wish to run the Named Entity Linking or Evaluation. 

## Named Entity Recognition

To train a model, use the following command:

```bash
python hf_trainer_MHC_full.py lcampillos/roberta-es-clinical-trials-ner --augmentation random --number_of_layer_per_head 3 --context 32 --epochs 60 --batch 16 --percentage_tags 0.25 --aug_prob 0.5 --classes SYMPTOM PROCEDURE DISEASE PROTEIN CHEMICAL
```

- `lcampillos/roberta-es-clinical-trials-ner`: Model checkpoint.
- `--number_of_layer_per_head`: Number of hidden layers to use in each CRF head (Good options: 1-3).
- `--context`: Context size for splitting documents exceeding the 512 token limit (Good options: 2 or 32).
- `--epochs`: Number of epochs to train.
- `--batch`: Batch size.
- `--augmentation`: Augmentation strategy (None, 'random', or 'unk').
- `--aug_prob`: Probability to apply augmentation to a sample.
- `--percentage_tags`: Percentage of tokens to change.
- `--classes`: Classes to train, must be a combination of: SYMPTOM PROCEDURE DISEASE PROTEIN CHEMICAL.
- `--val`: Whether to use a validation dataset; otherwise, the test dataset is utilized.

To run inference for the model, we provide an inference file, which will conduct inference over the test dataset by default:
`python inference.py --checkpoint MODEL_CHECKPOINT`


## Named Entity Linking


In order to utilize the SNOMED CT terminology, it is necessary to create a UMLS account and download the [file](https://download.nlm.nih.gov/umls/kss/IHTSDO20190131/SnomedCT_SpanishRelease-es_PRODUCTION_20190430T120000Z.zip). This folder is expected to be extracted into the embeddings directory. Although we do not supply the original resource, we do supply all the embeddings used for SNOMED CT and the various gazetteers, which are available [here](https://zenodo.org/records/11174163), with a script available in `embeddings/download_embeddings.sh`

In order to build the embeddings, it is required to run the `embeddings/prepare_jsonl_for_embedding.py` script, which will create jsonl files from the various gazetteers.

In order to build the embeddings it is required to run `embeddings/build_embeddings_index.py`.

`python build_embeddings_index.py snomedCT.jsonl`

With these embeddings we can conduct normalization (in `src`).

```python normalize.py INPUT_RUN --t 0.6 --use_gazetteer False --output_folder runs```

Were `--t` is the the threshold of acceptance, and `--use_gazetteer` is whether or not to use the gazetteers to normalize. 

## Evaluation

The evaluation (NER and entity linking) can be run in the [evaluation/](evaluation/) directory as follows:

`python3  evaluation.py  train/test  PREDICTIONS_FILE.tsv`


## TODO

- Verify Pipeline
- model download script



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
