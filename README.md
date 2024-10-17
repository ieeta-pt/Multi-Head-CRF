# Multi-Head-CRF

This repository contains the implementation for the Multi-Head-CRF model as described in:

**Multi-head CRF classifier for biomedical multi-class Named Entity Recognition on Spanish clinical notes**

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
- [Hugging Face](https://huggingface.co/datasets/IEETA/SPACCC-Spanish-NER)
- [Zenodo](https://zenodo.org/records/11174163)

This step is required if you wish to run the Named Entity Linking or Evaluation. 

## Named Entity Recognition

Go to src directory
```bash
cd src
```

To train a model, use the following command:

```bash
python hf_trainer.py lcampillos/roberta-es-clinical-trials-ner --augmentation random --number_of_layer_per_head 3 --context 32 --epochs 60 --batch 16 --percentage_tags 0.25 --aug_prob 0.5 --classes SYMPTOM PROCEDURE DISEASE PROTEIN CHEMICAL
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
`python inference.py MODEL_CHECKPOINT`


We also provide several of our best performing models available on [Hugging Face](https://huggingface.co/collections/IEETA/multi-head-crf-classifier-6641e5907fc7e7c22bc4a85d).

- [IEETA/RobertaMultiHeadCRF-C32-0](https://huggingface.co/IEETA/RobertaMultiHeadCRF-C32-0)
- [IEETA/RobertaMultiHeadCRF-C32-1](https://huggingface.co/IEETA/RobertaMultiHeadCRF-C32-1)
- [IEETA/RobertaMultiHeadCRF-C32-2](https://huggingface.co/IEETA/RobertaMultiHeadCRF-C32-2)
- [IEETA/RobertaMultiHeadCRF-C32-3](https://huggingface.co/IEETA/RobertaMultiHeadCRF-C32-3)

Example:

```bash
python inference.py IEETA/RobertaMultiHeadCRF-C32-0
```

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




## How to use the architecture on a new Dataset

In order to create a new Dataset, you will need to create a function that returns a corpus, for example the function `Spanish_Biomedical_NER_Corpus`. A full example can be seen below. 

### 1. `Corpus`
The `Corpus` class represents a collection of documents with annotations. Each document in the corpus must adhere to the following format:
```json
{
    "doc_id": "unique_document_identifier",
    "text": "document_text",
    "annotations": [
        {"label": "LABEL", "start_span": start_position, "end_span": end_position},
        ...
    ]
}
```

#### Methods:
- `__init__(data: list)`: Initializes the corpus. The `data` must be a list of documents with the structure mentioned above.
- `__len__()`: Returns the number of documents in the corpus.
- `get_entities()`: Returns a set of all unique entity labels in the corpus.
- `split(split: float)`: Splits the corpus into two sets (training and testing) based on the provided split ratio.

### 2. `CorpusPreProcessor`
The `CorpusPreProcessor` class is responsible for applying transformations and filters to the corpus.

#### Methods:
- `__init__(corpus: Corpus)`: Initializes the preprocessor with a `Corpus` object.
- `filter_labels(labels: list)`: Filters the annotations to keep only those with labels that match the provided list.
- `merge_annotations()`: Merges overlapping or adjacent annotations based on their span. If a collision is found, the annotations are merged.
- `split_data(test_split_percentage: float)`: Splits the corpus into training and test sets based on the provided split percentage.

### 4. `CorpusTokenizer`
The `CorpusTokenizer` class tokenizes the corpus using a given tokenizer (e.g., from the Hugging Face library) and prepares it for training by splitting it into context windows.

#### Parameters:
- `corpus: CorpusPreProcessor`: The preprocessed corpus to tokenize.
- `tokenizer`: A tokenizer that tokenizes the text (e.g., Hugging Face's tokenizers).
- `context_size`: The size of the context to be added before and after the main token sequence (optional).

#### Methods:
- `__tokenize()`: Tokenizes the entire corpus.
- `__split()`: Splits the tokenized corpus into windows, handling context and special tokens (if any).

### 5. `CorpusDataset`
This class wraps the tokenized corpus into a dataset compatible with PyTorch's `DataLoader`.

#### Parameters:
- `tokenized_corpus: CorpusTokenizer`: The tokenized corpus.
- `transforms`: Optional transformations to apply to each sample.
- `augmentations`: Optional augmentations to apply to each sample.



## Usage Example

```python
# First, create a generic Corpus
spanishCorpus = Spanish_Biomedical_NER_Corpus(
    "../dataset/merged_data_subtask1_train.tsv", 
    "../dataset/documents"
)

# Create a Corpus PreProcessor, which handles certain preprocessing tasks: 
# merging annotations, filtering labels, and splitting the data.
spanishCorpusProcessor = CorpusPreProcessor(spanishCorpus)
spanishCorpusProcessor.merge_annoatation()
spanishCorpusProcessor.filter_labels(classes)

# Split the corpus into training and testing sets with a 33% split.
train_corpus, test_corpus = spanishCorpusProcessor.split_data(0.33)

# Create a CorpusTokenizer, using the CorpusPreProcessor. 
# This internally tokenizes the dataset and splits the documents.
tokenized_train_corpus = CorpusTokenizer(train_corpus, tokenizer, CONTEXT_SIZE)

# Finally, create the dataset by applying transformations. 
# The order of transformations is important (BioTagging should be applied first).
train_ds = CorpusDataset(
    tokenized_corpus=tokenized_train_corpus, 
    transforms=transforms, 
    augmentation=train_augmentation
)

# Repeat the process for the test set
tokenized_test_corpus = CorpusTokenizer(test_corpus, tokenizer, CONTEXT_SIZE)
test_ds = CorpusDataset(
    tokenized_corpus=tokenized_test_corpus
)
```

This example shows the workflow of using the `Corpus`, `CorpusPreProcessor`, `CorpusTokenizer`, and `CorpusDataset` classes to create a dataset for a Named Entity Recognition (NER) task. It includes:

1. Loading a corpus from a dataset.
2. Preprocessing the corpus to merge annotations, filter specific labels, and split the data.
3. Tokenizing the processed corpus and splitting documents.
4. Creating a dataset with specified transformations and augmentations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Authors:*
- Richard A A Jonker ([ORCID: 0000-0002-3806-6940](https://orcid.org/0000-0002-3806-6940))
- Tiago Almeida ([ORCID: 0000-0002-4258-3350](https://orcid.org/0000-0002-4258-3350))
- Rui Antunes ([ORCID: 0000-0003-3533-8872](https://orcid.org/0000-0003-3533-8872))
- João R Almeida ([ORCID: 0000-0003-0729-2264](https://orcid.org/0000-0003-0729-2264))
- Sérgio Matos ([ORCID: 0000-0003-1941-3983](https://orcid.org/0000-0003-1941-3983))
