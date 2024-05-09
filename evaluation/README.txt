It is possible to run the evaluation.py script against the gold standard
merged data. Please follow the next steps:


1)  First, make sure you've run the download_dataset.sh bash script in
    the ../dataset/ directory to obtain the four original datasets.


2)  Second, create the merged data in the TSV format expected by the
    evaluation.py script by running:

    $ python3 make_merged_data_ready_for_evaluation.py

    This will create the following files (which will be in the TSV
    format expected by the evaluation.py script):
        merged_data_subtask1_train_ready_for_evaluation.tsv
        merged_data_subtask2_train_ready_for_evaluation.tsv
        merged_data_subtask1_test_ready_for_evaluation.tsv
        merged_data_subtask2_test_ready_for_evaluation.tsv

    Notice that:
        - subtask1 corresponds to NER data.
        - subtask2 corresponds to NEL data.


3)  Finally, we can perform NER and NEL evaluation, between
        - the four original gold standard datasets and
        - the merged gold standard dataset,
    by running one of the following commands:

    $ python3 evaluation.py train merged_data_subtask1_train_ready_for_evaluation.tsv
    $ python3 evaluation.py train merged_data_subtask2_train_ready_for_evaluation.tsv
    $ python3 evaluation.py test  merged_data_subtask1_test_ready_for_evaluation.tsv
    $ python3 evaluation.py test  merged_data_subtask2_test_ready_for_evaluation.tsv

    Note that the evalution result is expected to be a Micro-averaged
    F1-score of 1.0. This also serves as a sanity check that the merged
    data is equivalent to the four original datasets.


---


Abbreviations:
    NER: named entity recognition
    NEL: named entity linking
    TSV: tab-separated values
