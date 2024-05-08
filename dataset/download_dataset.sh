#!/bin/bash

# Download and extract symptemist
echo "Downloading symptemist..."
wget -O symptemist.zip https://zenodo.org/records/10635215/files/symptemist-complete_240208.zip?download=1
echo "Extracting symptemist..."
unzip symptemist.zip
mv symptemist-complete_240208 symptemist
rm symptemist.zip

# Download and extract distemist
echo "Downloading distemist..."
wget -O distemist.zip https://zenodo.org/records/7614764/files/distemist_zenodo.zip?download=1
echo "Extracting distemist..."
unzip distemist.zip
mv distemist_zenodo distemist
rm distemist.zip

# Download distemist gazetteer
echo "Downloading distemist gazetteer..."
wget -O distemist_gazetteer.tsv https://zenodo.org/records/6505583/files/dictionary_distemist.tsv?download=1
mkdir distemist/distemist_gazetteer/
mv distemist_gazetteer.tsv distemist/distemist_gazetteer/

# Download and extract medprocner
echo "Downloading medprocner..."
wget -O medprocner.zip https://zenodo.org/records/8224056/files/medprocner_gs_train+test+gazz+multilingual+crossmap_230808.zip?download=1
echo "Extracting medprocner..."
unzip medprocner.zip
mv medprocner_gs_train+test+gazz+multilingual+crossmap_230808 medprocner
rm medprocner.zip

# Download and extract pharmaconer
echo "Downloading pharmaconer..."
wget -O pharmaconer.zip https://zenodo.org/records/4270158/files/pharmaconer.zip?download=1
echo "Extracting pharmaconer..."
unzip pharmaconer.zip
rm pharmaconer.zip

echo "All datasets downloaded and extracted successfully."

# Prepare the documents directory

echo "Prepare the documents directory"
mkdir documents
cp symptemist/symptemist_train/subtask1-ner/txt/* documents
cp symptemist/symptemist_test/subtask1-ner/txt/* documents

python3 build_new_pharmaconer.py

python3 create_dataset.py

echo "Built the dataset."
