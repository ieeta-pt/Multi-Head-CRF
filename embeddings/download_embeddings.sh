echo "Downloading embeddings..."
wget -O embeddings.zip https://zenodo.org/records/11174163/files/embeddings.zip?download=1
echo "Extracting embeddings..."
unzip symptemist.zip
mv "embeddings copy" .
rm embeddings.zip
rm -r "embeddings copy"
