#!/bin/sh

wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir $HOME/GloVe
unzip glove.6B.zip -d $HOME/GloVe
rm glove.6B.zip

echo "Pretrained GloVe vectors downloaded."
