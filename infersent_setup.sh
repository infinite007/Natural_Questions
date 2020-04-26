cd ./data/
mkdir Glove
curl -Lo Glove/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip Glove/glove.840B.300d.zip -d Glove/
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
python -c "import nltk;nltk.download('punkt')"