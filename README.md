# Word Embeddings with FastText and GloVe

This repository demonstrates how to use **FastText** and **GloVe** for generating word embeddings, training models, and performing text classification.

---

## 📌 Requirements
Make sure you have the following installed:
- Python 3.7+
- pip
- Required libraries:
  ```bash
  pip install fasttext gensim
  ```
---

## 🚀 FastText
1. Using Pretrained Models
     ```python
      import fasttext
      import urllib.request
      
      # Download pretrained English embeddings
      link_en = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
      urllib.request.urlretrieve(link_en, "en.gz")
      
      # Unzip
      !gunzip en.gz
      
      # Load model
      model_en = fasttext.load_model('en')
      
      # Example: Find nearest neighbors
      model_en.get_nearest_neighbors('Kejriwal')

     ```
2. Training Model for Word Vectors
     ```python
       # Clone dataset
      !git clone https://github.com/AshishJangra27/datasets
      
      # Train on Sherlock Holmes script
      model = fasttext.train_unsupervised(input='datasets/Sherlock/script.txt', model='cbow')
      
      # Example usage
      model.get_nearest_neighbors('head')
      
      # Save model
      model.save_model('sherlock_cbow.bin')

     ```
3. Text Classification
     ```python
     # Train supervised model
      model = fasttext.train_supervised('/content/datasets/Cooking Labels/cooking.txt')
      
      # Predictions
      model.predict('Which baking dish is best to bake a banana bread?', k=5)
      model.predict('How to heat up already baked french bread in oven to get a crispy crust', k=5)

     ```
---

## 🌍 GloVe
Download Pretrained Embeddings

You can use different GloVe embeddings:

- Common Crawl (840B tokens, 300d) → [Download](https://nlp.stanford.edu/data/glove.840B.300d.zip)
- Twitter (27B tokens) → [Download](https://nlp.stanford.edu/data/glove.twitter.27B.zip)
- Wikipedia + Gigaword (6B tokens) → [Download](http://nlp.stanford.edu/data/glove.6B.zip)

```python
import zipfile
import urllib.request

# Download Wikipedia + Gigaword (6B tokens, 300d)
urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", "glove.zip")

# Extract
with zipfile.ZipFile('glove.zip', 'r') as zip_ref:
    zip_ref.extractall('glove')
```

Convert GloVe to Word2Vec format
```python
!pip install gensim
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove/glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.txt'

# Convert
glove2word2vec(glove_input_file, word2vec_output_file)

```

Load Model with Gensim
```python
from gensim.models import KeyedVectors

filename = 'glove.6B.300d.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

# Example: Find most similar words
model.most_similar('king')
```
---

## 📖 References
- https://fasttext.cc/docs/en/support.html
- https://nlp.stanford.edu/projects/glove/

---

## ✨ Author

Rishita Priyadarshini Saraf

mail: rishitasarafp@gmail.com
