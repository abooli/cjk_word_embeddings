# cjk_word_embeddings

In order to decompose the vector files into a word list and a npz file, run the command:
```
python -i embedding.py original-vector-file numpy-file.npy text-file.txt
```

For example,
```
python -i embedding.py cc.zh.300.vec zh.300.npy zh.300.txt
```

The `embedding.load_word_list()` and `embedding.load_vectors_array()` functions loads the above files into lists and numpy arrays (see jupyter notebook)