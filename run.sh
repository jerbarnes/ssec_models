#!/bin/bash

python3 multi_class.py -emb embeddings/twitter_embeddings.txt

python one_vs_all.py -emb embeddings/twitter_embeddings.txt