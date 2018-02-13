#!/bin/bash


for threshold in 0.0 0.33 0.5 0.66 0.99; do
	python3 multi_class.py -emb embeddings/twitter_embeddings.txt -threshold $threshold -output results/results_$threshold
done

python3 Utils/create_large_table.py -results_dir results -outfile fig/large_table.txt