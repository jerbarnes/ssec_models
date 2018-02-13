## Multi-Label Reannotation of the SemEval 2016 Stance Detection Data with Fine-Grained Emotions
Code for the paper to-appear.

Runs a series of models on the dataset
Please cite the original paper when using the data.

### Requirements
Code is written in Python (3.5), requires Keras (1) [https://keras.io] and tabulate [https://pypi.python.org/pypi/tabulate].



### How to use code
run ./run.sh from the command line
This runs the experiments and produces the latex code for the table containing the neural models, which is saved as /figs/large_table.txt


### Hyperparameters
We use a random search to find the best parameters for hidden layers, number of epochs and dropout probability.


