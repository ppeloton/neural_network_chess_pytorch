# neural_network_chess_pytorch
Pytorch models based on the book "Neural networks for Chess". Please go to the repo https://github.com/asdfjkl/neural_network_chess to check further details and where you can get the book (and you may also give a donation to the author). The models in the book are implemented in `tensorflow`, the aim of this repo is to provide an implementation of the models in chapter 5 in `pytorch`.

- The input files `positions.npy`, `moveProbs.npy` and `óutcomes.npy` are direct copies from the above mentioned repo (see details how to create them and their underlying logic there)
- the files `rnf_mcts.py`and `game.py` are also directly copied, as they are needed in the training process.
- All other files have corresponding files in the original repo, but have been modified to have `pytorch` models instead of `tensorflow` models

To execute the code for the supervised learning:

- run `sup_network.py` for training and then `sup_eval.py` for evaluation

For the MCTS approach:
- initialise a random model with `common/init_random_model.py`
- train the model with `rnf_train.py`
- evaluate the model with `rnf_eval.py`
