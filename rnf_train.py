import rnf_mcts
from common import common
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from common import game

from tqdm import tqdm
import numpy as np

def fst(x):
    return x[0]

class ReinfLearn():

    def __init__(self, model):
        self.model = model

    def playGame(self):

        # the next three arrays collect the
        # positions, associated move probabilities
        # from the MCT search and the final outcome
        # for the game that we play
        positionsData = []
        moveProbsData = []
        valuesData = []

        # set up a game with the starting position
        g = game.Board()
        g.setStartingPosition()

        # we play until we hit a final state
        while((not fst(g.isTerminal()))):
            # encode the current position to the
            # network input format
            positionsData.append(g.toNetworkInput())
            # setup the MCT search
            rootEdge = rnf_mcts.Edge(None, None)
            rootEdge.N = 1
            rootNode = rnf_mcts.Node(g, rootEdge)
            mctsSearcher = rnf_mcts.MCTS(self.model)
            moveProbs = mctsSearcher.search(rootNode)
            # MCT search return move probabilities for
            # all legal moves. To get an output vector
            # we need to consider all (incl. illegal) moves
            # but mask illegal moves to a probability of zero
            outputVec = [ 0.0 for x in range(0, 28)]
            for (move, prob, _, _) in moveProbs:
                move_idx = g.getNetworkOutputIndex(move)
                outputVec[move_idx] = prob
            # in order to explore enough positions
            # we interpret the result of the MCT search
            # as a multinomial distribution and randomly
            # select (w.r.t. the probabilites) a move
            rand_idx = np.random.multinomial(1, outputVec)
            idx = np.where(rand_idx==1)[0][0]
            nextMove = None
            # now we iterate through all legal moves
            # in order to find the one corresponding
            # to the randomly selected index
            for move, _, _, _ in moveProbs:
                move_idx = g.getNetworkOutputIndex(move)
                if(move_idx == idx):
                    nextMove = move
            if(g.turn == game.Board.WHITE):
                valuesData.append(1)
            else:
                valuesData.append(-1)
            moveProbsData.append(outputVec)
            g.applyMove(nextMove)
        else:
            # we have reached a final state
            _, winner = g.isTerminal()
            for i in range(0, len(moveProbsData)):
                if(winner == game.Board.BLACK):
                    valuesData[i] = valuesData[i] * -1.0
                if(winner == game.Board.WHITE):
                    valuesData[i] = valuesData[i] * 1.0
        return (positionsData, moveProbsData, valuesData)


# now let's try to train the ranomly initliazed network

model = common.SupNetwork()
model.load_state_dict(torch.load("common/random_model.pth"))

mctsSearcher = rnf_mcts.MCTS(model)
learner = ReinfLearn(model)
# we train the network in 11 iterations
for i in (range(0,11)):
    print("Training Iteration: "+str(i))
    allPos = []
    allMovProbs = []
    allValues = []
    # in each iteration we play ten games
    for j in tqdm(range(0,10)):
        pos, movProbs, values = learner.playGame()
        allPos += pos
        allMovProbs += movProbs
        allValues += values
    npPos = np.array(allPos)
    npProbs = np.array(allMovProbs)
    npVals = np.array(allValues)
    # we now have collected positions from ten training games
    # (considering a typical game length of 4 to 6 half moves,
    # that's approx. 40 up to 60 positions) and use those
    # to train the network
    
    dataset = common.CustomDataset(npPos, npProbs, npVals)
    training_dataloader = DataLoader(dataset=dataset, 
                                  batch_size=16, 
                                  num_workers=1, 
                                  shuffle=True) 


    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    
    loss_fn_value = nn.MSELoss()
    loss_fn_policy = nn.CrossEntropyLoss()
    common.trainModel(model, training_dataloader, 256, optimizer, loss_fn_policy,
                              loss_fn_value, "cpu")
            
    
    # model.fit(npPos,[npProbs, npVals], epochs=256, batch_size=16)
    if(i%10 == 0):
        torch.save(model.state_dict(), 
	 "model_it" + str(i) + ".pth")