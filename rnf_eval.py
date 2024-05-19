from common import common
import torch

from common import game
import random
import numpy as np


model = common.SupNetwork()
model.load_state_dict(torch.load("model_it10.pth"))


def fst(a):
    return a[0]

# white == random player
# black == network
def net_vs_rand(board):
    record = []
    while(not fst(board.isTerminal())):
        if(board.turn == game.Board.WHITE):
            moves = board.generateMoves()
            m = moves[random.randint(0, len(moves)-1)]
            board.applyMove(m)
            record.append(m)
            continue
        else:
            q = model.forward(torch.tensor([board.toNetworkInput()]).type(torch.float))
            masked_output = [ 0 for x in range(0,28)]
            for m in board.generateMoves():
                m_idx = board.getNetworkOutputIndex(m)
                masked_output[m_idx] = torch.exp(q[0][0][m_idx]).item()
            best_idx = np.argmax(masked_output)
            sel_move = None
            for m in board.generateMoves():
                m_idx = board.getNetworkOutputIndex(m)
                if(best_idx == m_idx):
                    sel_move = m
            board.applyMove(sel_move)
            record.append(sel_move)
            continue
    terminal, winner = board.isTerminal()
    return winner

# white random player
# black random player
def rand_vs_rand(board):
    while(not fst(board.isTerminal())):
        moves = board.generateMoves()
        m = moves[random.randint(0, len(moves)-1)]
        board.applyMove(m)
        continue
    terminal, winner = board.isTerminal()
    return winner


whiteWins = 0
blackWins = 0

for i in range(0,100):
    board = game.Board()
    board.setStartingPosition()
    moves = board.generateMoves()
    m = moves[random.randint(0, len(moves)-1)]
    board.applyMove(m)
    winner = net_vs_rand(board)
    if(winner == game.Board.WHITE):
        whiteWins += 1
    if(winner == game.Board.BLACK):
        blackWins += 1

all = whiteWins + blackWins
print("Rand Network vs Reinforcement: "+str(whiteWins/all) + "/"+str(blackWins/all))


whiteWins = 0
blackWins = 0

for i in range(0,100):
    board = game.Board()
    board.setStartingPosition()
    moves = board.generateMoves()
    m = moves[random.randint(0, len(moves)-1)]
    board.applyMove(m)
    winner = rand_vs_rand(board)
    if(winner == game.Board.WHITE):
        whiteWins += 1
    if(winner == game.Board.BLACK):
        blackWins += 1

all = whiteWins + blackWins
print("Rand vs Rand Network: "+str(whiteWins/all) + "/"+str(blackWins/all))