import chess
from stockfish import Stockfish

stockfish = Stockfish(path="D:/Chess-Challenge/Chess-Challenge/src/Training/eval.exe", depth=10, parameters={"Threads": 2, "Minimum Thinking Time": 0, "Hash": 4096},)

file = open('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Fens\\Positions Medium.txt', "r")
entries = file.read().split("\n")

file.close()

fens = []
moves = []
evaluations = []

for fen in entries:
  board = chess.Board(fen)
  
  for move in board.legal_moves:
    board.push(move)

    fens.append(board.fen())
    moves.append(move.uci())
    
    stockfish.set_fen_position(fen)
    evaluation = stockfish.get_evaluation()

    if evaluation["type"] == "cp":
      evaluation = evaluation["value"] / 100
    elif evaluation["type"] == "mate":
      evaluation = evaluation["value"] * 10

    evaluations.append(evaluation)

    board.pop()

  print('Writing '+ str(len(fens)) + ' positions to file...')

  file = open('D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Move Evaluations Small.txt', "w")

  for i in range(len(fens)):
    file.write(fens[i] + " | ")
    file.write(moves[i] + " | ")
    file.write(str(evaluations[i]) + "\n")

  file.close()