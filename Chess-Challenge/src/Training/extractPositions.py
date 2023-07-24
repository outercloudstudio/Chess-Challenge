import chess.pgn

pgn = open("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\positions.pgn")

positions = []

for i in range(1000):
  game = chess.pgn.read_game(pgn)
  moves = game.mainline_moves()
  board = game.board()

  movesPlayed = 0
  for move in moves:
    board.push(move)

    if movesPlayed % 2 == 1: positions.append(board.fen())

    movesPlayed += 1

pgn.close()

file = open("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Fens\\Positions Medium.txt", 'w')
file.write('\n'.join(positions))