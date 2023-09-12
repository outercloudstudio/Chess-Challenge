import chess.pgn

pgn = open("D:/Chess-Challenge/Training/Data/lichess_db_standard_rated_2014-09.pgn")

positions = []

for i in range(10000):
  print(str(i) + " / 10000")

  game = chess.pgn.read_game(pgn)
  moves = game.mainline_moves()
  board = game.board()

  for move in moves:
    board.push(move)
    positions.append(board.fen())

  if i % 100 == 0:
    file = open("D:/Chess-Challenge/Training/Data/Fens.txt", 'w')
    file.write('\n'.join(positions))
    file.close()

pgn.close()