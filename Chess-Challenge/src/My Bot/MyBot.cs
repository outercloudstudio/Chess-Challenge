using ChessChallenge.API;
using System;

public class MyBot : IChessBot
{
  static int[] pieceValues = { 0, 1, 3, 3, 5, 9, 1000 };
  int positionsEvaluated;
  bool botIsWhite;

  struct MoveChoice
  {
    public Move Move;
    public int Score;
  }

  public Move Think(Board board, Timer timer)
  {
    botIsWhite = board.IsWhiteToMove;
    positionsEvaluated = 0;

    Move[] moves = board.GetLegalMoves();
    Move bestMove = moves[0];
    int bestScore = int.MinValue;

    int depthToSearch;
    if (moves.Length < 5) depthToSearch = 3;
    else depthToSearch = 2;

    for (int i = 0; i < moves.Length; i++)
    {
      int score = Minimax(board, moves[i], depthToSearch);

      if (score > bestScore)
      {
        bestScore = score;
        bestMove = moves[i];
      }
    }

    return bestMove;
  }

  public int Minimax(Board board, Move move, int depth)
  {
    positionsEvaluated++;

    board.MakeMove(move);

    // We should always evaluate from the perspective of the bot's color
    // The problem was if you cause the bot to evaluate for the other color, then it will minimize for that colo, which means it will pick the worst move instead of the best move for the other color
    int heuristic = EvaluateBoard(board, botIsWhite);

    if (depth == 0)
    {
      board.UndoMove(move);

      return heuristic;
    }

    Move[] legalResponses = board.GetLegalMoves();
    int bestLegalResponseValue;

    // We also no longer need to pass if we are maximizing, since we can always tell by the color of our bot and the color of the board
    // if it is our turn to move, we maximize score
    if (botIsWhite == board.IsWhiteToMove)
    {
      bestLegalResponseValue = int.MinValue;
      for (int i = 0; i < legalResponses.Length; i++)
      {
        int value = Minimax(board, legalResponses[i], depth - 1);
        bestLegalResponseValue = Math.Max(value, bestLegalResponseValue);
      }
    }
    // if it is not our turn to move, we minimize OUR score, which is the same as maximizing the opponent's score
    else
    {
      bestLegalResponseValue = int.MaxValue;
      for (int i = 0; i < legalResponses.Length; i++)
      {
        int value = Minimax(board, legalResponses[i], depth - 1);
        bestLegalResponseValue = Math.Min(value, bestLegalResponseValue);
      }
    }

    board.UndoMove(move);

    return bestLegalResponseValue;
  }

  public int EvaluateBoard(Board board, bool asWhite)
  {
    int whiteScore = 0;
    int blackScore = 0;

    PieceList[] pieces = board.GetAllPieceLists();

    for (int i = 0; i < pieces.Length; i++)
    {
      for (int j = 0; j < pieces[i].Count; j++)
      {
        Piece piece = pieces[i][j];
        int pieceScore = pieceValues[(int)piece.PieceType];

        if (piece.IsWhite) whiteScore += pieceScore;
        else blackScore += pieceScore;
      }
    }

    return (asWhite ? 1 : -1) * (whiteScore - blackScore);
  }
}