using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class NegamaxBot : IChessBot
{
  Board _board;

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  int ColorEvaluationFactor(bool white) => white ? 1 : -1;

  int Evaluate()
  {
    if (_board.IsInCheckmate()) return -1000;

    if (_board.IsInsufficientMaterial() || _board.IsRepeatedPosition() || _board.FiftyMoveCounter >= 100) return 1;

    int materialEvaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      materialEvaluation += _board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      materialEvaluation -= _board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    return materialEvaluation * ColorEvaluationFactor(_board.IsWhiteToMove);
  }

  public int Negamax(int depth, bool logging = false)
  {
    var moves = _board.GetLegalMoves();

    if (depth == 0 || moves.Length == 0) return Evaluate();

    int max = -9999999;

    foreach (Move move in moves)
    {
      _board.MakeMove(move);

      int score = -Negamax(depth - 1, logging);

      _board.UndoMove(move);

      max = Math.Max(max, score);
    }

    return max;
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    return board.GetLegalMoves().MaxBy(move =>
    {
      board.MakeMove(move);

      int score = -Negamax(3);

      board.UndoMove(move);

      return score;
    });
  }
}