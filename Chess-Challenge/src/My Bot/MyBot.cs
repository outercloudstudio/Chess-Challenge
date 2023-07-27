using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class ARCNET3 : IChessBot
{
  int _statesSearched;
  float _maxDepthSearched;

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  float Evaluate(Board board)
  {
    if (board.IsInCheckmate()) return -1000 * (board.IsWhiteToMove ? 1 : -1);

    if (board.IsInsufficientMaterial() || board.IsRepeatedPosition() || board.FiftyMoveCounter >= 100) return -0.5f * (board.IsWhiteToMove ? 1 : -1);

    float evaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      evaluation += board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      evaluation -= board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    if (board.IsInCheck()) evaluation += -0.5f * (board.IsWhiteToMove ? 1 : -1);

    return evaluation;
  }

  float EvaluateOrder(Board board, Move move)
  {
    float evaluation = 0;

    if (move.IsCapture) evaluation += 1f;
    if (move.IsPromotion) evaluation += 2f;

    board.MakeMove(move);

    if (board.IsInCheck()) evaluation += 0.5f;

    board.UndoMove(move);

    return evaluation;
  }

  struct MoveEvaluationPair
  {
    public Move Move;
    public float Value;
  }

  float Search(Board board, float previousValue = 0, int depth = 0)
  {
    _statesSearched++;
    _maxDepthSearched = Math.Max(_maxDepthSearched, depth);

    var moves = board.GetLegalMoves().Select(move => new MoveEvaluationPair
    {
      Move = move,
      Value = EvaluateOrder(board, move)
    }).OrderByDescending(evaluation => evaluation.Value);

    if (moves.Count() == 0) return Evaluate(board);

    if (depth > 1)
    {
      var evaluations = moves.Select(evaluation =>
      {
        board.MakeMove(evaluation.Move);

        float value = Evaluate(board);

        board.UndoMove(evaluation.Move);

        return new MoveEvaluationPair
        {
          Move = evaluation.Move,
          Value = value
        };
      });

      if (board.IsWhiteToMove)
      {
        return evaluations.Max(evaluation => evaluation.Value);
      }
      else
      {
        return evaluations.Min(evaluation => evaluation.Value);
      }
    }

    bool set = false;
    float best = 0;

    foreach (MoveEvaluationPair evaluation in moves.Reverse())
    {
      board.MakeMove(evaluation.Move);

      float result = Search(board, previousValue, depth + 1);

      board.UndoMove(evaluation.Move);

      Console.WriteLine(new string('\t', depth) + " Evaluated " + evaluation.Move + " for " + result);

      if ((board.IsWhiteToMove && result > previousValue + 1f) || (!board.IsWhiteToMove && result < previousValue - 1f)) return result;

      if (!set || (board.IsWhiteToMove && result >= best) || (!board.IsWhiteToMove && result <= best)) best = result;

      set = true;
    }

    Console.WriteLine(new string('\t', depth) + " Best is " + best);

    return best;
  }

  public Move Think(Board board, Timer timer)
  {
    _statesSearched = 0;
    _maxDepthSearched = 0;

    float evaluation = 0;
    Move bestMove = board.GetLegalMoves().MaxBy(move =>
    {
      board.MakeMove(move);

      float result = Search(board);

      board.UndoMove(move);

      evaluation = Math.Max(evaluation, result);

      return result * (board.IsWhiteToMove ? 1 : -1);
    });

    Console.WriteLine("Arcnet 3 Found " + bestMove + " Evaluation: " + evaluation + " Searched " + _statesSearched + " states. Max depth: " + _maxDepthSearched);

    return bestMove;
  }
}