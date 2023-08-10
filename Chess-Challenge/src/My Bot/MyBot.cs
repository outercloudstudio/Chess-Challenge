using System;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  System.Text.StringBuilder _log = new System.Text.StringBuilder("Search:\n"); // #DEBUG

  record struct TranspositionEntry(ulong Hash, int Depth, int LowerBound, int UpperBound);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  Board _board;
  Move _bestMove;

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 10000 };

  int Interest(Move move)
  {
    if (move.IsCapture) return pieceValues[(int)move.CapturePieceType] - pieceValues[(int)move.MovePieceType] / 100;

    return 0;
  }

  int Evaluate()
  {
    int materialEvaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      materialEvaluation += _board.GetPieceList((PieceType)typeIndex, _board.IsWhiteToMove).Count * pieceValues[typeIndex];
      materialEvaluation -= _board.GetPieceList((PieceType)typeIndex, !_board.IsWhiteToMove).Count * pieceValues[typeIndex];
    }

    return materialEvaluation;
  }

  record struct OrderedMove(Move Move, int Interest);

  int AlphaBetaWM(int lowerBound, int upperBound, int ply, int depth, bool qSearch)
  {
    ulong hash = _board.ZobristKey;
    int key = (int)(hash % 100000);
    TranspositionEntry transpositionEntry = _transpositionTable[key];

    if (transpositionEntry.Depth > 0 && transpositionEntry.Depth >= depth && transpositionEntry.Hash == hash)
    {
      if (transpositionEntry.LowerBound >= upperBound) return transpositionEntry.LowerBound;
      if (transpositionEntry.UpperBound <= lowerBound) return transpositionEntry.UpperBound;

      lowerBound = Math.Max(lowerBound, transpositionEntry.LowerBound);
      upperBound = Math.Min(upperBound, transpositionEntry.UpperBound);
    }

    int max = -99999999;

    if (depth <= 0 && !qSearch) max = Evaluate();
    else
    {
      Move[] moves = _board.GetLegalMoves();

      if (moves.Length == 0) return _board.IsInCheck() ? -99999999 : 0;

      var orderedMoves = moves.Select(move => new OrderedMove(move, Interest(move))).OrderByDescending(orderedMove => orderedMove.Interest);

      if (ply == 0) _bestMove = orderedMoves.First().Move;

      foreach (OrderedMove orderedMove in orderedMoves)
      {
        Move move = orderedMove.Move;

        _board.MakeMove(move);

        _log.AppendLine(new string('\t', ply * 2 + 1) + "- " + move + ":"); // #DEBUG
        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Lower Bound: {lowerBound}"); // #DEBUG
        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Upper Bound: {upperBound}"); // #DEBUG
        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Q: {qSearch}"); // #DEBUG
        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Depth: {depth}"); // #DEBUG
        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Other Moves: {moves.Length}"); // #DEBUG
        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Children:"); // #DEBUG

        int score = -AlphaBetaWM(-upperBound, -lowerBound, ply + 1, depth - 1, depth == 1 && move.IsCapture);

        _log.AppendLine(new string('\t', ply * 2 + 2) + $"Score: {score}"); // #DEBUG

        _board.UndoMove(move);

        if (score >= upperBound)
        {
          if (ply == 0) _bestMove = move;

          max = score;

          upperBound = score;

          break;
        }

        if (score > max)
        {
          max = score;

          if (ply == 0) _bestMove = move;

          if (score > lowerBound) lowerBound = score;
        }
      }
    }

    if (!qSearch && depth >= transpositionEntry.Depth) _transpositionTable[key] = new TranspositionEntry(hash, depth, lowerBound, upperBound);

    return max;
  }

  int MTDF(int initialGuess, int depth)
  {
    int max = initialGuess;

    int upperBound = 99999999;
    int lowerBound = -99999999;

    while (lowerBound < upperBound)
    {
      int beta = max == lowerBound ? max + 1 : max;

      max = AlphaBetaWM(beta - 1, beta, 0, depth, false);

      if (max < beta) upperBound = max;
      else lowerBound = max;
    }

    return max;
  }

  int bestMoveGuess = 0;

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    int depth = 1;

    while (depth == 1 || timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60)
    {
      _log = new System.Text.StringBuilder("Search:\n"); // #DEBUG

      bestMoveGuess = MTDF(bestMoveGuess, depth);

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms");

      depth++;

      // System.IO.File.WriteAllText(@"D:\Chess-Challenge\Chess-Challenge\log.yml", _log.ToString()); // #DEBUG
    }

    return _bestMove;
  }
}