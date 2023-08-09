using System;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  record struct TranspositionEntry(ulong Hash, int Depth, int lowerBound, int upperBound);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  Board _board;
  Move _bestMove;

  (TranspositionEntry, bool) RetrieveTransposition(int depth)
  {
    ulong hash = _board.ZobristKey;
    int key = (int)(hash % 100000);
    TranspositionEntry entry = _transpositionTable[key];

    if (entry.Depth > 0 && entry.Depth >= depth && entry.Hash == hash) return (entry, true);

    return (entry, false);
  }

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 10000 };

  int Interest(Move move)
  {
    if (move == _bestMove) return 999;

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
    int mobilityEvaluation = _board.GetLegalMoves().Length / 20;
    return materialEvaluation + mobilityEvaluation;
  }

  record struct OrderedMove(Move Move, int Interest);

  int AlphaBetaWM(int lowerBound, int upperBound, int ply, int depth)
  {
    (TranspositionEntry transpositionEntry, bool found) = RetrieveTransposition(depth);

    if (found)
    {
      if (transpositionEntry.lowerBound >= upperBound) return transpositionEntry.lowerBound;
      if (transpositionEntry.upperBound <= lowerBound) return transpositionEntry.upperBound;

      lowerBound = Math.Max(lowerBound, transpositionEntry.lowerBound);
      upperBound = Math.Min(upperBound, transpositionEntry.upperBound);
    }

    if (_board.IsInCheckmate()) return -99999999;

    int max = 0;

    if (depth == 0) max = Evaluate();
    else
    {
      Move[] moves = _board.GetLegalMoves();

      if (moves.Length == 0) return Evaluate();

      var orderedMoves = moves.Select(move => new OrderedMove(move, Interest(move))).OrderByDescending(orderedMove => orderedMove.Interest);

      if (ply == 0) _bestMove = orderedMoves.First().Move;

      foreach (OrderedMove orderedMove in orderedMoves)
      {
        Move move = orderedMove.Move;

        _board.MakeMove(move);

        int score = -AlphaBetaWM(-upperBound, -lowerBound, ply + 1, depth - 1);

        _board.UndoMove(move);

        if (score >= upperBound)
        {
          if (ply == 0) _bestMove = move;

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

      max = AlphaBetaWM(beta - 1, beta, 0, depth);

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
      bestMoveGuess = MTDF(bestMoveGuess, depth);

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms");

      depth++;
    }

    return _bestMove;
  }
}