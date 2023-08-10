using System;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  record struct TranspositionEntry(ulong Hash, int Depth, int LowerBound, int UpperBound, Move BestMove);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  int[,,] _historyTable;

  Board _board;
  Move _bestMove;
  bool _white;
  Timer _timer;
  bool _initialSearch;

  int nodesSearched = 0;

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 10000 };

  bool hasTime => _timer.MillisecondsElapsedThisTurn < _timer.MillisecondsRemaining / 60;

  int Interest(Move move, Move bestHashMove)
  {
    if (move == bestHashMove) return 99999999;

    if (move.IsCapture) return pieceValues[(int)move.CapturePieceType] - pieceValues[(int)move.MovePieceType] / 100;

    return _historyTable[_board.IsWhiteToMove ? 0 : 1, (int)move.MovePieceType - 1, move.TargetSquare.Index];
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
    nodesSearched++;

    Move bestMove = Move.NullMove;

    ulong hash = _board.ZobristKey;
    ulong key = hash % 100000L;
    TranspositionEntry transpositionEntry = _transpositionTable[key];

    if (transpositionEntry.Depth > 0 && transpositionEntry.Hash == hash)
    {
      if (transpositionEntry.Depth >= depth)
      {
        if (transpositionEntry.LowerBound >= upperBound) return transpositionEntry.LowerBound;
        if (transpositionEntry.UpperBound <= lowerBound) return transpositionEntry.UpperBound;

        lowerBound = Math.Max(lowerBound, transpositionEntry.LowerBound);
        upperBound = Math.Min(upperBound, transpositionEntry.UpperBound);

        if (ply == 0) _bestMove = transpositionEntry.BestMove;
      }

      bestMove = transpositionEntry.BestMove;
    }

    Move[] moves = _board.GetLegalMoves(qSearch);

    int max = -99999999 + ply;

    if (qSearch) max = Evaluate();

    if (depth <= 0 && !qSearch) return Evaluate();
    else
    {
      if (_board.IsRepeatedPosition() || _board.IsFiftyMoveDraw()) return _white == _board.IsWhiteToMove ? -5 : 5;

      var orderedMoves = moves.Select(move => new OrderedMove(move, Interest(move, bestMove))).OrderByDescending(orderedMove => orderedMove.Interest);

      foreach (OrderedMove orderedMove in orderedMoves)
      {
        if (!_initialSearch && !hasTime) break;

        Move move = orderedMove.Move;

        _board.MakeMove(move);

        int score = -AlphaBetaWM(-upperBound, -lowerBound, ply + 1, depth - 1, depth == 1 && move.IsCapture);

        _board.UndoMove(move);

        if (score >= upperBound)
        {
          bestMove = move;

          max = score;

          upperBound = score;

          if (!move.IsCapture)
            _historyTable[_board.IsWhiteToMove ? 0 : 1, (int)move.MovePieceType - 1, move.TargetSquare.Index] += depth * depth;

          break;
        }

        if (score > max)
        {
          max = score;

          bestMove = move;

          if (score > lowerBound) lowerBound = score;
        }
      }
    }

    if (!qSearch && depth >= transpositionEntry.Depth) _transpositionTable[key] = new TranspositionEntry(hash, depth, lowerBound, upperBound, bestMove);

    if (ply == 0) _bestMove = bestMove;

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
    _white = board.IsWhiteToMove;
    _timer = timer;

    _historyTable = new int[2, 6, 64];

    int depth = 1;

    while (_initialSearch || hasTime)
    {
      _initialSearch = depth < 5;

      Move lastBestMove = _bestMove;

      bestMoveGuess = MTDF(bestMoveGuess, depth);

      if (!_initialSearch && !hasTime)
      {
        _bestMove = lastBestMove;

        break;
      }

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms"); // #DEBUG

      depth++;
    }

    depth--;

    Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms {nodesSearched / (timer.MillisecondsElapsedThisTurn / (float)1000)} nodes/sec"); // #DEBUG

    return _bestMove;
  }
}