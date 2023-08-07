using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  Board _board;
  Timer _timer;
  Move _bestMove;
  int _searchedMoves;
  bool _cancelledSearchEarly;
  // System.Text.StringBuilder _log = new System.Text.StringBuilder("Search:\n"); // #DEBUG

  record struct TranspositionEntry(ulong Hash, int Depth, int Score, int Bound);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  record struct MoveChoice(Move Move, int Interest);

  int Search(int depth, int ply, int alpha, int beta, bool isLoud, bool initial)
  {
    if (!initial && _timer.MillisecondsElapsedThisTurn > _timer.MillisecondsRemaining / 60)
    {
      _cancelledSearchEarly = true;

      return 0;
    }

    bool qSearch = isLoud && depth <= 0;

    _searchedMoves++;

    ulong hash = _board.ZobristKey;
    int key = (int)(hash % 100000);
    TranspositionEntry entry = _transpositionTable[key];

    if (entry.Depth > 0 && entry.Depth >= depth && entry.Hash == hash && !qSearch)
    {
      if (entry.Bound == 0) return entry.Score;

      if (entry.Bound == 1) alpha = Math.Max(alpha, entry.Score);

      if (entry.Bound == -1) beta = Math.Min(beta, entry.Score);

      if (alpha >= beta) return entry.Score;
    }

    if (depth <= 0 && !(qSearch && ply < 12)) return Evaluate();

    MoveChoice[] moveChoices = _board.GetLegalMoves().Select(move => new MoveChoice(move, Interest(move))).OrderByDescending(moveChoice => moveChoice.Interest).ToArray();

    if (moveChoices.Length == 0) return Evaluate();

    if (ply == 0) _bestMove = moveChoices[0].Move;

    int max = -999999995;

    foreach (MoveChoice moveChoice in moveChoices)
    {
      Move move = moveChoice.Move;

      _board.MakeMove(move);

      // _log.AppendLine(new string('\t', ply * 2 + 1) + "- " + move + ":"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Alpha: {alpha}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Beta: {beta}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Loud: {isLoud}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Depth: {depth}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Other Moves: {moveChoices.Length}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Children:"); // #DEBUG

      int score;

      if (ply == 3)
      {
        score = -Search(depth - 3, ply + 1, -beta, -alpha, move.IsCapture, initial);

        if (score >= beta) score = -Search(depth - 1, ply + 1, -beta, -alpha, move.IsCapture, initial);
      }
      else
      {
        score = -Search(depth - 1, ply + 1, -beta, -alpha, move.IsCapture, initial);
      }

      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Score: {score}"); // #DEBUG

      _board.UndoMove(move);

      if (score >= beta)
      {
        if (ply == 0) _bestMove = move;

        if (depth > entry.Depth) _transpositionTable[key] = new TranspositionEntry(hash, depth, max, 1);

        return score;
      }

      if (score > max)
      {
        max = score;

        if (ply == 0) _bestMove = move;

        if (score > alpha) alpha = score;
      };
    }

    if (depth > entry.Depth && !_cancelledSearchEarly) _transpositionTable[key] = new TranspositionEntry(hash, depth, max, max <= alpha ? -1 : 0);

    return max;
  }

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 10000 };

  int ColorEvaluationFactor(bool white) => white ? 1 : -1;

  int Interest(Move move)
  {
    if (move == _bestMove) return 999;

    if (move.IsCapture) return pieceValues[(int)move.CapturePieceType] - pieceValues[(int)move.MovePieceType] / 100;

    return 0;
  }

  int Evaluate()
  {
    if (_board.IsInCheckmate()) return -999993;

    if (_board.IsDraw()) return 0;

    int materialEvaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      materialEvaluation += _board.GetPieceList((PieceType)typeIndex, _board.IsWhiteToMove).Count * pieceValues[typeIndex];
      materialEvaluation -= _board.GetPieceList((PieceType)typeIndex, !_board.IsWhiteToMove).Count * pieceValues[typeIndex];
    }

    int mobilityEvaluation = _board.GetLegalMoves().Length / 20;

    return materialEvaluation + mobilityEvaluation;
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    _cancelledSearchEarly = false;

    int depth = 4;
    int bestMoveScore = 0;
    Move lastSearchBestMove = Move.NullMove;
    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60 || depth == 4)
    {
      // _log = new System.Text.StringBuilder("Search:\n"); // #DEBUG

      int score = Search(depth, 0, bestMoveScore - 100, bestMoveScore + 100, false, depth == 4);

      if (score <= bestMoveScore - 100 || score >= bestMoveScore + 100)
      {
        bestMoveScore = Search(depth, 0, -999999991, 999999992, false, depth == 4);
      }
      else
      {
        bestMoveScore = score;
      }

      if (!_cancelledSearchEarly) lastSearchBestMove = _bestMove;

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn} ms, cancelled early {_cancelledSearchEarly}"); // #DEBUG

      // System.IO.File.WriteAllText(@"D:\Chess-Challenge\Chess-Challenge\log.yml", _log.ToString()); // #DEBUG

      depth++;
    }

    Console.WriteLine($"Searched {_searchedMoves} moves");

    return lastSearchBestMove;
  }
}