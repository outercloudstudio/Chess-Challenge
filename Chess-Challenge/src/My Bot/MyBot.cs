﻿using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  Board _board;
  Timer _timer;
  Move _bestMove;
  int _searchedMoves;

  record struct TranspositionEntry(ulong Hash, int Depth, int Score);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  record struct MoveChoice(Move Move, int Interest);

  int Search(int depth, int ply, int alpha, int beta, bool qSearch, bool debug)
  {
    if (ply != 0 && _timer.MillisecondsElapsedThisTurn > _timer.MillisecondsRemaining / 60 && !qSearch) return 0;

    _searchedMoves++;

    ulong hash = _board.ZobristKey;
    int key = (int)(hash % 100000);
    TranspositionEntry entry = _transpositionTable[key];

    if (entry.Depth > 0 && entry.Depth >= depth && entry.Hash == hash && !qSearch) return entry.Score;

    if (depth <= 0 && !qSearch) return Evaluate();

    int moveCount = _board.GetLegalMoves(false).Length;
    MoveChoice[] moveChoices = _board.GetLegalMoves(false).Select(move => new MoveChoice(move, Interest(move))).OrderByDescending(moveChoice => moveChoice.Interest).ToArray();

    if (moveChoices.Length == 0) return Evaluate();

    int max = -999999995;

    foreach (MoveChoice moveChoice in moveChoices)
    {
      Move move = moveChoice.Move;

      _board.MakeMove(move);

      int score = -Search(depth - 1, ply + 1, -beta, -alpha, depth <= 1 && move.IsCapture, (ply == 0 && move.ToString() == "Move: 'd4f5'") || debug);

      _board.UndoMove(move);

      if (score >= beta)
      {
        if (ply == 0) _bestMove = move;

        return score;
      }

      if (score > max)
      {
        max = score;

        if (ply == 0) _bestMove = move;

        if (score > alpha) alpha = score;
      };
    }

    if (depth > entry.Depth) _transpositionTable[key] = new TranspositionEntry(hash, depth, max);

    return max;
  }

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 0 };

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
      materialEvaluation += _board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      materialEvaluation -= _board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    return materialEvaluation * ColorEvaluationFactor(_board.IsWhiteToMove);
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    int depth = 5;
    int bestMoveScore = 0;
    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60)
    {
      int score = Search(depth, 0, bestMoveScore - 100, bestMoveScore + 100, false, false);

      if (score <= bestMoveScore - 100 || score >= bestMoveScore + 100)
      {
        bestMoveScore = Search(depth, 0, -999999991, 999999992, false, false);
      }
      else
      {
        bestMoveScore = score;
      }

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn} ms");

      depth++;
    }

    // Console.WriteLine($"Searched {_searchedMoves} moves");

    return _bestMove;
  }
}