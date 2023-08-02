﻿using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  static Board _board;
  static int _maxDepth = 0;

  record struct TranspositionEntry(int Score, int Depth);
  static TranspositionEntry[] _transpositionTable = new TranspositionEntry[10000];

  class State
  {
    public Move Move;
    public int Score;
    public State[] ChildStates = null;

    public void Expand(int targetDepth, bool root, int alpha = -99999, int beta = 99999, int depth = 0)
    {
      _maxDepth = Math.Max(_maxDepth, depth);

      _board.MakeMove(Move);

      // TranspositionEntry _transpositionEntry = _transpositionTable[_board.ZobristKey % 10000];

      // if (_transpositionEntry.Depth >= targetDepth - depth)
      // {
      //   Console.WriteLine(String.Format("Transposition Hit! entry depth: {0} target depth: {1} actual depth: {2}", _transpositionEntry.Depth, targetDepth, depth));

      //   Score = _transpositionEntry.Score;

      //   _board.UndoMove(Move);

      //   return;
      // }

      if (ChildStates == null)
      {
        ChildStates = _board.GetLegalMoves().Select(move => new State(move)).OrderByDescending(state => -state.Score).ToArray();

        if (ChildStates.Length != 0) Score = -ChildStates[0].Score;

        // foreach (State state in ChildStates) Console.WriteLine(new String('\t', _currentDepth) + String.Format("{0} Score: {1} Alpha: {2} Beta: {3}", state.Move, state.Score, alpha, beta));
      }
      else
      {
        int max = -99999;

        foreach (State state in ChildStates)
        {
          // Console.WriteLine(new String('\t', _currentDepth) + String.Format("Looking at {0} Alpha: {1} Beta: {2}", state.Move, alpha, beta));

          state.Expand(targetDepth, false, -beta, -alpha, depth + 1);

          int score = -state.Score;

          // Console.WriteLine(new String('\t', _currentDepth) + String.Format("Score: {0} Beta Check: {1} Alpha Check: {2}", score, score >= beta, score > alpha));

          if (score >= beta)
          {
            Score = beta;

            break;
          }

          if (score > max)
          {
            max = score;

            if (score > alpha) alpha = score;
          }
        }

        Score = max;

        ChildStates = ChildStates.OrderByDescending(state => -state.Score).ToArray();
      }

      // if (targetDepth - depth > _transpositionEntry.Depth && depth > 0)
      // {
      // _transpositionTable[_board.ZobristKey % 10000] = new TranspositionEntry(Score, depth);
      // }

      _board.UndoMove(Move);
    }

    public State(Move move)
    {
      Move = move;

      if (move.IsNull) return;

      _board.MakeMove(move);

      Score = Evaluate();

      _board.UndoMove(move);
    }
  }

  static int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  static int ColorEvaluationFactor(bool white) => white ? 1 : -1;

  static int Evaluate()
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

  Dictionary<string, State> _reuseableStates = new Dictionary<string, State>();

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _maxDepth = 0;

    string boardFen = board.GetFenString();

    State tree;

    if (_reuseableStates.ContainsKey(boardFen)) tree = _reuseableStates[boardFen];
    else tree = new State(Move.NullMove);

    for (int targetDepth = 1; timer.MillisecondsElapsedThisTurn < 12 || tree.ChildStates == null; targetDepth++)
    {
      Console.WriteLine("Expanding...");

      tree.Expand(targetDepth, true);
    }


    tree = tree.ChildStates.MaxBy(state => -state.Score);

    _reuseableStates = new Dictionary<string, State>();

    if (tree.ChildStates != null)
    {
      board.MakeMove(tree.Move);

      foreach (State state in tree.ChildStates)
      {
        board.MakeMove(state.Move);

        string fen = board.GetFenString();

        board.UndoMove(state.Move);

        _reuseableStates[fen] = state;

        state.Move = Move.NullMove;

        break;
      }

      board.UndoMove(tree.Move);
    }

    Console.WriteLine(String.Format("Searched to depth of {0}", _maxDepth));

    return tree.Move;
  }
}