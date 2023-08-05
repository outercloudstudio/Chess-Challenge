using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class FastSearch : IChessBot
{
  /*
  TODO:
  PSTS
  Move Ordering
  Q Search
  */

  Board _board;
  Timer _timer;

  record class TranspositionEntry(ulong Hash, int Depth, int Score);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[100000];

  class State
  {
    public Move Move;
    public int Score;
    public int Interest;
    public State[] ChildStates = null;
    public int Key;
    public ulong Hash;
    public int Depth;

    FastSearch Me;

    public void Expand(int targetDepth, int depth = 0, int alpha = -9999999, int beta = 9999999, bool qSearch = false)
    {
      if (depth > targetDepth && !qSearch) return;

      if (!qSearch && ChildStates != null && Me._timer.MillisecondsElapsedThisTurn > Me._timer.MillisecondsRemaining / 30) return;

      TranspositionEntry entry = Me._transpositionTable[Key];

      if (entry != null && entry.Hash == Hash && entry.Depth >= Depth && entry.Depth >= targetDepth && !qSearch)
      {
        Score = entry.Score;

        CalculateInterest();

        return;
      }

      Me._board.MakeMove(Move);

      if (ChildStates == null)
      {
        ChildStates = Me._board.GetLegalMoves().Select(move => new State(move, Me)).ToArray();

        if (ChildStates.Length != 0)
        {
          Score = ChildStates.Max(State => -State.Score);

          CalculateInterest();
        }

        Depth = 1;

        if (depth < targetDepth)
        {
          Me._board.UndoMove(Move);

          Expand(targetDepth, depth, alpha, beta);

          return;
        }
        else if ((Move.IsCapture || Me._board.IsInCheck()) && targetDepth > 4)
        {
          Me._board.UndoMove(Move);

          Expand(targetDepth, depth, alpha, beta, true);

          return;
        }
      }
      else
      {
        ChildStates = ChildStates.OrderByDescending(state => -state.Score).ToArray();

        int max = -9999999;

        foreach (State state in ChildStates)
        {
          state.Expand(targetDepth, depth + 1, -beta, -alpha, qSearch);
          if (!qSearch) Depth = Math.Max(state.Depth + 1, Depth);

          int score = -state.Score;

          if (score >= beta)
          {
            max = beta;

            break;
          }

          if (score > max)
          {
            max = score;

            if (score > alpha) alpha = score;
          }
        }

        Score = max;
        CalculateInterest();
      }

      if (entry == null || entry.Depth < Depth) Me._transpositionTable[Key] = new TranspositionEntry(Hash, Depth, Score);

      Me._board.UndoMove(Move);
    }

    public State(Move move, FastSearch me)
    {
      Me = me;

      Move = move;

      Me._board.MakeMove(move);

      Hash = Me._board.ZobristKey;
      Key = (int)(Hash % (ulong)Me._transpositionTable.Length);

      Score = Me.Evaluate();

      CalculateInterest();

      Me._board.UndoMove(move);
    }


    private void CalculateInterest()
    {
      // Interest = Score;
      // Interest = -Score;

      // if (Move.IsCapture)
      // {
      //   Interest -= Me.pieceValues[(int)Move.MovePieceType] / 10;
      //   Interest += Me.pieceValues[(int)Move.CapturePieceType] / 10;
      // }
    }
  }

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 0 };

  int ColorEvaluationFactor(bool white) => white ? 1 : -1;

  int Evaluate()
  {
    if (_board.IsInCheckmate()) return -100000;

    if (_board.IsInsufficientMaterial() || _board.IsRepeatedPosition() || _board.FiftyMoveCounter >= 100) return -200;

    int materialEvaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      materialEvaluation += _board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      materialEvaluation -= _board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    return materialEvaluation * ColorEvaluationFactor(_board.IsWhiteToMove);
  }

  Dictionary<ulong, State> _reuseableStates = new Dictionary<ulong, State>();

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    ulong hash = board.ZobristKey;

    State tree;

    if (_reuseableStates.Count != 0) tree = _reuseableStates[hash];
    else tree = new State(Move.NullMove, this);

    tree.Move = Move.NullMove;

    for (int targetDepth = tree.Depth + 1; tree.ChildStates == null || timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 30; targetDepth++)
    {
      int lowerWindow = tree.Score - 100;
      int upperWindow = tree.Score + 100;

      tree.Expand(targetDepth, 0, lowerWindow, upperWindow);

      if (tree.Score <= lowerWindow || tree.Score >= upperWindow)
      {
        tree.Expand(targetDepth);
      }
    }

    int maxDepth = tree.Depth;

    tree = tree.ChildStates.MaxBy(state => -state.Score);

    _reuseableStates = new Dictionary<ulong, State>();

    if (tree.ChildStates != null) foreach (State state in tree.ChildStates) _reuseableStates[state.Hash] = state;

    return tree.Move;
  }
}