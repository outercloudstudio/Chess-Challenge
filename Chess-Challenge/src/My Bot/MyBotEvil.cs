using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBotEvil : IChessBot
{
  Board _board;
  int _maxDepth = 0;

  record class TranspositionEntry(ulong Hash, int Depth, int Score);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[100000];

  class State
  {
    public Move Move;
    public int Score;
    public State[] ChildStates = null;
    public int Key;
    public ulong Hash;

    MyBotEvil Me;

    public void Expand(int targetDepth, int depth = 0, int alpha = -99999, int beta = 99999)
    {
      TranspositionEntry entry = Me._transpositionTable[Key];

      int evaluationDepth = targetDepth - depth;

      if (entry != null && entry.Hash == Hash && entry.Depth >= evaluationDepth)
      {
        Score = entry.Score;

        return;
      }

      if (depth > targetDepth) return;

      Me._board.MakeMove(Move);

      Me._maxDepth = Math.Max(Me._maxDepth, depth);

      if (ChildStates == null)
      {
        ChildStates = Me._board.GetLegalMoves().Select(move => new State(move, Me)).OrderByDescending(state => -state.Score).ToArray();

        if (ChildStates.Length != 0) Score = -ChildStates[0].Score;
      }
      else
      {
        int max = -99999;

        foreach (State state in ChildStates)
        {
          state.Expand(targetDepth, depth + 1, -beta, -alpha);

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

        ChildStates = ChildStates.OrderByDescending(state => -state.Score).ToArray();
      }

      if (entry == null || entry.Depth < evaluationDepth) Me._transpositionTable[Key] = new TranspositionEntry(Hash, evaluationDepth, Score);

      Me._board.UndoMove(Move);
    }

    public State(Move move, MyBotEvil me)
    {
      Me = me;

      Move = move;

      Me._board.MakeMove(move);

      Hash = Me._board.ZobristKey;
      Key = (int)(Hash % (ulong)Me._transpositionTable.Length);

      Score = Me.Evaluate();

      Me._board.UndoMove(move);
    }
  }

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  int ColorEvaluationFactor(bool white) => white ? 1 : -1;

  int Evaluate()
  {
    if (_board.IsInCheckmate()) return -1000;

    if (_board.IsInsufficientMaterial() || _board.IsRepeatedPosition() || _board.FiftyMoveCounter >= 100) return -2;

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
    _maxDepth = 0;

    ulong hash = board.ZobristKey;

    State tree;

    if (_reuseableStates.Count != 0) tree = _reuseableStates[hash];
    else tree = new State(Move.NullMove, this);

    tree.Move = Move.NullMove;

    for (int targetDepth = 0; tree.ChildStates == null || timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60; targetDepth++) tree.Expand(targetDepth);

    tree = tree.ChildStates.MaxBy(state => -state.Score);

    _reuseableStates = new Dictionary<ulong, State>();

    if (tree.ChildStates != null) foreach (State state in tree.ChildStates) _reuseableStates[state.Hash] = state;

    Console.WriteLine(String.Format("My Bot Evil: Searched to depth of {0} in {1}", _maxDepth, timer.MillisecondsElapsedThisTurn)); //#DEBUG

    return tree.Move;
  }
}