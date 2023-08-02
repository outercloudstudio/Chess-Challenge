using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBotNoTransposition : IChessBot
{
  static Board _board;
  static int _maxDepth = 0;

  class State
  {
    public Move Move;
    public int Score;
    public State[] ChildStates = null;

    public void Expand(int targetDepth, int depth = 0, int alpha = -99999, int beta = 99999)
    {
      if (depth > targetDepth) return;

      _maxDepth = Math.Max(_maxDepth, depth);

      _board.MakeMove(Move);

      // Console.WriteLine(String.Format("Transposition check. entry depth: {0} evaluation depth: {1} call depth: {2} target depth: {3}", _transpositionEntry.Depth, targetDepth - depth, depth, targetDepth));

      if (ChildStates == null) ChildStates = _board.GetLegalMoves().Select(move => new State(move, targetDepth, depth)).OrderByDescending(state => -state.Score).ToArray();

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

      _board.UndoMove(Move);
    }

    public State(Move move, int targetDepth, int depth = 0)
    {
      Move = move;

      _board.MakeMove(move);

      if (move.IsNull)
      {
        _board.UndoMove(move);

        return;
      }

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

  // Dictionary<string, State> _reuseableStates = new Dictionary<string, State>();

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _maxDepth = 0;

    string boardFen = board.GetFenString();

    State tree = new State(Move.NullMove, -1);

    // if (_reuseableStates.ContainsKey(boardFen)) tree = _reuseableStates[boardFen];
    // else tree = new State(Move.NullMove, -1);

    // for (int targetDepth = 0; timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60 || tree.ChildStates == null; targetDepth++)
    // {
    //   // Console.WriteLine("\nSearching Depth " + targetDepth);

    //   tree.Expand(targetDepth, true);
    // }

    for (int targetDepth = 0; tree.ChildStates == null || timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60; targetDepth++) tree.Expand(targetDepth);

    // foreach (State state in tree.ChildStates) Console.WriteLine(String.Format("{0} Score: {1}", state.Move, state.Score)); //#DEBUG

    tree = tree.ChildStates.MaxBy(state => -state.Score);

    // _reuseableStates = new Dictionary<string, State>();

    // if (tree.ChildStates != null)
    // {
    //   board.MakeMove(tree.Move);

    //   foreach (State state in tree.ChildStates)
    //   {
    //     board.MakeMove(state.Move);

    //     string fen = board.GetFenString();

    //     board.UndoMove(state.Move);

    //     _reuseableStates[fen] = state;

    //     state.Move = Move.NullMove;

    //     break;
    //   }

    //   board.UndoMove(tree.Move);
    // }

    Console.WriteLine(String.Format("My Bot: Searched to depth of {0} in {1}", _maxDepth, timer.MillisecondsElapsedThisTurn));

    return tree.Move;
  }
}