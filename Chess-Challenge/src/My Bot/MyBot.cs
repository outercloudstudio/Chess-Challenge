using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  static Board _board;
  static int _maxDepth = 0;
  static int _currentDepth = 0;

  class State
  {
    public Move Move;
    public float Score;
    public State[] ChildStates = null;

    public void Expand()
    {
      _currentDepth++;

      _board.MakeMove(Move);

      if (ChildStates == null)
      {
        ChildStates = _board.GetLegalMoves().Select(move => new State(move)).ToArray();

        foreach (State state in ChildStates) Console.WriteLine(new String('\t', _currentDepth) + String.Format("{0} Score: {1}", state.Move, state.Score));
      }
      else
      {
        foreach (State state in ChildStates)
        {
          state.Expand();

          Console.WriteLine(new String('\t', _currentDepth) + String.Format("{0} Score: {1}", state.Move, state.Score));
        }
      }

      ChildStates = ChildStates.OrderByDescending(state => -state.Score).ToArray();

      Console.WriteLine(new String('\t', _currentDepth) + "Sorted");

      foreach (State state in ChildStates) Console.WriteLine(new String('\t', _currentDepth) + String.Format("{0} Score: {1}", state.Move, state.Score));

      if (ChildStates.Length != 0) Score = -ChildStates[0].Score;

      _board.UndoMove(Move);

      _currentDepth--;
    }

    public State ParentState;
    public bool WhiteMove;

    public State(Move move)
    {
      _maxDepth = Math.Max(_maxDepth, _currentDepth);

      Move = move;

      if (move.IsNull) return;

      _board.MakeMove(move);

      Score = Evaluate();

      _board.UndoMove(move);
    }

    // public State(bool whiteMove)
    // {
    //   WhiteMove = whiteMove;
    // }

    // public void UpdateEvaluation()
    // {
    //   if (ChildStates.Length == 0) return;

    //   Evaluation = WhiteMove ? ChildStates.MinBy(state => state.Evaluation).Evaluation : ChildStates.MaxBy(state => state.Evaluation).Evaluation;
    //   Interest = Evaluation;

    //   if (ParentState != null) ParentState.UpdateEvaluation();
    // }

    // public void Search(Board board)
    // {
    //   if (ParentState != null) board.MakeMove(Move);

    //   if (ChildStates == null)
    //   {
    //     ChildStates = board.GetLegalMoves().Select(move =>
    //     {
    //       State state = new State(!WhiteMove)
    //       {
    //         ParentState = this,
    //         Move = move
    //       };

    //       board.MakeMove(move);

    //       state.Evaluation = Evaluate(board);

    //       board.UndoMove(move);

    //       state.Interest = InterestEvaluation(board, move, state.Evaluation);

    //       return state;
    //     }).ToArray();

    //     if (ChildStates.Length != 0) UpdateEvaluation();

    //     if (ParentState != null) board.UndoMove(Move);

    //     return;
    //   }

    //   if (ChildStates.Length == 0)
    //   {
    //     if (ParentState != null) board.UndoMove(Move);

    //     return;
    //   }

    //   ChildStates.MaxBy(state => state.Interest * ColorEvaluationFactor(!WhiteMove)).Search(board);

    //   if (ParentState != null) board.UndoMove(Move);
    // }
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

  // public int Negamax(int depth, bool logging = false)
  // {
  //   var moves = _board.GetLegalMoves();

  //   if (depth == 0 || moves.Length == 0) return Evaluate();

  //   int max = -9999999;

  //   foreach (Move move in moves)
  //   {
  //     _board.MakeMove(move);

  //     int score = -Negamax(depth - 1, logging);

  //     _board.UndoMove(move);

  //     max = Math.Max(max, score);
  //   }

  //   return max;
  // }

  Dictionary<string, State> _reuseableStates = new Dictionary<string, State>();

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _maxDepth = 0;
    _currentDepth = 0;

    string boardFen = board.GetFenString();

    State tree;

    if (_reuseableStates.ContainsKey(boardFen))
    {
      tree = _reuseableStates[boardFen];
    }
    else
    {
      tree = new State(Move.NullMove);
    }

    tree.Expand();

    Console.WriteLine("\n Expanding 2 \n");

    tree.Expand();

    tree = tree.ChildStates.MaxBy(state => -state.Score);

    _reuseableStates = new Dictionary<string, State>();

    // if (tree.ChildStates != null)
    // {
    //   board.MakeMove(tree.Move);

    //   foreach (State state in tree.ChildStates)
    //   {
    //     board.MakeMove(state.Move);
    //     string fen = board.GetFenString();
    //     board.UndoMove(state.Move);

    //     state.ParentState = null;

    //     _reuseableStates[fen] = state;

    //     break;
    //   }

    //   board.UndoMove(tree.Move);
    // }

    Console.WriteLine(String.Format("Searched to depth of {0}", _maxDepth));

    return tree.Move;
  }
}