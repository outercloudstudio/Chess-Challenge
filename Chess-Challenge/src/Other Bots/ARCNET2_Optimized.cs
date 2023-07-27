using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class ARCNET2_Optimized : IChessBot
{
  static int _statesSearched;
  static float _maxDepthSearched;

  class State
  {
    public Move Move;
    public float Evaluation;
    public State ParentState;
    public State[] ChildStates = null;
    public bool WhiteMove;

    public State(bool whiteMove)
    {
      WhiteMove = whiteMove;
    }

    public void UpdateEvaluation()
    {
      if (ChildStates.Length == 0) return;

      Evaluation = WhiteMove ? ChildStates.MinBy(state => state.Evaluation).Evaluation : ChildStates.MaxBy(state => state.Evaluation).Evaluation;

      if (ParentState != null) ParentState.UpdateEvaluation();
    }

    public void Search(Board board, float depth = 0.5f)
    {
      _statesSearched++;
      _maxDepthSearched = Math.Max(_maxDepthSearched, (int)depth);

      if (ParentState != null) board.MakeMove(Move);

      if (ChildStates == null)
      {
        ChildStates = board.GetLegalMoves().Select(move =>
        {
          State state = new State(!WhiteMove)
          {
            ParentState = this,
            Move = move
          };

          board.MakeMove(move);

          state.Evaluation = Evaluate(board);

          board.UndoMove(move);

          return state;
        }).ToArray();

        UpdateEvaluation();

        if (ParentState != null) board.UndoMove(Move);

        return;
      }

      if (ChildStates.Length == 0)
      {
        if (ParentState != null) board.UndoMove(Move);

        return;
      }

      ChildStates.MaxBy(state => state.Evaluation * (WhiteMove ? -1 : 1)).Search(board, depth + 0.5f);

      if (ParentState != null) board.UndoMove(Move);
    }
  }

  static int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  static float Evaluate(Board board)
  {
    if (board.IsInCheckmate()) return -1000 * (board.IsWhiteToMove ? 1 : -1);

    if (board.IsInsufficientMaterial() || board.IsRepeatedPosition() || board.FiftyMoveCounter >= 100) return -0.5f * (board.IsWhiteToMove ? 1 : -1);

    float evaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      evaluation += board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      evaluation -= board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    if (board.IsInCheck()) evaluation += -0.5f * (board.IsWhiteToMove ? 1 : -1);

    return evaluation;
  }

  // void Debug(State targetState, int depth = 0, int maxDepth = 99999)
  // {
  //   Console.WriteLine(new string('\t', depth) + targetState.Move + " Evaluation: " + targetState.Evaluation + " White Move: " + targetState.WhiteMove);

  //   if (depth >= maxDepth) return;

  //   if (targetState.ChildStates == null) return;

  //   foreach (State state in targetState.ChildStates) Debug(state, depth + 1, maxDepth);
  // }

  Dictionary<string, State> _reuseableStates = new Dictionary<string, State>();

  public Move Think(Board board, Timer timer)
  {
    _statesSearched = 0;
    _maxDepthSearched = 0;

    string boardFen = board.GetFenString();

    State _tree;

    if (_reuseableStates.ContainsKey(boardFen))
    {
      _tree = _reuseableStates[boardFen];
    }
    else
    {
      _tree = new State(!board.IsWhiteToMove);
    }

    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60 || _tree.ChildStates == null) _tree.Search(board);

    _tree = _tree.ChildStates.MaxBy(state => state.Evaluation * (board.IsWhiteToMove ? 1 : -1));

    _reuseableStates = new Dictionary<string, State>();

    if (_tree.ChildStates != null)
    {
      board.MakeMove(_tree.Move);

      foreach (State state in _tree.ChildStates)
      {
        board.MakeMove(state.Move);
        string fen = board.GetFenString();
        board.UndoMove(state.Move);

        state.ParentState = null;

        _reuseableStates[fen] = state;

        break;
      }

      board.UndoMove(_tree.Move);
    }

    // Debug(_tree);

    Console.WriteLine("Arcnet 2 Optimized Searched " + _statesSearched + " states. Max depth: " + _maxDepthSearched);

    return _tree.Move;
  }
}