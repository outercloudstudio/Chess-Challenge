using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

/*
Think:
  Loop over available moves
  Add and Evaluate them if not already in list
  Sort list by evaluation
  Pick best move
  Search Move

Search Move:
  Loop over available moves
  Add and Evaluate them if not already in list
  Sort list by evaluation
  Update evaluation of previous move
*/

public class ARCNET : IChessBot
{
  class State
  {
    public Board CurrentBoard;
    public Move Move;
    public float Evaluation;
    public float Confidence;
    public List<State> PreviousStates = new List<State>();
    public List<State> NextStates = new List<State>();
    public bool WhiteMove;

    private void UpdateConfidence()
    {
      Confidence = Evaluation * (WhiteMove ? 1 : -1) - PreviousStates.Count * 2;
    }

    public State(Board board, Move move, float evaluation)
    {
      Move = move;
      Evaluation = evaluation;

      WhiteMove = board.IsWhiteToMove;

      UpdateConfidence();

      board.MakeMove(move);

      CurrentBoard = Board.CreateBoardFromFEN(board.GetFenString());

      board.UndoMove(move);
    }

    public State(State state, Move move, float evaluation)
    {
      Move = move;
      Evaluation = evaluation;

      WhiteMove = !state.WhiteMove;

      state.CurrentBoard.MakeMove(move);

      CurrentBoard = Board.CreateBoardFromFEN(state.CurrentBoard.GetFenString());

      state.CurrentBoard.UndoMove(move);

      PreviousStates = state.PreviousStates.ToList();
      PreviousStates.Add(state);

      UpdateConfidence();

      state.UpdateEvaluation();
    }

    public void UpdateEvaluation()
    {
      if (NextStates.Count == 0) return;

      float updatedEvaluation = NextStates[0].Evaluation;

      foreach (State nextState in NextStates) updatedEvaluation = WhiteMove ? Math.Min(updatedEvaluation, nextState.Evaluation) : Math.Max(updatedEvaluation, nextState.Evaluation);

      if (updatedEvaluation == Evaluation) return;

      Evaluation = updatedEvaluation;

      UpdateConfidence();

      if (PreviousStates.Count != 0) PreviousStates[PreviousStates.Count - 1].UpdateEvaluation();
    }
  }

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  float Evaluate(Board board)
  {
    if (board.IsInCheckmate()) return -10 * (board.IsWhiteToMove ? 1 : -1);

    if (board.IsDraw()) return 0;

    float evaluation = 0;

    for (int index = 0; index < 64; index++)
    {
      Piece piece = board.GetPiece(new Square(index));

      evaluation += pieceValues[(int)piece.PieceType] * (piece.IsWhite ? 1 : -1);
    }

    if (board.IsInCheck()) evaluation += -0.5f * (board.IsWhiteToMove ? 1 : -1);

    return evaluation;
  }

  float Evaluate(Board board, Move move)
  {
    board.MakeMove(move);

    float evaluation = Evaluate(board);

    board.UndoMove(move);

    return evaluation;
  }

  List<State> _searchedStates = new List<State>();
  List<State> _statesToSearch = new List<State>();

  void Search()
  {
    State bestState = _statesToSearch[0];

    Search(bestState);
  }

  void Search(State state)
  {
    _statesToSearch.Remove(state);
    _searchedStates.Add(state);

    foreach (Move move in state.CurrentBoard.GetLegalMoves())
    {
      State newState = new State(state, move, Evaluate(state.CurrentBoard, move));

      _statesToSearch.Add(newState);
      state.NextStates.Add(newState);
    }

    _statesToSearch = _statesToSearch.OrderByDescending(x => x.Confidence).ToList();

  }

  // void Debug(State targetState, int depth = 0)
  // {
  //   Console.WriteLine(new string('\t', depth) + targetState.Move + " " + targetState.Evaluation + " " + targetState.Confidence + " " + targetState.PreviousStates.Count + " " + targetState.CurrentBoard.GetFenString());

  //   foreach (State state in _searchedStates.Where(otherState => otherState.PreviousStates.Count == depth + 1 && otherState.PreviousStates.Contains(targetState))) Debug(state, depth + 1);
  //   foreach (State state in _statesToSearch.Where(otherState => otherState.PreviousStates.Count == depth + 1 && otherState.PreviousStates.Contains(targetState))) Debug(state, depth + 1);
  // }

  public Move Think(Board board, Timer timer)
  {
    string boardFen = board.GetFenString();

    _searchedStates = _searchedStates.Where(state => state.PreviousStates.Count == 2 && state.CurrentBoard.GetFenString() == boardFen).ToList();
    _statesToSearch = _statesToSearch.Where(state => state.PreviousStates.Count == 2 && state.CurrentBoard.GetFenString() == boardFen).ToList();

    foreach (State state in _searchedStates) state.PreviousStates.RemoveAt(0);
    foreach (State state in _statesToSearch) state.PreviousStates.RemoveAt(0);

    List<Move> moves = new List<Move>();

    foreach (Move move in board.GetLegalMoves()) _statesToSearch.Add(new State(board, move, Evaluate(board, move)));

    _statesToSearch = _statesToSearch.OrderByDescending(state => state.Confidence).ToList();

    for (int i = 0; i < 1000; i++) Search();

    _searchedStates = _searchedStates.OrderByDescending(state => state.Evaluation).ToList();

    List<State> searchedPossibleMoves = _searchedStates.Where(state => state.PreviousStates.Count == 0).ToList();

    if (searchedPossibleMoves.Count == 0) searchedPossibleMoves = _statesToSearch.Where(state => state.PreviousStates.Count == 0).ToList();

    State bestState = searchedPossibleMoves[board.IsWhiteToMove ? 0 : (searchedPossibleMoves.Count - 1)];

    // Debug(bestState);

    return bestState.Move;
  }
}