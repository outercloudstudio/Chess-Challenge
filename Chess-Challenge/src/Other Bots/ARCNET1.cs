using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;


//quiscene
//transposition table
//not ordering moves to search
//playing unsearched but evaluated moves with less confidence

public class ARCNET1 : IChessBot
{
  class State
  {
    public string BoardFen;
    public Move Move;
    public float Evaluation;
    public float Confidence;
    public List<State> PreviousStates = new List<State>();
    public List<State> NextStates = new List<State>();
    public bool WhiteMove;

    private Board _boardCached;

    public void UpdateConfidence()
    {
      Confidence = Evaluation * (WhiteMove ? 1 : -1) - MathF.Pow(PreviousStates.Count, 4) * 2;
    }

    public State(Board board, Move move, float evaluation)
    {
      Move = move;
      Evaluation = evaluation;

      UpdateConfidence();

      WhiteMove = board.IsWhiteToMove;

      board.MakeMove(move);

      BoardFen = board.GetFenString();
      _boardCached = Board.CreateBoardFromFEN(BoardFen);

      board.UndoMove(move);
    }

    public State(State state, Move move, float evaluation)
    {
      Move = move;
      Evaluation = evaluation;

      WhiteMove = !state.WhiteMove;

      PreviousStates = state.PreviousStates.ToList();
      PreviousStates.Add(state);

      state.NextStates.Add(this);
    }

    public void UpdateEvaluation()
    {
      if (NextStates.Count == 0) return;

      float updatedEvaluation = NextStates[0].Evaluation;

      foreach (State nextState in NextStates) updatedEvaluation = WhiteMove ? MathF.Min(updatedEvaluation, nextState.Evaluation) : MathF.Max(updatedEvaluation, nextState.Evaluation);

      if (updatedEvaluation == Evaluation) return;

      Evaluation = updatedEvaluation;

      UpdateConfidence();

      if (PreviousStates.Count != 0) PreviousStates[PreviousStates.Count - 1].UpdateEvaluation();
    }

    public Board CurrentBoard()
    {
      if (_boardCached == null)
      {
        Board previousBoard = PreviousStates[PreviousStates.Count - 1].CurrentBoard();

        previousBoard.MakeMove(Move);

        _boardCached = Board.CreateBoardFromFEN(previousBoard.GetFenString());

        previousBoard.UndoMove(Move);

        BoardFen = _boardCached.GetFenString();
      }

      return _boardCached;
    }

    public string GetBoardFen()
    {
      CurrentBoard();

      return BoardFen;
    }
  }

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  float Evaluate(Board board)
  {
    if (board.IsInCheckmate()) return -1000 * (board.IsWhiteToMove ? 1 : -1);

    if (board.IsDraw()) return -0.5f;

    float evaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      evaluation += board.GetPieceList((PieceType)typeIndex, true).Count * -pieceValues[typeIndex];
      evaluation += board.GetPieceList((PieceType)typeIndex, false).Count * -pieceValues[typeIndex];
    }

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

    if (board.PlyCount >= 50 && move.MovePieceType == PieceType.Pawn) evaluation += 1f;

    return evaluation;
  }

  List<State> _searchedStates = new List<State>();
  List<State> _statesToSearch = new List<State>();

  int _searches = 0;

  void Search()
  {
    State bestState = _statesToSearch.MaxBy(state => state.Confidence);

    Search(bestState);
  }

  void Search(State state)
  {
    _searches++;

    _statesToSearch.Remove(state);
    _searchedStates.Add(state);

    string boardFen = state.CurrentBoard().GetFenString();

    foreach (Move move in state.CurrentBoard().GetLegalMoves())
    {
      State newState = new State(state, move, Evaluate(state.CurrentBoard(), move));
      _statesToSearch.Add(newState);
    }

    state.UpdateEvaluation();

    state.UpdateConfidence();
  }

  // void Debug(State targetState, int depth = 0, int maxDepth = 99999)
  // {
  //   Console.WriteLine(new string('\t', depth) + targetState.Move + " " + targetState.WhiteMove + " " + targetState.Evaluation + " " + targetState.Confidence + " " + targetState.PreviousStates.Count + " " + targetState.BoardFen);

  //   if (depth >= maxDepth) return;

  //   foreach (State state in _searchedStates.Where(otherState => otherState.PreviousStates.Count == depth + 1 && otherState.PreviousStates.Contains(targetState))) Debug(state, depth + 1, maxDepth);
  //   foreach (State state in _statesToSearch.Where(otherState => otherState.PreviousStates.Count == depth + 1 && otherState.PreviousStates.Contains(targetState))) Debug(state, depth + 1, maxDepth);
  // }

  public Move Think(Board board, Timer timer)
  {
    string boardFen = board.GetFenString();

    var canReuseState = (State state) => (state.PreviousStates.Count >= 2 && state.PreviousStates[1].GetBoardFen() == boardFen && state.CurrentBoard() != null);

    _searchedStates = _searchedStates.Where(canReuseState).ToList();
    _statesToSearch = _statesToSearch.Where(canReuseState).ToList();

    foreach (State state in _searchedStates) state.PreviousStates.RemoveRange(0, 2);
    foreach (State state in _statesToSearch) state.PreviousStates.RemoveRange(0, 2);

    Console.WriteLine("Reused " + (_statesToSearch.Count + _searchedStates.Count) + " states!");

    List<Move> moves = new List<Move>();

    foreach (Move move in board.GetLegalMoves()) _statesToSearch.Add(new State(board, move, Evaluate(board, move)));

    _searches = 0;

    while (timer.MillisecondsElapsedThisTurn < Math.Min(1000, timer.MillisecondsRemaining / 10)) Search();

    List<State> searchedPossibleMoves = _searchedStates.Concat(_statesToSearch).Where(state => state.PreviousStates.Count == 0).ToList();

    State bestState = searchedPossibleMoves.MaxBy(state => (state.PreviousStates.Count > 0) ? state.Confidence : (state.Confidence - 2));

    Console.WriteLine("Found move " + bestState.Move + " in " + timer.MillisecondsElapsedThisTurn / 1000f + "s with " + _searches + " searches and " + (_searchedStates.Count + _statesToSearch.Count) + " states!");

    // Debug(bestState);

    return bestState.Move;
  }
}