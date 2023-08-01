using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class ARCNET2_MoveOrdering : IChessBot
{
  static ulong[] CompressedParameters = new ulong[] { 13288209464809731723UL, 12518802469381486097UL, 3344523617948874324UL, 4621879348935538624UL, 12178083276082393869UL, 13321416734633273190UL, 13384900171568493081UL, 3471778089610492018UL, 3830361185332704128UL, 3835008606042371145UL, 13081466270535398355UL, 2515320208709328083UL, 3151708440055164235UL, 13158730694294843620UL, 12973381103348462651UL, 3798127732735323639UL, 3869354179804147255UL, 3479797779242888857UL, 12515259565742077142UL, 12711891200001683535UL, 12804065710193783144UL, 13002935727012852146UL, 12739475144316990869UL, 12131489040247107026UL, 3662183103667185904UL, 3470353916969858525UL, 12767055669924966986UL, 3342001351105948584UL, 3658388912192203276UL, 12700477092404637092UL, 12798022240337604874UL, 3911710846555208478UL, 12547368829347670087UL, 2660549279439435076UL, 3373786664733025495UL, 3776748246727145080UL, 2839151485012351975UL, 3822767519941767691UL, 12381293975922128052UL, 12443904750846752069UL, 3582667457826139794UL, 12972097054180453250UL, 11650444151655248500UL, 2699685715056407327UL, 3923122170955019297UL, 3144270031218521279UL, 8990UL };
  static float[] Parameters = new float[185];

  public ARCNET2_MoveOrdering()
  {
    for (int i = 0; i < 185; i++) Parameters[i] = (float)BitConverter.ToHalf(BitConverter.GetBytes(CompressedParameters[i / 4]), i % 4 * 2);
  }

  static float[] Layer(float[] input, int previousLayerSize, int layerSize, ref int parameterOffset, Func<float, float> activationFunction)
  {
    var layer = new float[layerSize];

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * Parameters[parameterOffset + nodeIndex * previousLayerSize + weightIndex];
      }

      layer[nodeIndex] = activationFunction(layer[nodeIndex] + Parameters[parameterOffset + layerSize * previousLayerSize + nodeIndex]);
    }

    parameterOffset += layerSize * previousLayerSize + layerSize;

    return layer;
  }

  static float ColorEvaluationFactor(bool white) { return white ? 1 : -1; }

  class State
  {
    public Move Move;
    public float Evaluation;
    public float Interest;
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
      Interest = Evaluation;

      if (ParentState != null) ParentState.UpdateEvaluation();
    }

    public void Search(Board board)
    {
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

          state.Interest = InterestEvaluation(board, move, state.Evaluation);

          return state;
        }).ToArray();

        if (ChildStates.Length != 0) UpdateEvaluation();

        if (ParentState != null) board.UndoMove(Move);

        return;
      }

      if (ChildStates.Length == 0)
      {
        if (ParentState != null) board.UndoMove(Move);

        return;
      }

      ChildStates.MaxBy(state => state.Interest * ColorEvaluationFactor(!WhiteMove)).Search(board);

      if (ParentState != null) board.UndoMove(Move);
    }
  }

  static int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 0 };

  static float Evaluate(Board board)
  {
    if (board.IsInCheckmate()) return -1000 * ColorEvaluationFactor(board.IsWhiteToMove);

    if (board.IsInsufficientMaterial() || board.IsRepeatedPosition() || board.FiftyMoveCounter >= 100) return -0.5f * ColorEvaluationFactor(board.IsWhiteToMove);

    float materialEvaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      materialEvaluation += board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      materialEvaluation -= board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    float checkEvaluation = 0;

    if (board.IsInCheck()) checkEvaluation += -0.5f * ColorEvaluationFactor(board.IsWhiteToMove);

    var input = new float[] { materialEvaluation, checkEvaluation, board.PlyCount };

    int parameterOffset = 0;

    var ReLU = (float x) => Math.Max(0, x);

    var hidden1 = Layer(input, 3, 8, ref parameterOffset, ReLU);
    var hidden2 = Layer(hidden1, 8, 8, ref parameterOffset, ReLU);
    var hidden3 = Layer(hidden2, 8, 8, ref parameterOffset, ReLU);
    float output = Layer(hidden3, 8, 1, ref parameterOffset, ReLU)[0];

    return materialEvaluation + output * 0.5f;
  }

  static float InterestEvaluation(Board board, Move move, float evaluation)
  {
    if (move.IsCapture) evaluation += 0.1f;
    if (move.IsCastles) evaluation += 1f;
    if (move.IsPromotion) evaluation += 0.3f;
    if (move.MovePieceType == PieceType.Pawn) evaluation += 0.2f - board.PlyCount * 0.01f;
    if (move.MovePieceType == PieceType.King)
    {
      evaluation -= 0.2f;
      evaluation += (MathF.Abs(move.TargetSquare.File - 4.5f) + MathF.Abs(move.TargetSquare.Rank - 4.5f)) * 0.01f;
    }
    else
    {
      evaluation -= (MathF.Abs(move.TargetSquare.File - 4.5f) + MathF.Abs(move.TargetSquare.Rank - 4.5f)) * 0.02f;
    }



    board.MakeMove(move);

    if (board.IsInCheck()) evaluation += 0.05f;

    evaluation -= board.GetLegalMoves().Count() * 0.001f;

    board.UndoMove(move);

    return evaluation;
  }

  Dictionary<string, State> _reuseableStates = new Dictionary<string, State>();

  public Move Think(Board board, Timer timer)
  {
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

    _tree = _tree.ChildStates.MaxBy(state => state.Evaluation * ColorEvaluationFactor(board.IsWhiteToMove));

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

    return _tree.Move;
  }
}