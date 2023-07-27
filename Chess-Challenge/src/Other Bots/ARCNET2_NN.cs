using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class ARCNET2_NN : IChessBot
{
  //Stop over evaluating checkmates
  //stop evaluating over 20 moves deep

  static float[] Parameters;

  public ARCNET2_NN()
  {
    string[] stringParameters = System.IO.File.ReadAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 2.txt").Split('\n');

    Parameters = stringParameters[..(stringParameters.Length - 1)].Select(float.Parse).ToArray();
  }

  static float Parameter(int index)
  {
    return Parameters[index];
  }

  static float[] Layer(float[] input, int previousLayerSize, int layerSize, ref int parameterOffset, Func<float, float> activationFunction)
  {
    float[] layer = new float[layerSize];

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * Parameter(parameterOffset + nodeIndex * previousLayerSize + weightIndex);
      }

      layer[nodeIndex] = activationFunction(layer[nodeIndex] + Parameter(parameterOffset + layerSize * previousLayerSize + nodeIndex));
    }

    parameterOffset += layerSize * previousLayerSize + layerSize;

    return layer;
  }

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

        if (ChildStates.Length != 0) UpdateEvaluation();

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

    float materialEvaluation = 0;

    for (int typeIndex = 1; typeIndex < 7; typeIndex++)
    {
      materialEvaluation += board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
      materialEvaluation -= board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
    }

    float checkEvaluation = 0;

    if (board.IsInCheck()) checkEvaluation += -0.5f * (board.IsWhiteToMove ? 1 : -1);

    float[] input = new float[] { materialEvaluation, checkEvaluation, board.PlyCount };

    int parameterOffset = 0;

    var ReLU = (float x) => Math.Max(0, x);

    float[] hidden1 = Layer(input, 3, 8, ref parameterOffset, ReLU);
    float[] hidden2 = Layer(hidden1, 8, 8, ref parameterOffset, ReLU);
    float[] hidden3 = Layer(hidden2, 8, 8, ref parameterOffset, ReLU);
    float output = Layer(hidden3, 8, 1, ref parameterOffset, ReLU)[0];

    return materialEvaluation + checkEvaluation;
    // return materialEvaluation + checkEvaluation + output * 0.5f;
    // return output;
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

    Console.WriteLine("Arcnet 2 Searched " + _statesSearched + " states. Max depth: " + _maxDepthSearched);

    return _tree.Move;
  }
}