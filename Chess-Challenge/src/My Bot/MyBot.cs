using System;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  class State { public Move Move; public State[] Children; public State Parent; public int Visits = 1; public float TotalScore; }
  float Score(State state) => -state.TotalScore / MathF.Pow(state.Visits, 1f);

  public Move Think(Board board, Timer timer)
  {
    State root = new State() { Move = default };

    do
    {
      State currentState = root;

      while (currentState.Children != null)
      {
        if (currentState.Children.Length == 0) break;

        currentState = currentState.Children.MinBy(Score);

        board.MakeMove(currentState.Move);
      }

      if (currentState.Children == null) currentState.Children = board.GetLegalMoves().Select(move => new State() { Move = move, TotalScore = Evaluate(move), Parent = currentState }).ToArray();

      float score = currentState.Children.Length == 0 ? (board.IsDraw() ? 0 : -100) : currentState.Children.MinBy(Score).TotalScore;

      while (currentState.Parent != null)
      {
        currentState.Visits++;

        currentState.TotalScore += score *= -1;

        board.UndoMove(currentState.Move);

        currentState = currentState.Parent;
      }

      // DebugState(root);
      // Console.WriteLine("\n");
    } while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60f);

    return root.Children.MinBy(Score).Move;

    float Evaluate(Move move)
    {
      board.MakeMove(move);

      int moveSide = board.IsWhiteToMove ? 1 : -1;

      for (int i = 0; i < 36; i++)
      {
        Array.Copy(_emptyTensor, _sightTensor, 54);

        for (int kernelI = 0; kernelI < 9; kernelI++)
        {
          Piece piece = board.GetPiece(new Square(i / 6 + kernelI / 6, i % 6 + kernelI % 6));

          if (piece.PieceType != PieceType.None) _sightTensor[kernelI / 6 * 18 + kernelI % 6 * 6 + (int)piece.PieceType - 1] = piece.IsWhite ? 1 : -1;
        }

        parameterOffset = 0;

        Array.Copy(_sightTensor, _layerInput, 54);

        Layer(54, 12);
        Layer(12, 12);
        Layer(12, 2);

        Array.Copy(_layerOutput, 0, _evaluationTensor, 2, i * 2);
      }

      _evaluationTensor[72] = moveSide;

      Array.Copy(_evaluationTensor, _layerInput, 73);

      Layer(73, 32);
      Layer(32, 32);
      Layer(32, 1);

      int evaluation = 0;

      for (int type = 1; type < 14; type++) evaluation += board.GetPieceList((PieceType)type, type % 2 == 0).Count * pieceValues[type / 2] * (type % 2 * 2 - 1);

      board.UndoMove(move);

      return (_layerOutput[0] + evaluation) * -moveSide;
    }
  }

  // void DebugState(State state, int depth = 0)
  // {
  //   // if (depth == 2) return;

  //   Console.WriteLine(new string('\t', depth) + $"{state.Move} {Score(state)} {state.TotalScore} {state.Visits}");

  //   if (state.Children == null) return;

  //   foreach (State child in state.Children) DebugState(child, depth + 1);
  // }

  int[] pieceValues = { 0, 1, 3, 3, 5, 9, 1000 };

  public MyBot()
  {
    for (int parameter = 0; parameter < 4299; parameter++)
    {
      var ints = decimal.GetBits(_compressedParameters[parameter / 16]);
      int bits = parameter % 16 * 6, bitsOffset = bits % 32, intIndex = bits / 32, quantized = ints[intIndex] >> bitsOffset & 0b111111;
      if (bitsOffset > 27) quantized |= ints[intIndex + 1] << 32 - bitsOffset & 0b111111;

      _parameters[parameter] = MathF.Pow(quantized / 64f - 0.5f, 3) * 6f;
    }
  }

  int parameterOffset = 0;

  float[] _parameters = new float[4299], _layerInput = new float[73], _layerOutput = new float[32], _evaluationTensor = new float[73], _sightTensor = new float[54], _emptyTensor = new float[73];

  void Layer(int previousLayerSize, int layerSize)
  {
    Array.Copy(_emptyTensor, _layerOutput, 16);

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++) _layerOutput[nodeIndex] += _layerInput[weightIndex] * _parameters[parameterOffset + nodeIndex * previousLayerSize + weightIndex];

      _layerOutput[nodeIndex] = MathF.Max(MathF.Min(_layerOutput[nodeIndex] + _parameters[parameterOffset + layerSize * previousLayerSize + nodeIndex], 1), -1);
    }

    parameterOffset += layerSize * previousLayerSize + layerSize;

    Array.Copy(_layerOutput, _layerInput, layerSize);
  }

  decimal[] _compressedParameters = { };
}