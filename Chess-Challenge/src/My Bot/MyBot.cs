using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  public float[] Weights;

  public MyBot()
  {
    string[] stringWeights = System.IO.File.ReadAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 3.txt").Split('\n');

    Weights = stringWeights[..(stringWeights.Length - 1)].Select(float.Parse).ToArray();

    Console.WriteLine("Weights: " + Weights.Length);
  }

  float Weight(int index)
  {
    return Weights[index];
  }

  float TanH(float x)
  {
    return (float)Math.Tanh(x);
  }

  float[] Layer(float[] input, int previousLayerSize, int layerSize, int layerOffset, Func<float, float> activationFunction)
  {
    float[] layer = new float[layerSize];

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * Weight(layerOffset + nodeIndex * previousLayerSize + weightIndex);
      }

      layer[nodeIndex] = activationFunction(layer[nodeIndex] + Weight(layerOffset + layerSize * previousLayerSize + nodeIndex));
    }

    return layer;
  }

  float GetPieceId(Board board, int index)
  {
    if (index >= 64) return 0;

    Piece piece = board.GetPiece(new Square(index));

    return (float)((int)piece.PieceType * (piece.IsWhite ? 1 : -1)) / 6;
  }

  int PositionToIndex(int x, int y)
  {
    return new Square(x, y).Index;
  }

  float Inference(Board board, Move move)
  {
    board.MakeMove(move);

    float[] input = new float[64];
    for (int x = 0; x < 8; x++)
    {
      for (int y = 0; y < 8; y++)
      {
        input[x * 8 + y] = GetPieceId(board, PositionToIndex(x, y));
      }
    }

    float[] hiddenLayer1 = Layer(input, 64, 16, 0, TanH);
    float[] hiddenLayer2 = Layer(hiddenLayer1, 16, 16, 64 * 16 + 16, TanH);
    float[] hiddenLayer3 = Layer(hiddenLayer2, 16, 16, 64 * 16 + 16 + 16 * 16 + 16, TanH);
    float[] output = Layer(hiddenLayer3, 16, 1, 64 * 16 + 16 + 16 * 16 + 16 + 16 * 1 + 1, (x) => x);

    if (board.IsInCheckmate())
    {
      board.UndoMove(move);

      return 9999999;
    }

    // float[] flattened = new float[16];

    // int flatteneIndex = 0;
    // for (int convolutionX = 1; convolutionX < 8; convolutionX += 2)
    // {
    //   for (int convolutionY = 1; convolutionY < 8; convolutionY += 2)
    //   {
    //     float[] input = new float[] {
    //       GetPieceId(board, PositionToIndex(convolutionX - 1, convolutionY - 1)),
    //       GetPieceId(board, PositionToIndex(convolutionX, convolutionY - 1)),
    //       GetPieceId(board, PositionToIndex(convolutionX + 1, convolutionY - 1)),
    //       GetPieceId(board, PositionToIndex(convolutionX - 1, convolutionY)),
    //       GetPieceId(board, PositionToIndex(convolutionX, convolutionY)),
    //       GetPieceId(board, PositionToIndex(convolutionX + 1, convolutionY)),
    //       GetPieceId(board, PositionToIndex(convolutionX - 1, convolutionY + 1)),
    //       GetPieceId(board, PositionToIndex(convolutionX, convolutionY + 1)),
    //       GetPieceId(board, PositionToIndex(convolutionX + 1, convolutionY + 1)),
    //     };

    //     float[] convolutionHiddenLayer = Layer(input, 9, 16, 0, TanH);
    //     flattened[flatteneIndex] = Layer(convolutionHiddenLayer, 16, 1, 9 * 16 + 16, TanH)[0];

    //     flatteneIndex++;
    //   }
    // }

    board.UndoMove(move);

    // float[] hiddenLayer = Layer(flattened, 16, 16, 9 * 16 + 16 + 16 * 1 + 1, TanH);
    // float[] OutputLayer = Layer(flattened, 16, 1, 9 * 16 + 16 + 16 * 1 + 1 + 16 * 16 + 16, (x) => x);

    return output[0];
  }

  struct MoveChoice
  {
    public Move Move;
    public float Evaluation;
  }

  public Move Think(Board board, Timer timer)
  {
    if (Weights == null) Weights = new float[Trainer.WeightCount];

    List<Move> moves = new List<Move>(board.GetLegalMoves());
    List<MoveChoice> moveChoices = new List<MoveChoice>();

    foreach (Move move in moves)
    {
      moveChoices.Add(new MoveChoice()
      {
        Move = move,
        Evaluation = Inference(board, move)
      });
    }

    if (board.IsWhiteToMove)
    {
      moveChoices.Sort((a, b) => b.Evaluation.CompareTo(a.Evaluation));
    }
    else
    {
      moveChoices.Sort((a, b) => a.Evaluation.CompareTo(b.Evaluation));
    }

    foreach (MoveChoice moveChoice in moveChoices)
    {
      Console.WriteLine(moveChoice.Move + " " + moveChoice.Evaluation);
    }

    return moveChoices[0].Move;
  }
}