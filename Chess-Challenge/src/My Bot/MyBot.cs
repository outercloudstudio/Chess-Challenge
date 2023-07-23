using System;
using System.Collections.Generic;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  public float[] Weights;

  bool IsWhite;

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

    return (float)((int)piece.PieceType * ((piece.IsWhite == IsWhite) ? 1 : -1)) / 6;
  }

  int PositionToIndex(int x, int y)
  {
    return new Square(x, y).Index;
  }

  float Inference(Board board, Move move)
  {
    board.MakeMove(move);

    // Check for checkmate
    if (board.IsInCheckmate())
    {
      board.UndoMove(move);

      return 9999999;
    }

    float[] flattened = new float[16];

    int flatteneIndex = 0;
    for (int convolutionX = 1; convolutionX < 8; convolutionX += 2)
    {
      for (int convolutionY = 1; convolutionY < 8; convolutionY += 2)
      {
        float[] input = new float[] {
          GetPieceId(board, PositionToIndex(convolutionX - 1, convolutionY - 1)),
          GetPieceId(board, PositionToIndex(convolutionX, convolutionY - 1)),
          GetPieceId(board, PositionToIndex(convolutionX + 1, convolutionY - 1)),
          GetPieceId(board, PositionToIndex(convolutionX - 1, convolutionY)),
          GetPieceId(board, PositionToIndex(convolutionX, convolutionY)),
          GetPieceId(board, PositionToIndex(convolutionX + 1, convolutionY)),
          GetPieceId(board, PositionToIndex(convolutionX - 1, convolutionY + 1)),
          GetPieceId(board, PositionToIndex(convolutionX, convolutionY + 1)),
          GetPieceId(board, PositionToIndex(convolutionX + 1, convolutionY + 1)),
        };

        float[] convolutionHiddenLayer = Layer(input, 9, 16, 0, TanH);
        flattened[flatteneIndex] = Layer(convolutionHiddenLayer, 16, 1, 9 * 16 + 16, TanH)[0];

        flatteneIndex++;
      }
    }

    board.UndoMove(move);

    float[] hiddenLayer = Layer(flattened, 16, 16, 9 * 16 + 16 + 16 * 1 + 1, TanH);
    float[] OutputLayer = Layer(flattened, 16, 1, 9 * 16 + 16 + 16 * 1 + 1 + 16 * 16 + 16, (x) => x);

    return OutputLayer[0];
  }

  struct MoveChoice
  {
    public Move Move;
    public float Evaluation;
  }

  public Move Think(Board board, Timer timer)
  {
    IsWhite = board.IsWhiteToMove;

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

    moveChoices.Sort((a, b) => b.Evaluation.CompareTo(a.Evaluation));

    Console.WriteLine("Evaluations: ");

    foreach (MoveChoice choice in moveChoices)
    {
      Console.WriteLine(choice.Move + " " + choice.Evaluation);
    }

    return moveChoices[0].Move;
  }
}