using System;
using System.Collections.Generic;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  public float[] Weights;

  Dictionary<PieceType, int> _pieceIds = new Dictionary<PieceType, int>() {
    { PieceType.Pawn, 1 },
    { PieceType.Knight, 3 },
    { PieceType.Bishop, 4 },
    { PieceType.Rook, 5 },
    { PieceType.Queen, 9 },
    { PieceType.King, 10 },
    { PieceType.None, 0 }
  };

  // float Sigmoid(float x)
  // {
  //   return 1 / (1 + (float)Math.Exp(-x));
  // }

  float[] Layer(float[] input, int previousLayerSize, int layerSize, int layerOffser)
  {
    float[] layer = new float[layerSize];

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * Weights[layerOffser + nodeIndex * previousLayerSize + weightIndex];
      }

      // layer[nodeIndex] = Sigmoid(layer[nodeIndex]);
    }

    return layer;
  }

  float Inference(Board board, Move move)
  {
    float[] inputValues = new float[64];

    board.MakeMove(move);

    if (board.IsInCheckmate())
    {
      board.UndoMove(move);

      return 9999999;
    }

    for (int squareIndex = 0; squareIndex < inputValues.Length; squareIndex++)
    {
      Piece piece = board.GetPiece(new Square(squareIndex));
      bool isMyPiece = piece.IsWhite && !board.IsWhiteToMove;

      inputValues[squareIndex] = _pieceIds[piece.PieceType] * (isMyPiece ? 1 : -1);
    }

    board.UndoMove(move);

    float[] hiddenValues = Layer(inputValues, 64, 32, 0);

    float[] hiddenValues2 = Layer(hiddenValues, 32, 32, 64 * 32);

    return Layer(hiddenValues2, 32, 1, 64 * 32 + 32 * 32)[0];
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

    moveChoices.Sort((a, b) => b.Evaluation.CompareTo(a.Evaluation));

    return moveChoices[new System.Random().Next(0, Math.Min(moveChoices.Count, 1))].Move;
  }
}