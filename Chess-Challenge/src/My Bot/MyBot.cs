using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  public float[] Weights;

  Dictionary<PieceType, int> _pieceIds = new Dictionary<PieceType, int>() {
    { PieceType.Pawn, 1 },
    { PieceType.Knight, 2 },
    { PieceType.Bishop, 3 },
    { PieceType.Rook, 5 },
    { PieceType.Queen, 9 },
    { PieceType.King, 100 },
    { PieceType.None, 0 }
  };

  float Sigmoid(float x)
  {
    return 1 / (1 + (float)Math.Exp(-x));
  }

  float[] Layer(float[] input, int previousLayerSize, int layerSize, int layerOffser)
  {
    float[] layer = new float[layerSize];

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * Weights[layerOffser + nodeIndex * previousLayerSize + weightIndex];
      }

      layer[nodeIndex] = Sigmoid(layer[nodeIndex]);
    }

    return layer;
  }

  /*
  Calculate the beginning material offset
  Check for checkmates
  Calculate the ending material offset
  generate tactical model input
  run tactical model
  concat info onto tactical model output
  run move model
  return move model input
  */

  float Inference(Board board, Move move)
  {
    //Calculate beginning material offset
    int materialOffset = 0;

    for (int squareIndex = 0; squareIndex < 64; squareIndex++)
    {
      Piece piece = board.GetPiece(new Square(squareIndex));

      if (piece.IsWhite == board.IsWhiteToMove)
      {
        materialOffset += _pieceIds[piece.PieceType];
      }
      else
      {
        materialOffset -= _pieceIds[piece.PieceType];
      }
    }

    board.MakeMove(move);

    // Check for checkmate
    if (board.IsInCheckmate())
    {
      board.UndoMove(move);

      return 9999999;
    }

    //Caclulate ending material offset
    int endMaterialOffset = 0;

    for (int squareIndex = 0; squareIndex < 64; squareIndex++)
    {
      Piece piece = board.GetPiece(new Square(squareIndex));

      if (piece.IsWhite == board.IsWhiteToMove)
      {
        endMaterialOffset += _pieceIds[piece.PieceType];
      }
      else
      {
        endMaterialOffset -= _pieceIds[piece.PieceType];
      }
    }

    float[] tacticalModelInput = new float[16];

    for (int squareIndex = 0; squareIndex < 64; squareIndex++)
    {
      Piece piece = board.GetPiece(new Square(squareIndex));
      int squareValue = (piece.IsWhite == board.IsWhiteToMove) ? _pieceIds[piece.PieceType] : -_pieceIds[piece.PieceType];

      int x = squareIndex % 8;
      int y = squareIndex / 8;

      tacticalModelInput[x * 4 + y] += squareValue;
    }

    board.UndoMove(move);

    float[] tacticalModelHidden1 = Layer(tacticalModelInput, 16, 16, 0);
    float[] tacticalModelHidden2 = Layer(tacticalModelInput, 16, 16, 16 * 16);
    float[] tacticalModelOutput = Layer(tacticalModelInput, 16, 3, 16 * 16 + 16 * 16);

    float[] moveModelInput = tacticalModelOutput.Concat(new float[] { move.StartSquare.File, move.StartSquare.Rank, move.TargetSquare.File, move.TargetSquare.Rank, (int)move.MovePieceType, materialOffset, endMaterialOffset }).ToArray();
    float[] moveModelHidden = Layer(moveModelInput, 10, 8, 16 * 16 + 16 * 16 + 16 * 3);
    float[] moveModelOutput = Layer(moveModelHidden, 8, 1, 16 * 16 + 16 * 16 + 16 * 3 + 10 * 8);

    return moveModelOutput[0];
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