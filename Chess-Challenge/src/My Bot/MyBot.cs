using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class ARCNET : IChessBot
{
  public float[] Weights;

  public ARCNET()
  {
    string[] stringWeights = System.IO.File.ReadAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 7.txt").Split('\n');

    Weights = stringWeights[..(stringWeights.Length - 1)].Select(float.Parse).ToArray();

    Console.WriteLine("ARCNET loaded! Weights: " + Weights.Length);
  }

  float Weight(int index)
  {
    return Weights[index];
  }

  float Sigmoid(float x)
  {
    return (float)(1 / (1 + Math.Exp(-x)));
  }

  float[] Layer(float[] input, int previousLayerSize, int layerSize, ref int layerOffset, Func<float, float> activationFunction)
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

    layerOffset += layerSize * previousLayerSize + layerSize;

    return layer;
  }

  float GetPieceId(Board board, int index)
  {
    if (index >= 64) return 0;

    Piece piece = board.GetPiece(new Square(index));

    return (float)((int)piece.PieceType * (piece.IsWhite ? 1 : -1));
  }

  List<int> _PieceValues = new List<int>() { 0, 1, 3, 3, 5, 9, 0 };

  int GetMaterial(ChessChallenge.API.Board board, bool white)
  {
    int material = 0;

    for (int squareIndex = 0; squareIndex < 64; squareIndex++)
    {
      ChessChallenge.API.Square square = new ChessChallenge.API.Square(squareIndex);
      ChessChallenge.API.Piece piece = board.GetPiece(square);

      if (piece.IsWhite == white) material += _PieceValues[(int)piece.PieceType];
    }

    return material;
  }

  float Evaluate(Board board, Move move)
  {
    float whiteMaterial = GetMaterial(board, true);
    float blackMaterial = GetMaterial(board, true);

    board.MakeMove(move);

    if (board.IsInCheckmate())
    {
      board.UndoMove(move);

      return board.IsWhiteToMove ? 9999 : -9999;
    }

    if (board.IsDraw())
    {
      board.UndoMove(move);

      return 0;
    }

    float[] input = new float[] {
      whiteMaterial,
      blackMaterial,
      GetMaterial(board, true),
      GetMaterial(board, false),
      board.GameMoveHistory.Length,
      board.GetKingSquare(true).File,
      board.GetKingSquare(true).Rank,
      board.GetKingSquare(false).File,
      board.GetKingSquare(false).Rank,
      move.IsCapture ? 1 : 0,
      board.IsDraw() ? 1 : 0,
      board.IsInCheck() ? 1 : 0,
      _PieceValues[(int)move.MovePieceType],
      move.StartSquare.File,
      move.StartSquare.Rank,
      move.TargetSquare.File,
      move.TargetSquare.Rank
    };

    board.UndoMove(move);

    int weightOffset = 0;

    float[] hiddenLayer1 = Layer(input, 17, 32, ref weightOffset, Sigmoid);
    float[] hiddenLayer2 = Layer(hiddenLayer1, 32, 16, ref weightOffset, Sigmoid);
    float[] hiddenLayer3 = Layer(hiddenLayer2, 16, 32, ref weightOffset, Sigmoid);
    float[] output = Layer(hiddenLayer3, 32, 2, ref weightOffset, (x) => x);

    return output[0] - output[1];
  }

  struct MoveChoice
  {
    public Move Move;
    public float Evaluation;
  }

  public Move Think(Board board, Timer timer)
  {
    List<Move> moves = new List<Move>(board.GetLegalMoves());
    List<MoveChoice> moveChoices = new List<MoveChoice>();

    foreach (Move move in moves)
    {
      moveChoices.Add(new MoveChoice()
      {
        Move = move,
        Evaluation = Evaluate(board, move)
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

    Console.WriteLine("Current evaluation: " + moveChoices[0].Evaluation);

    return moveChoices[0].Move;
  }
}