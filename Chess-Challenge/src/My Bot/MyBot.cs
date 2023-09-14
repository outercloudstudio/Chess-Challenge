using System;
using System.Collections.Generic;
using System.IO; //#DEBUG
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters;

  public MyBot()
  {
    // int pruned = 0;

    _parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_5.txt")[0..2898].Select(text =>//#DEBUG
    {
      float raw = float.Parse(text);//#DEBUG

      return raw;

      // if (MathF.Abs(raw) < 0.04f)//#DEBUG
      // {
      //   raw = 0f;//#DEBUG

      //   pruned++;//#DEBUG
      // }

      // int compressed = (int)MathF.Floor(MathF.Max(MathF.Min((raw + 1.5f) / 3f, 1f), 0f) * 128f); //#DEBUG

      // float uncompressed = compressed / 128f * 3f - 1.5f;

      // return uncompressed;
    }).ToArray();

    // Console.WriteLine($"Pruned {pruned} weights"); //#DEBUG
  }

  int parameterOffset = 0;

  float[] Layer(float[] input, int previousLayerSize, int layerSize)
  {
    var layer = new float[layerSize];

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * _parameters[parameterOffset + nodeIndex * previousLayerSize + weightIndex];
      }

      layer[nodeIndex] = MathF.Tanh(layer[nodeIndex] + _parameters[parameterOffset + layerSize * previousLayerSize + nodeIndex]);
    }

    parameterOffset += layerSize * previousLayerSize + layerSize;

    return layer;
  }

  float Inference()
  {
    var evaluationTensor = new float[36];

    for (int x = 0; x < 6; x++)
    {
      for (int y = 0; y < 6; y++)
      {
        var sightTensor = new List<float>();

        for (int kernelX = 0; kernelX < 3; kernelX++)
        {
          for (int kernelY = 0; kernelY < 3; kernelY++)
          {
            var pieceTensor = new float[6];

            Piece piece = _board.GetPiece(new Square(x + kernelX, y + kernelY));

            if (piece.PieceType != PieceType.None) pieceTensor[(int)piece.PieceType - 1] = piece.IsWhite ? 1 : -1;

            sightTensor.AddRange(pieceTensor);
          }
        }

        parameterOffset = 0;

        evaluationTensor[x * 6 + y] = Layer(Layer(Layer(sightTensor.ToArray(), 6 * 9, 16), 16, 16), 16, 1)[0];
      }
    }

    return Layer(Layer(Layer(evaluationTensor, 36, 32), 32, 16), 16, 1)[0];
  }

  Board _board;
  Move _bestMove;
  int _nodes; //#DEBUG

  float Search(int ply, int depth, float alpha, float beta)
  {
    _nodes++; //#DEBUG

    if (depth == 0) return Inference() * (_board.IsWhiteToMove ? 1 : -1);

    var moves = _board.GetLegalMoves();

    foreach (Move move in moves)
    {
      _board.MakeMove(move);

      // Console.WriteLine(new string('\t', ply) + $"Searching {move}"); //#DEBUG

      float score = -Search(ply + 1, depth - 1, -beta, -alpha);

      // Console.WriteLine(new string('\t', ply) + score); //#DEBUG

      _board.UndoMove(move);

      if (score > alpha)
      {
        if (ply == 0) _bestMove = move;

        alpha = score;
      }

      if (score >= beta) return beta;
    }

    return alpha;
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    _nodes = 0; //#DEBUG

    Search(0, 5, -100000f, 100000f);

    Console.WriteLine($"Nodes per second {_nodes / (timer.MillisecondsElapsedThisTurn / 1000f + 0.00001f)}"); //#DEBUG

    return _bestMove;
  }
}