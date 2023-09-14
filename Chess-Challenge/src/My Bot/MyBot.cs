using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
    var stopwatch = new System.Diagnostics.Stopwatch();
    stopwatch.Start();

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

    // foreach (float num in evaluationTensor) Console.Write(num + " ");//#DEBUG
    // Console.WriteLine(""); //#DEBUG

    float result = Layer(Layer(Layer(evaluationTensor, 36, 32), 32, 16), 16, 1)[0];

    stopwatch.Stop();
    Console.WriteLine($"Inference in {stopwatch.ElapsedMilliseconds}ms");
    Console.WriteLine($"Inference in {stopwatch.ElapsedTicks} ticks");

    return result;
  }

  float UpperConfidenceBound(Node node) => node.Value + 2 * MathF.Sqrt(MathF.Log(node.Parent.Visits) / node.Visits) * (_board.IsWhiteToMove ? 1 : -1);

  record class Node(Move Move, Node Parent)
  {
    public float Visits;
    public float Value;
    public Node[] Children;
  };

  Board _board;

  void Search(Node node)
  {
    while (node.Children != null && node.Children.Length > 0)
    {
      node = node.Children.MaxBy(UpperConfidenceBound);

      // Console.WriteLine($"Entering Node {node.Move}"); //#DEBUG

      _board.MakeMove(node.Move);
    }

    node.Children = _board.GetLegalMoves().Select(move => new Node(move, node)).ToArray();

    float value = Inference();

    // Console.WriteLine($"Value: {value}"); //#DEBUG

    while (node.Parent != null)
    {
      node.Parent.Value += value;
      node.Parent.Visits += 1;

      _board.UndoMove(node.Move);

      node = node.Parent;
    }
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    Node root = new Node(Move.NullMove, null);

    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 30f) Search(root);

    return root.Children.MaxBy(node => node.Visits).Move;
  }
}