using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters;

  public MyBot()
  {
    int pruned = 0;

    _parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_4.txt")[0..12865].Select(text =>//#DEBUG
    {
      float raw = float.Parse(text);//#DEBUG

      return raw;

      if (MathF.Abs(raw) < 0.04f)//#DEBUG
      {
        raw = 0f;//#DEBUG

        pruned++;//#DEBUG
      }

      int compressed = (int)MathF.Floor(MathF.Max(MathF.Min((raw + 1.5f) / 3f, 1f), 0f) * 128f); //#DEBUG

      float uncompressed = compressed / 128f * 3f - 1.5f;

      return uncompressed;
    }).ToArray();

    Console.WriteLine($"Pruned {pruned} weights"); //#DEBUG
  }

  int parameterOffset = 0;

  float[] Layer(float[] input, int previousLayerSize, int layerSize)
  {
    float[] layer = new float[layerSize];

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
    var tensor = new float[6 * 64];

    for (int i = 0; i < 64; i++)
    {
      int x = i % 8;
      int y = i / 8;

      Piece piece = _board.GetPiece(new Square(x, y));

      if (piece.PieceType == PieceType.None) continue;

      tensor[x * 8 * 6 + y * 6 + (int)piece.PieceType - 1] = piece.IsWhite ? 1 : -1;
    }

    parameterOffset = 0;

    return Layer(Layer(Layer(tensor, 6 * 64, 32), 32, 16), 16, 1)[0];
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