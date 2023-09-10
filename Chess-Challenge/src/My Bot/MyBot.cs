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

    _parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_3.txt")[0..16081].Select(text =>//#DEBUG
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

  int weightOffset = 0;

  float[,,] Convolution(float[,,] input, int inputChannels, int outputChannels)
  {
    var output = new float[outputChannels, 8, 8];

    for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
    {
      for (int i = 0; i < 8 * 8; i += 1)
      {
        int x = i % 8;
        int y = i / 8;

        float sum = _parameters[weightOffset + inputChannels * outputChannels * 9 + outputChannel];

        for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
        {
          for (int kernal = 0; kernal < 9; kernal++)
          {
            int kX = kernal % 3;
            int kY = kernal / 3;
            int aX = x + kX - 1;
            int aY = y + kY - 1;

            if (aX < 0 || aX > 7 || aY < 0 || aY > 7) continue;

            float weight = _parameters[weightOffset + inputChannels * outputChannel * 9 + inputChannel * 9 + 3 * kY + kX];

            sum += input[inputChannel, aY, aX] * weight;
          }
        }

        output[outputChannel, y, x] = MathF.Tanh(sum);
      }
    }

    weightOffset += inputChannels * outputChannels * 9 + outputChannels;

    return output;
  }

  float Inference()
  {
    var tensor = new float[1, 8, 8];

    for (int i = 0; i < 64; i++)
    {
      int x = i % 8;
      int y = i / 8;

      Piece piece = _board.GetPiece(new Square(x, y));

      tensor[0, x, y] = (int)piece.PieceType * (piece.IsWhite ? 1 : -1);
    }

    weightOffset = 0;

    var output = Convolution(Convolution(Convolution(Convolution(Convolution(tensor, 1, 24), 24, 24), 24, 24), 24, 24), 24, 1);

    float result = 0;

    for (int i = 0; i < 64; i++) result += output[0, i % 8, i / 8] / 64f;

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

      Console.WriteLine($"Entering Node {node.Move}"); //#DEBUG

      _board.MakeMove(node.Move);
    }

    node.Children = _board.GetLegalMoves().Select(move => new Node(move, node)).ToArray();

    float value = Inference();

    Console.WriteLine($"Value: {value}"); //#DEBUG

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

    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60f) Search(root);

    return root.Children.MaxBy(node => node.Visits).Move;
  }
}