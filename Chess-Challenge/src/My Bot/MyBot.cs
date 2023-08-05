using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  // static float[] Weights;

  // public MyBot() // #DEBUG
  // {
  //   string[] stringWeights = System.IO.File.ReadAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET.txt").Split('\n'); // #DEBUG

  //   Weights = stringWeights[..(stringWeights.Length - 1)].Select(float.Parse).ToArray(); // #DEBUG

  //   Console.WriteLine("Weights: " + Weights.Length); // #DEBUG
  // } // #DEBUG

  // static float TanH(float x) => (float)Math.Tanh(x);

  // static float[,,] Convolution(float[,,] input, int inputChannels, int outputChannels, ref int weightOffset, Func<float, float> activationFunction)
  // {
  //   int imageHeight = input.GetLength(1);
  //   int imageWidth = input.GetLength(2);

  //   float[,,] output = new float[outputChannels, imageHeight, imageWidth];

  //   for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
  //   {
  //     for (int y = 0; y < imageHeight; y += 1)
  //     {
  //       for (int x = 0; x < imageWidth; x += 1)
  //       {
  //         float bias = Weights[weightOffset + inputChannels * outputChannels * 3 * 3 + outputChannel];

  //         float sum = 0;

  //         for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
  //         {
  //           float kernalValue = 0;
  //           for (int kernalY = -1; kernalY <= 1; kernalY++)
  //           {
  //             for (int kernalX = -1; kernalX <= 1; kernalX++)
  //             {
  //               float pixelValue = 0;

  //               try
  //               {
  //                 if (x + kernalX >= 0 && y + kernalY >= 0 && x + kernalX < imageWidth && y + kernalY < imageHeight) pixelValue = input[inputChannel, y + kernalY, x + kernalX];
  //               }
  //               catch
  //               {
  //                 Console.WriteLine("Error reading input at " + inputChannel + " " + (y + kernalY) + " " + (x + kernalX) + " from dimensions " + input.GetLength(0) + " " + input.GetLength(1) + " " + input.GetLength(2));
  //               }

  //               float weight = Weights[weightOffset + inputChannels * outputChannel * 3 * 3 + inputChannel * 3 * 3 + 3 * (kernalY + 1) + (kernalX + 1)];
  //               kernalValue += pixelValue * weight;
  //             }
  //           }
  //           sum += kernalValue;
  //         }
  //         output[outputChannel, y, x] = activationFunction(bias + sum);
  //       }
  //     }
  //   }

  //   weightOffset += inputChannels * outputChannels * 3 * 3 + outputChannels;

  //   return output;
  // }

  // static float[] Layer(float[] input, int previousLayerSize, int layerSize, ref int layerOffset, Func<float, float> activationFunction)
  // {
  //   float[] layer = new float[layerSize];
  //   for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
  //   {
  //     for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
  //     {
  //       layer[nodeIndex] += input[weightIndex] * Weights[layerOffset + nodeIndex * previousLayerSize + weightIndex];
  //     }
  //     layer[nodeIndex] = activationFunction(layer[nodeIndex] + Weights[layerOffset + layerSize * previousLayerSize + nodeIndex]);
  //   }
  //   layerOffset += layerSize * previousLayerSize + layerSize;
  //   return layer;
  // }

  class Node
  {
    public Move Move;
    public Node[] Children;
    public float Score;
    public float Visits;

    MyBot _bot;

    public float Confidence(Node child) => (child.Score / child.Visits) + 2f * MathF.Sqrt(MathF.Log(Visits) / child.Visits);

    public Node(Move move, MyBot bot)
    {
      Move = move;
      _bot = bot;
    }

    public void Search()
    {
      // Console.WriteLine("Searching " + Move);

      Visits++;

      _bot._board.MakeMove(Move);

      if (Children != null)
      {
        Node pickedNode = Children.MaxBy(node => Confidence(node));

        pickedNode.Search();

        Score = Children.Average(node => node.Score);

        _bot._board.UndoMove(Move);

        return;
      }

      Children = _bot._board.GetLegalMoves().Select(move => new Node(move, _bot)).ToArray();

      foreach (Node node in Children) node.Rollout();

      Score = Children.Average(node => node.Score);

      _bot._board.UndoMove(Move);
    }

    public void Rollout()
    {
      // Console.WriteLine("Rolling out " + Move);

      Visits++;

      _bot._board.MakeMove(Move);

      var moves = new List<Move>();
      for (int depth = 0; depth < 3; depth++)
      {
        var legalMoves = _bot._board.GetLegalMoves();

        if (legalMoves.Length == 0) break;

        Move move = legalMoves[new System.Random().Next(legalMoves.Length)];

        moves.Add(move);
        _bot._board.MakeMove(move);
      }

      Score = 0.5f;
      float[] pieceValues = { 0, 0.0128f, 0.0384f, 0.0384f, 0.0641f, 0.1153f };

      for (int typeIndex = 1; typeIndex < 6; typeIndex++)
      {
        Score += _bot._board.GetPieceList((PieceType)typeIndex, _bot._white).Count * pieceValues[typeIndex];
        Score -= _bot._board.GetPieceList((PieceType)typeIndex, !_bot._white).Count * pieceValues[typeIndex];
      }

      if (_bot._board.IsInCheckmate()) Score = (_bot._board.IsWhiteToMove == _bot._white) ? 0f : 1f;
      if (_bot._board.IsDraw()) Score = 0.5f;

      // Console.WriteLine("Rollout score " + Score);

      moves.Reverse();

      foreach (Move move in moves) _bot._board.UndoMove(move);

      _bot._board.UndoMove(Move);
    }
  }

  Board _board;
  bool _white;

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _white = board.IsWhiteToMove;

    Node root = new Node(Move.NullMove, this);

    while (root.Children == null || timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60) root.Search();

    foreach (Node child in root.Children) Console.WriteLine(child.Move + " " + child.Score + " " + child.Visits + " " + root.Confidence(child)); // #DEBUG

    return root.Children.MaxBy(node => node.Score).Move;
  }
}