using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  static float[] Weights;

  public MyBot() // #DEBUG
  {
    string[] stringWeights = System.IO.File.ReadAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET.txt").Split('\n'); // #DEBUG

    Weights = stringWeights[..(stringWeights.Length - 1)].Select(float.Parse).ToArray(); // #DEBUG

    Console.WriteLine("Weights: " + Weights.Length); // #DEBUG
  } // #DEBUG

  static float TanH(float x) => (float)Math.Tanh(x);

  static float[,,] Convolution(float[,,] input, int inputChannels, int outputChannels, ref int weightOffset, Func<float, float> activationFunction)
  {
    int imageHeight = input.GetLength(1);
    int imageWidth = input.GetLength(2);

    float[,,] output = new float[outputChannels, imageHeight, imageWidth];

    for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
    {
      for (int y = 0; y < imageHeight; y += 1)
      {
        for (int x = 0; x < imageWidth; x += 1)
        {
          float bias = Weights[weightOffset + inputChannels * outputChannels * 3 * 3 + outputChannel];

          float sum = 0;

          for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
          {
            float kernalValue = 0;
            for (int kernalY = -1; kernalY <= 1; kernalY++)
            {
              for (int kernalX = -1; kernalX <= 1; kernalX++)
              {
                float pixelValue = 0;

                try
                {
                  if (x + kernalX >= 0 && y + kernalY >= 0 && x + kernalX < imageWidth && y + kernalY < imageHeight) pixelValue = input[inputChannel, y + kernalY, x + kernalX];
                }
                catch
                {
                  Console.WriteLine("Error reading input at " + inputChannel + " " + (y + kernalY) + " " + (x + kernalX) + " from dimensions " + input.GetLength(0) + " " + input.GetLength(1) + " " + input.GetLength(2));
                }

                float weight = Weights[weightOffset + inputChannels * outputChannel * 3 * 3 + inputChannel * 3 * 3 + 3 * (kernalY + 1) + (kernalX + 1)];
                kernalValue += pixelValue * weight;
              }
            }
            sum += kernalValue;
          }
          output[outputChannel, y, x] = activationFunction(bias + sum);
        }
      }
    }

    weightOffset += inputChannels * outputChannels * 3 * 3 + outputChannels;

    return output;
  }

  static float[] Layer(float[] input, int previousLayerSize, int layerSize, ref int layerOffset, Func<float, float> activationFunction)
  {
    float[] layer = new float[layerSize];
    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        layer[nodeIndex] += input[weightIndex] * Weights[layerOffset + nodeIndex * previousLayerSize + weightIndex];
      }
      layer[nodeIndex] = activationFunction(layer[nodeIndex] + Weights[layerOffset + layerSize * previousLayerSize + nodeIndex]);
    }
    layerOffset += layerSize * previousLayerSize + layerSize;
    return layer;
  }

  Board _board;

  class Node
  {
    public int Score;
    public int Depth;
    public Move Move;
    public Node[] ChildNodes;

    MyBot _bot;

    public Node(Move move, MyBot bot)
    {
      Move = move;
      _bot = bot;
    }

    public void Expand()
    {
      _bot._board.MakeMove(Move);

      if (ChildNodes != null)
      {
        ChildNodes.MinBy(node => node.Depth).Expand();

        Depth = ChildNodes.Min(node => node.Depth) + 1;
      }
      else
      {
        ChildNodes = _bot._board.GetLegalMoves().Select(move => new Node(move, _bot)).ToArray();

        foreach (Node node in ChildNodes) node.Simulate(3);

        Depth = 1;
      }

      if (ChildNodes.Length == 0)
      {
        Depth = 999999999;

        _bot._board.UndoMove(Move);

        return;
      }

      Score = ChildNodes.Min(node => -node.Score);

      _bot._board.UndoMove(Move);
    }

    public void Simulate(int moves)
    {
      _bot._board.MakeMove(Move);

      var simulatedMoves = new List<Move>();

      for (int i = 0; i < moves; i++)
      {
        Move[] legalMoves = _bot._board.GetLegalMoves();

        if (legalMoves.Length == 0) break;

        Move nextMove = legalMoves[0];

        nextMove = legalMoves.MaxBy(Evaluate);

        simulatedMoves.Add(nextMove);

        _bot._board.MakeMove(nextMove);
      }

      Score = Evaluate() * (moves % 2 == 0 ? -1 : 1);

      simulatedMoves.Reverse();

      foreach (Move move in simulatedMoves) _bot._board.UndoMove(move);

      _bot._board.UndoMove(Move);
    }

    public int Evaluate()
    {
      if (_bot._board.IsInCheckmate()) return -999999999;

      float[,,] board = new float[6, 8, 8];

      for (int typeIndex = 0; typeIndex < 6; typeIndex++)
      {
        foreach (Piece piece in _bot._board.GetPieceList((PieceType)typeIndex + 1, _bot._board.IsWhiteToMove)) board[typeIndex, piece.Square.File, piece.Square.Rank] = -1;
        foreach (Piece piece in _bot._board.GetPieceList((PieceType)typeIndex + 1, !_bot._board.IsWhiteToMove)) board[typeIndex, piece.Square.File, piece.Square.Rank] = 1;
      }

      int parameterOffset = 0;

      float[,,] convolution1 = Convolution(board, 6, 32, ref parameterOffset, TanH);
      float[,,] convolution2 = Convolution(convolution1, 32, 8, ref parameterOffset, TanH);
      float[,,] convolution3 = Convolution(convolution2, 8, 1, ref parameterOffset, TanH);

      float[] input = new float[64];
      for (int i = 0; i < 64; i++) input[i] = convolution3[0, i / 8, i % 8];

      float[] layer1 = Layer(input, 64, 128, ref parameterOffset, TanH);
      float[] layer2 = Layer(layer1, 128, 64, ref parameterOffset, TanH);
      float[] layer3 = Layer(layer2, 64, 16, ref parameterOffset, TanH);

      return (int)(Layer(layer3, 16, 1, ref parameterOffset, x => x)[0] * 100f);
    }

    public int Evaluate(Move move)
    {
      _bot._board.MakeMove(move);

      int score = 0;

      _bot._board.UndoMove(move);

      return score;
    }

    public void Debug(int maxDepth = 2, int depth = 0) // #DEBUG
    {
      if (depth > maxDepth) return; // #DEBUG

      Console.WriteLine(new string('\t', depth) + "Node with " + Move.ToString() + " has score " + Score + " and depth " + Depth); // #DEBUG

      if (ChildNodes != null) foreach (Node node in ChildNodes) node.Debug(maxDepth, depth + 1); // #DEBUG
    } // #DEBUG
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    Node rootNode = new Node(Move.NullMove, this);

    // Console.WriteLine(rootNode.Evaluate() / 100f);

    // return Move.NullMove;

    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60) rootNode.Expand();

    // rootNode.Debug(1); // #DEBUG

    return rootNode.ChildNodes.MaxBy(node => node.Score).Move;
  }
}