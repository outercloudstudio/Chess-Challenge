using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class ARCNET : IChessBot
{
  public float[] Weights;

  bool _isWhite;

  public ARCNET()
  {
    string[] stringWeights = System.IO.File.ReadAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Models\\ARCNET 5.txt").Split('\n');

    Weights = stringWeights[..(stringWeights.Length - 1)].Select(float.Parse).ToArray();

    Console.WriteLine("Weights: " + Weights.Length);
  }

  float Weight(int index)
  {
    return Weights[index];
  }

  float TanH(float x)
  {
    return (float)Math.Tanh(x);
  }

  float ReLU(float x)
  {
    return Math.Max(0, x);
  }

  float[,,] Convolution(float[,,] input, int inputChannels, int outputChannels, ref int weightOffset, Func<float, float> activationFunction)
  {
    int imageHeight = input.GetLength(1);
    int imageWidth = input.GetLength(2);

    float[,,] output = new float[outputChannels, imageHeight, imageWidth];

    int channels = input.GetLength(0);

    for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
    {
      for (int y = 0; y < imageHeight; y += 1)
      {
        for (int x = 0; x < imageWidth; x += 1)
        {
          float bias = Weight(weightOffset + inputChannels * outputChannels * 3 * 3 + outputChannel);
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

                float weight = Weight(weightOffset + inputChannels * outputChannel * 3 * 3 + inputChannel * 3 * 3 + 3 * (kernalY + 1) + (kernalX + 1));

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

  float[,,] Downscale(float[,,] input)
  {
    float[,,] output = new float[input.GetLength(0), input.GetLength(1) / 2, input.GetLength(2) / 2];

    for (int channels = 0; channels < input.GetLength(0); channels++)
    {
      for (int y = 0; y < input.GetLength(1); y += 2)
      {
        for (int x = 0; x < input.GetLength(2); x += 2)
        {
          output[channels, y / 2, x / 2] = (input[channels, y, x] + input[channels, y + 1, x] + input[channels, y, x + 1] + input[channels, y + 1, x + 1]) / 4;
        }
      }
    }

    return output;
  }

  float[] Flatten(float[,,] input)
  {
    float[] output = new float[input.GetLength(0) * input.GetLength(1) * input.GetLength(2)];

    for (int channels = 0; channels < input.GetLength(0); channels++)
    {
      for (int y = 0; y < input.GetLength(1); y++)
      {
        for (int x = 0; x < input.GetLength(2); x++)
        {
          output[channels * input.GetLength(1) * input.GetLength(2) + y * input.GetLength(2) + x] = input[channels, y, x];
        }
      }
    }

    return output;
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

  int PositionToIndex(int x, int y)
  {
    return new Square(x, y).Index;
  }

  float Evaluate(Board board, Move move)
  {
    board.MakeMove(move);

    board.UndoMove(move);

    return 0;

    // if (board.IsInCheckmate()) return 9999;

    // float[,,] input = new float[1, 8, 8];

    // for (int x = 0; x < 8; x++)
    // {
    //   for (int y = 0; y < 8; y++)
    //   {
    //     input[0, 7 - y, 7 - x] = GetPieceId(board, PositionToIndex(x, y));
    //   }
    // }

    // int weightOffset = 0;

    // float[,,] convolutionLayer1 = Convolution(input, 1, 16, ref weightOffset, ReLU);
    // float[,,] convolutionLayer2 = Convolution(convolutionLayer1, 16, 8, ref weightOffset, ReLU);
    // float[,,] convolutionLayer3 = Convolution(convolutionLayer2, 8, 4, ref weightOffset, ReLU);

    // float[,,] downscaleLayer = Downscale(convolutionLayer3);

    // float[] flattenLayer = Flatten(downscaleLayer);

    // float[] hiddenLayer1 = Layer(flattenLayer, 64, 128, ref weightOffset, ReLU);
    // float[] hiddenLayer2 = Layer(hiddenLayer1, 128, 64, ref weightOffset, ReLU);
    // float[] hiddenLayer3 = Layer(hiddenLayer2, 64, 32, ref weightOffset, ReLU);
    // float[] output = Layer(hiddenLayer3, 32, 1, ref weightOffset, (x) => x);

    // return output[0];
  }

  struct MoveChoice
  {
    public Move Move;
    public float Evaluation;
  }

  public Move Think(Board board, Timer timer)
  {
    _isWhite = board.IsWhiteToMove;

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

    return moveChoices[0].Move;
  }
}