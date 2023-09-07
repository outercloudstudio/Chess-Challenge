using System;
using System.IO;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters;

  public MyBot()
  {
    _parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_2.txt")[0..9857].Select(float.Parse).ToArray();
  }

  float[,,] BoardToTensor(Board board)
  {
    float[,,] tensor = new float[1, 8, 8];

    for (int x = 0; x < 8; x++)
    {
      for (int y = 0; y < 8; y++)
      {
        Piece piece = board.GetPiece(new Square(x, y));

        tensor[0, x, y] = (int)piece.PieceType * (piece.IsWhite ? 1 : -1);
      }
    }

    return tensor;
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
          float bias = _parameters[weightOffset + inputChannels * outputChannels * 3 * 3 + outputChannel];
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
                float weight = _parameters[weightOffset + inputChannels * outputChannel * 3 * 3 + inputChannel * 3 * 3 + 3 * (kernalY + 1) + (kernalX + 1)];
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

  float Inference(Board board)
  {
    int weight = 0;

    float[,,] tensor = BoardToTensor(board);

    float[,,] layer1 = Convolution(tensor, 1, 32, ref weight, (x) => MathF.Tanh(x));
    float[,,] layer2 = Convolution(layer1, 32, 32, ref weight, (x) => MathF.Tanh(x));
    float[,,] output = Convolution(layer2, 32, 1, ref weight, (x) => MathF.Tanh(x));

    float result = 0;
    int resultTotal = 0;

    for (int x = 0; x < 8; x++)
    {
      for (int y = 0; y < 8; y++)
      {
        result += output[0, x, y];
        resultTotal++;
      }
    }

    result /= resultTotal;

    return result;
  }

  public Move Think(Board board, Timer timer)
  {
    return board.GetLegalMoves().MaxBy(move =>
    {
      board.MakeMove(move);

      float result = Inference(board);

      board.UndoMove(move);

      return result;
    });
  }
}