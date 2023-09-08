using System;
using System.IO;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters;

  public MyBot()
  {
    _parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_2.txt")[0..9857].Select(text =>
    {
      float raw = float.Parse(text);

      int compressed = (int)MathF.Floor(MathF.Max(MathF.Min((raw + 1.5f) / 3f, 1f), 0f) * 128f);

      float uncompressed = compressed / 128f * 3f - 1.5f;

      return uncompressed;
    }).ToArray();
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
    for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
    {
      for (int y = 0; y < imageHeight; y += 1)
      {
        for (int x = 0; x < imageWidth; x += 1)
        {
          float sum = _parameters[weightOffset + inputChannels * outputChannels * 3 * 3 + outputChannel];

          for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
          {
            for (int kernal = 0; kernal < 9; kernal++)
            {
              int kX = kernal % 3 - 1;
              int kY = kernal / 3 - 1;

              if (x + kX < 0 || y + kY < 0 || x + kX >= imageWidth || y + kY >= imageHeight) continue;

              float weight = _parameters[weightOffset + inputChannels * outputChannel * 3 * 3 + inputChannel * 3 * 3 + 3 * (kY + 1) + (kX + 1)];

              sum += input[inputChannel, y + kY, x + kX] * weight;
            }
          }

          output[outputChannel, y, x] = activationFunction(sum);
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