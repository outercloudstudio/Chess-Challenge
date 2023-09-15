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

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 1000 };

  float Inference()
  {
    if (_board.IsInCheckmate()) return -100000 * WhiteToMoveFactor;

    int evaluation = 0;

    for (int type = 1; type < 7; type++)
    {
      evaluation += _board.GetPieceList((PieceType)type, true).Count * pieceValues[type];
      evaluation -= _board.GetPieceList((PieceType)type, false).Count * pieceValues[type];
    }

    return evaluation;

    // var evaluationTensor = new float[36];

    // for (int x = 0; x < 6; x++)
    // {
    //   for (int y = 0; y < 6; y++)
    //   {
    //     var sightTensor = new List<float>();

    //     for (int kernelX = 0; kernelX < 3; kernelX++)
    //     {
    //       for (int kernelY = 0; kernelY < 3; kernelY++)
    //       {
    //         var pieceTensor = new float[6];

    //         Piece piece = _board.GetPiece(new Square(x + kernelX, y + kernelY));

    //         if (piece.PieceType != PieceType.None) pieceTensor[(int)piece.PieceType - 1] = piece.IsWhite ? 1 : -1;

    //         sightTensor.AddRange(pieceTensor);
    //       }
    //     }

    //     parameterOffset = 0;

    //     evaluationTensor[x * 6 + y] = Layer(Layer(Layer(sightTensor.ToArray(), 6 * 9, 16), 16, 16), 16, 1)[0];
    //   }
    // }

    // return Layer(Layer(Layer(evaluationTensor, 36, 32), 32, 16), 16, 1)[0];
  }

  Board _board;
  Timer _timer;
  Move _bestMove;

  int _nodes; //#DEBUG

  int[] MoveScores = new int[218];

  int WhiteToMoveFactor => _board.IsWhiteToMove ? 1 : -1;

  float Search(int ply, int depth, float alpha, float beta)
  {
    _nodes++; //#DEBUG

    bool qSearch = depth <= 0;

    if (qSearch)
    {
      alpha = MathF.Max(alpha, Inference() * WhiteToMoveFactor);

      if (alpha >= beta) return alpha;
    }

    bool isCheck = _board.IsInCheck();

    Span<Move> moves = stackalloc Move[218];
    _board.GetLegalMovesNonAlloc(ref moves, qSearch && !isCheck);

    if (qSearch && moves.Length == 0) return Inference() * WhiteToMoveFactor;

    int index = 0;

    // Scores are sorted low to high
    foreach (Move move in moves)
    {
      MoveScores[index++] = move.IsCapture
        ? (int)move.MovePieceType - 100 * (int)move.CapturePieceType
        : (int)(new Random().NextDouble() * 100);
    }

    MoveScores.AsSpan(0, moves.Length).Sort(moves);

    index = 0;

    foreach (Move move in moves)
    {
      if (outOfTime && ply > 1) return -100000f;

      _board.MakeMove(move);

      float score = -Search(ply + 1, depth - 1, -beta, -alpha);

      _board.UndoMove(move);

      if (score > alpha)
      {
        if (ply == 0) _bestMove = move;

        alpha = score;

        if (score >= beta) break;
      }
    }

    return alpha;
  }

  bool outOfTime => _timer.MillisecondsElapsedThisTurn >= _timer.MillisecondsRemaining / 60f;

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    _nodes = 0; //#DEBUG

    int depth = 1;

    Move lastBestMove = Move.NullMove;

    while (true)
    {
      Search(0, depth++, -100000f, 100000f);

      if (lastBestMove == Move.NullMove) lastBestMove = _bestMove;

      if (outOfTime) break;

      lastBestMove = _bestMove;
    }

    Console.WriteLine($"Nodes per second {_nodes / (timer.MillisecondsElapsedThisTurn / 1000f + 0.00001f)} Depth: {depth} Seconds {timer.MillisecondsElapsedThisTurn / 1000f}"); //#DEBUG

    return lastBestMove;
  }
}