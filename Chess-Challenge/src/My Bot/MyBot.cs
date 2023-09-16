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

    _parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_6.txt")[0..2930].Select(text =>//#DEBUG
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

  float[] _layerInput = new float[54];
  float[] _layerOutput = new float[32];
  float[] _evaluationTensor = new float[37];
  float[] _sightTensor = new float[54];
  float[] _emptyTensor = new float[54];

  void Layer(int previousLayerSize, int layerSize)
  {
    Array.Copy(_emptyTensor, _layerOutput, 32);

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        _layerOutput[nodeIndex] += _layerInput[weightIndex] * _parameters[parameterOffset + nodeIndex * previousLayerSize + weightIndex];
      }

      _layerOutput[nodeIndex] = MathF.Tanh(_layerOutput[nodeIndex] + _parameters[parameterOffset + layerSize * previousLayerSize + nodeIndex]);
    }

    parameterOffset += layerSize * previousLayerSize + layerSize;

    Array.Copy(_layerOutput, _layerInput, layerSize);
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

    for (int x = 0; x < 6; x++)
    {
      for (int y = 0; y < 6; y++)
      {
        Array.Copy(_emptyTensor, _sightTensor, 54);

        for (int kernelX = 0; kernelX < 3; kernelX++)
        {
          for (int kernelY = 0; kernelY < 3; kernelY++)
          {
            Piece piece = _board.GetPiece(new Square(x + kernelX, y + kernelY));

            if (piece.PieceType != PieceType.None) _sightTensor[kernelX * 18 + kernelY * 6 + (int)piece.PieceType - 1] = piece.IsWhite ? 1 : -1;
          }
        }

        parameterOffset = 0;

        Array.Copy(_sightTensor, _layerInput, 54);

        Layer(54, 16);
        Layer(16, 16);
        Layer(16, 1);

        _evaluationTensor[x * 6 + y] = _layerOutput[0];
      }
    }

    _evaluationTensor[36] = WhiteToMoveFactor;

    Array.Copy(_evaluationTensor, _layerInput, 37);
    Layer(37, 32);
    Layer(32, 16);
    Layer(16, 1);

    return _layerOutput[0] + evaluation;
  }

  Board _board;
  Timer _timer;
  Move _bestMove;

  int _nodes; //#DEBUG

  int[] MoveScores = new int[218];

  int WhiteToMoveFactor => _board.IsWhiteToMove ? 1 : -1;

  // Hash, Move, Score, Depth, Bound
  (ulong, Move, float, int, int)[] _transpositionTable = new (ulong, Move, float, int, int)[40000];

  float Search(int ply, int depth, float alpha, float beta)
  {
    _nodes++; //#DEBUG

    ulong zobristKey = _board.ZobristKey;
    var (transpositionHash, transpositionMove, transpositionScore, transpositionDepth, transpositionFlag) = _transpositionTable[zobristKey % 40000];

    if (transpositionHash == zobristKey && transpositionDepth >= depth && (
      transpositionFlag == 1 ||
      transpositionFlag == 2 && transpositionScore <= alpha ||
      transpositionFlag == 3 && transpositionScore >= beta)
    ) return transpositionScore;

    bool qSearch = depth <= 0;

    if (qSearch)
    {
      alpha = MathF.Max(alpha, Inference() * WhiteToMoveFactor);

      if (alpha >= beta) return alpha;
    }

    bool isCheck = _board.IsInCheck();

    Span<Move> moves = stackalloc Move[218];
    _board.GetLegalMovesNonAlloc(ref moves, qSearch && !isCheck);

    if (moves.Length == 0) return Inference() * WhiteToMoveFactor;

    int index = 0;

    // Scores are sorted low to high
    foreach (Move move in moves)
    {
      MoveScores[index++] = move == transpositionMove ? -1000000 : move.IsCapture
        ? (int)move.MovePieceType - 100 * (int)move.CapturePieceType
        : 1000000;
    }

    MoveScores.AsSpan(0, moves.Length).Sort(moves);

    Move bestMove = moves[0];
    int newTranspositionFlag = 1;

    foreach (Move move in moves)
    {
      if (outOfTime && ply > 0) return -100000f;

      _board.MakeMove(move);

      float score = -Search(ply + 1, depth - 1, -beta, -alpha);

      _board.UndoMove(move);

      if (score > alpha)
      {
        newTranspositionFlag = 0;

        bestMove = move;

        if (ply == 0) _bestMove = move;

        alpha = score;

        if (score >= beta)
        {
          newTranspositionFlag = 2;

          break;
        }
      }
    }

    _transpositionTable[zobristKey % 40000] = (zobristKey, bestMove, alpha, depth, newTranspositionFlag);

    return alpha;
  }

  bool outOfTime => _timer.MillisecondsElapsedThisTurn >= _timer.MillisecondsRemaining / 30f;

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    _nodes = 0; //#DEBUG

    int depth = 1;

    Move lastBestMove = Move.NullMove;
    _bestMove = Move.NullMove;

    while (true)
    {
      Search(0, depth++, -1000000f, 1000000f);

      if (lastBestMove == Move.NullMove) lastBestMove = _bestMove;

      if (outOfTime) break;

      lastBestMove = _bestMove;

      if (depth > 50) break;
    }

    Console.WriteLine($"Nodes per second {_nodes / (timer.MillisecondsElapsedThisTurn / 1000f + 0.00001f)} Depth: {depth} Seconds {timer.MillisecondsElapsedThisTurn / 1000f}"); //#DEBUG

    return lastBestMove;
  }
}