﻿using System;
using System.Collections.Generic;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  public float[] Weights;

  Dictionary<PieceType, int> _pieceIds = new Dictionary<PieceType, int>() {
    { PieceType.Pawn, 1 },
    { PieceType.Knight, 3 },
    { PieceType.Bishop, 4 },
    { PieceType.Rook, 5 },
    { PieceType.Queen, 9 },
    { PieceType.King, 10 },
    { PieceType.None, 0 }
  };

  float Inference(Board board, Move move)
  {
    float[] inputValues = new float[64];

    board.MakeMove(move);

    if (board.IsInCheckmate())
    {
      board.UndoMove(move);

      return 9999999;
    }

    for (int squareIndex = 0; squareIndex < inputValues.Length; squareIndex++)
    {
      Piece piece = board.GetPiece(new Square(squareIndex));
      bool isMyPiece = piece.IsWhite && !board.IsWhiteToMove;

      inputValues[squareIndex] = _pieceIds[piece.PieceType] * (isMyPiece ? 1 : -1);
    }

    board.UndoMove(move);

    float[] hiddenValues = new float[32];

    for (int nodeIndex = 0; nodeIndex < hiddenValues.Length; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < inputValues.Length; weightIndex++)
      {
        hiddenValues[nodeIndex] += inputValues[weightIndex] * Weights[nodeIndex * inputValues.Length + weightIndex];
      }
    }

    float[] hiddenValues2 = new float[32];

    for (int nodeIndex = 0; nodeIndex < hiddenValues2.Length; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < hiddenValues.Length; weightIndex++)
      {
        hiddenValues2[nodeIndex] += hiddenValues[weightIndex] * Weights[inputValues.Length * hiddenValues.Length + nodeIndex * hiddenValues.Length + weightIndex];
      }
    }

    float outputValue = 0;

    for (int weightIndex = 0; weightIndex < hiddenValues2.Length; weightIndex++)
    {
      outputValue += hiddenValues2[weightIndex] * Weights[inputValues.Length * hiddenValues.Length + hiddenValues.Length * hiddenValues2.Length + weightIndex];
    }

    return outputValue;
  }

  /*
  Network Architecture:
  Input: 64,
  Hidden: 16,
  Hidden: 4,
  Output: 1
  */

  struct MoveChoice
  {
    public Move Move;
    public float Evaluation;
  }

  public Move Think(Board board, Timer timer)
  {
    if (Weights == null) Weights = new float[Trainer.WeightCount];

    List<Move> moves = new List<Move>(board.GetLegalMoves());
    List<MoveChoice> moveChoices = new List<MoveChoice>();

    foreach (Move move in moves)
    {
      moveChoices.Add(new MoveChoice()
      {
        Move = move,
        Evaluation = Inference(board, move)
      });
    }

    moveChoices.Sort((a, b) => b.Evaluation.CompareTo(a.Evaluation));

    return moveChoices[new System.Random().Next(0, Math.Min(moveChoices.Count, 3))].Move;
  }
}