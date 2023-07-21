using System;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  public float[] Weights = Train.GetWeights();

  int Evaluate(Board board, Move move)
  {
    board.MakeMove(move);
    bool isMate = board.IsInCheckmate();
    bool isCheck = board.IsInCheck();
    board.UndoMove(move);

    if (isMate) return 1000000;

    if (isCheck) return 1000;

    if (move.IsCapture) return 500;

    if (move.IsCastles) return 200;

    return 1;
  }

  float Inference(int evaluation, float[] weights)
  {
    float[] hiddenValues = new float[8];

    for (int index = 0; index < 8; index++)
    {
      hiddenValues[index] = evaluation * weights[index];
    }

    float outputValue = 0;

    for (int index = 0; index < 8; index++)
    {
      outputValue += hiddenValues[index] * weights[8 + index];
    }

    return outputValue;
  }

  /*
  Network Architecture:
  Input: 1,
  Hidden: 8,
  Output: 1
  */

  public Move Think(Board board, Timer timer)
  {
    Move[] moves = board.GetLegalMoves();

    Move bestMove = moves[new System.Random().Next(moves.Length)];
    float bestMoveEvaluation = Inference(Evaluate(board, bestMove), Weights);

    foreach (Move move in moves)
    {
      float evaluation = Inference(Evaluate(board, move), Weights);

      if (evaluation <= bestMoveEvaluation) continue;

      bestMove = move;
      bestMoveEvaluation = evaluation;
    }

    return bestMove;
  }
}