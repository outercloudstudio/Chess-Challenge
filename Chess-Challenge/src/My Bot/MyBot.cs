using System;
using ChessChallenge.API;

public class MyBot : IChessBot
{
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

  int ChooseMove(int[] evaluations)
  {
    int evaluationSum = 0;

    foreach (int evaluation in evaluations)
    {
      evaluationSum += evaluation;
    }

    int target = (int)new System.Random().NextInt64(0, evaluationSum);

    int indexSum = 0;

    for (int index = 0; index < evaluations.Length; index++)
    {
      indexSum += evaluations[index];

      if (indexSum > target) return index;
    }

    return evaluations.Length - 1;
  }

  public Move Think(Board board, Timer timer)
  {
    Move[] moves = board.GetLegalMoves();
    int[] evaluations = new int[moves.Length];

    for (int moveIndex = 0; moveIndex < moves.Length; moveIndex++)
    {
      evaluations[moveIndex] = Evaluate(board, moves[moveIndex]);
    }

    int chosenMoveIndex = ChooseMove(evaluations);

    Console.WriteLine(chosenMoveIndex);

    return moves[chosenMoveIndex];
  }
}