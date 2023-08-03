using System.Diagnostics;
using ChessChallenge.API;

public class StockFish : IChessBot
{
  int Elo;

  Process evaluationProcess = Evaluation.CreateEvaluationProcess();

  public StockFish(int elo = 1800)
  {
    Elo = elo;
  }

  public Move Think(Board board, Timer timer)
  {
    return new Move(Evaluation.BestMoveAtElo(
      evaluationProcess,
      Elo,
      board.GetFenString(),
      board.IsWhiteToMove ? timer.MillisecondsRemaining : timer.OpponentMillisecondsRemaining,
      board.IsWhiteToMove ? timer.OpponentMillisecondsRemaining : timer.OpponentMillisecondsRemaining
    ), board);
  }
}