using ChessChallenge.API;

class FastGame
{
  public enum Result
  {
    WhiteWin,
    BlackWin,
    Draw
  }

  public static Result Play(IChessBot bot1, IChessBot bot2, Board board, int duration)
  {
    int whiteTime = duration;
    int blackTime = duration;

    while (!board.IsInCheckmate() && !board.IsDraw() && whiteTime > 0 && blackTime > 0)
    {
      if (board.IsWhiteToMove)
      {
        Timer timer = new Timer(whiteTime, blackTime, duration);

        board.MakeMove(bot1.Think(board, timer));

        whiteTime = timer.MillisecondsRemaining;
      }
      else
      {
        Timer timer = new Timer(blackTime, whiteTime, duration);

        board.MakeMove(bot2.Think(board, timer));

        blackTime = timer.MillisecondsRemaining;
      }
    }

    if (board.IsInCheckmate()) return board.IsWhiteToMove ? Result.BlackWin : Result.WhiteWin;
    else if (whiteTime == 0) return Result.BlackWin;
    else if (blackTime == 0) return Result.WhiteWin;

    return Result.Draw;
  }
}