using System;
using System.Linq;
using System.Threading.Tasks;
using ChessChallenge.API;
using ChessChallenge.Application;

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

  public static void Match(Func<IChessBot> bot1Constructor, Func<IChessBot> bot2Constructor, int duration)
  {
    string[] fens = FileHelper.ReadResourceFile("Fens.txt").Split('\n').Where(fen => fen.Length > 0).ToArray()[..500];

    int wins = 0;
    int draws = 0;
    int losses = 0;

    Task[] tasks = new Task[fens.Length * 2];

    int index = 0;
    foreach (string fen in fens)
    {
      Task whiteTask = Task.Factory.StartNew(() =>
      {
        Result whiteResult = Play(bot1Constructor(), bot2Constructor(), Board.CreateBoardFromFEN(fen), 30 * 1000);

        if (whiteResult == Result.WhiteWin)
        {
          wins++;
        }
        else if (whiteResult == Result.BlackWin)
        {
          losses++;
        }
        else
        {
          draws++;
        }

        Console.WriteLine(String.Format("Finished game {0} / {1} {2}", wins + losses + draws, fens.Length * 2, whiteResult));
        Console.WriteLine(String.Format("Wins: {0} Draws: {1} Losses: {2}", wins, draws, losses));
      });

      Task blackTask = Task.Factory.StartNew(() =>
      {
        FastGame.Result blackResult = FastGame.Play(bot2Constructor(), bot1Constructor(), Board.CreateBoardFromFEN(fen), 30 * 1000);

        if (blackResult == Result.WhiteWin)
        {
          losses++;
        }
        else if (blackResult == Result.BlackWin)
        {
          wins++;
        }
        else
        {
          draws++;
        }

        Console.WriteLine(String.Format("Finished game {0} / {1} {2}", wins + losses + draws, fens.Length * 2, blackResult));
        Console.WriteLine(String.Format("Wins: {0} Draws: {1} Losses: {2}", wins, draws, losses));
      });

      tasks[index * 2] = whiteTask;
      tasks[index * 2 + 1] = whiteTask;

      index++;

      // if (index % 40 == 0) Task.WaitAll(tasks.Where(task => task != null).ToArray());
    }

    Task.WaitAll(tasks);

    Console.WriteLine(String.Format("Wins: {0} Draws: {1} Losses: {2}", wins, draws, losses));
  }
}