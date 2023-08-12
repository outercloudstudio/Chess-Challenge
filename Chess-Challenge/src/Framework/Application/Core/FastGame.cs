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

  public static (Result, bool) Play(IChessBot bot1, IChessBot bot2, Board board, int duration)
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

    if (board.IsInCheckmate()) return (board.IsWhiteToMove ? Result.BlackWin : Result.WhiteWin, false);
    else if (whiteTime == 0) return (Result.BlackWin, true);
    else if (blackTime == 0) return (Result.WhiteWin, true);

    return (Result.Draw, false);
  }

  private static float InverseError(float x)
  {
    float a = 8 * (MathF.PI - 3) / (3 * MathF.PI * (4 - MathF.PI));
    float y = MathF.Log(1 - x * x);
    float z = 2 / (MathF.PI * a) + y / 2;

    float result = MathF.Sqrt(MathF.Sqrt(z * z - y / a) - z);

    if (x < 0) return -result;

    return result;
  }

  private static float PhiInverse(float value)
  {
    return MathF.Sqrt(2) * InverseError(2 * value - 1);
  }

  public static float EloDifference(float percentage) => -400 * MathF.Log(1f / percentage - 1f) / MathF.Log(10f);

  public static (int, int) Elo(int wins, int draws, int losses)
  {
    float score = wins + draws / 2f;
    float total = wins + draws + losses;
    float percentage = score / total;
    float eloDifference = EloDifference(percentage);

    float winPercent = wins / total;
    float drawPercent = draws / total;
    float lossPercent = losses / total;

    float winDeviation = winPercent * MathF.Pow(1f - percentage, 2f);
    float drawDeviation = drawPercent * MathF.Pow(0.5f - percentage, 2f);
    float lossDeviation = lossPercent * MathF.Pow(0f - percentage, 2f);
    float standardDeviation = MathF.Sqrt(winDeviation + drawDeviation + lossDeviation) / MathF.Sqrt(total);

    float confidencePercentage = 0.95f;
    float minConfidencePercentage = (1f - confidencePercentage) / 2f;
    float maxConfidencePercentage = 1f - minConfidencePercentage;
    float deviationMin = percentage + PhiInverse(minConfidencePercentage) * standardDeviation;
    float deviationMax = percentage + PhiInverse(maxConfidencePercentage) * standardDeviation;

    float difference = (EloDifference(deviationMax) - EloDifference(deviationMin)) / 2f;

    return ((int)MathF.Round(eloDifference), (int)MathF.Round(difference));
  }

  public static void Match(Func<IChessBot> bot1Constructor, Func<IChessBot> bot2Constructor, int duration)
  {
    string[] fens = FileHelper.ReadResourceFile("Fens.txt").Split('\n').Where(fen => fen.Length > 0).ToArray()[..500];

    int wins = 0;
    int draws = 0;
    int losses = 0;
    int timeouts = 0;

    Task[] tasks = new Task[fens.Length * 2];

    int index = 0;
    foreach (string fen in fens)
    {
      Task whiteTask = Task.Factory.StartNew(() =>
      {
        (Result whiteResult, bool timeout) = Play(bot1Constructor(), bot2Constructor(), Board.CreateBoardFromFEN(fen), duration);

        if (whiteResult == Result.WhiteWin)
        {
          wins++;
        }
        else if (whiteResult == Result.BlackWin)
        {
          losses++;

          if (timeout) timeouts++;
        }
        else
        {
          draws++;
        }

        Console.WriteLine(String.Format("Finished game {0} / {1} {2}", wins + losses + draws, fens.Length * 2, whiteResult));
        (int elo, int variation) = Elo(wins, draws, losses);
        Console.WriteLine(String.Format("Wins: {0} Draws: {1} Losses: {2} My Bot Timeouts: {3} Elo: {4} +- {5}", wins, draws, losses, timeouts, elo, variation));
      });

      Task blackTask = Task.Factory.StartNew(() =>
      {
        (Result blackResult, bool timeout) = FastGame.Play(bot2Constructor(), bot1Constructor(), Board.CreateBoardFromFEN(fen), duration);

        if (blackResult == Result.WhiteWin)
        {
          losses++;

          if (timeout) timeouts++;
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
        (int elo, int variation) = Elo(wins, draws, losses);
        Console.WriteLine(String.Format("Wins: {0} Draws: {1} Losses: {2} My Bot Timeouts: {3} Elo: {4} +- {5}", wins, draws, losses, timeouts, elo, variation));
      });

      tasks[index * 2] = whiteTask;
      tasks[index * 2 + 1] = whiteTask;

      index++;

      if (index % 6 == 0) Task.WaitAll(tasks.Where(task => task != null).ToArray());
    }

    Task.WaitAll(tasks);

    (int elo, int variation) = Elo(wins, draws, losses);
    Console.WriteLine(String.Format("Wins: {0} Draws: {1} Losses: {2} My Bot Timeouts: {3} Elo: {4} +- {5}", wins, draws, losses, timeouts, elo, variation));
  }
}