using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using ChessChallenge.Application;
using ChessChallenge.Chess;

public class Trainer
{
  //                              Conv 1        Conv 2       Inf 1          Inf 2
  public static int WeightCount = 9 * 16 + 16 + 16 * 1 + 1 + 16 * 16 + 16 + 16 * 1 + 1;
  public static int RoundGameCount = 40;
  public static int RoundCount = 100000 + 1;

  public static float Mutation = 4;

  public static float Reward(Board board, Move move)
  {
    ChessChallenge.API.Board botBoard = new ChessChallenge.API.Board(new Board(board));
    ChessChallenge.API.Move botMove = new ChessChallenge.API.Move(MoveUtility.GetMoveNameUCI(move), botBoard);

    if (botMove.IsCapture) return PieceWorth[botMove.CapturePieceType];

    return 0;
  }

  public static float OponentReward(Board board, Move move)
  {
    ChessChallenge.API.Board botBoard = new ChessChallenge.API.Board(new Board(board));
    ChessChallenge.API.Move botMove = new ChessChallenge.API.Move(MoveUtility.GetMoveNameUCI(move), botBoard);

    if (botMove.IsCapture) return -PieceWorth[botMove.CapturePieceType];

    return 0;
  }

  public static float WinReward = 20;
  public static float DrawReward = 0;

  public static Dictionary<ChessChallenge.API.PieceType, float> PieceWorth = new Dictionary<ChessChallenge.API.PieceType, float>() {
    { ChessChallenge.API.PieceType.Pawn, 1 },
    { ChessChallenge.API.PieceType.Knight, 3 },
    { ChessChallenge.API.PieceType.Bishop, 3 },
    { ChessChallenge.API.PieceType.Rook, 5 },
    { ChessChallenge.API.PieceType.Queen, 9 },
    { ChessChallenge.API.PieceType.King, 0 }
  };

  public List<float[]> _oldWeightPool = new List<float[]>();
  public List<float[]> _weightPool = new List<float[]>();

  public static Process CreateEvaluationProcess()
  {
    Process process = new Process();
    process.StartInfo.FileName = "cmd.exe";
    process.StartInfo.RedirectStandardInput = true;
    process.StartInfo.RedirectStandardOutput = true;

    process.Start();
    process.BeginOutputReadLine();

    process.StandardInput.WriteLine("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\eval.exe");

    bool ready = false;

    process.OutputDataReceived += (sender, args) =>
    {
      if (args.Data == "uciok") ready = true;
    };

    process.StandardInput.WriteLine("uci");

    while (!ready) { }

    return process;
  }

  public static void EndEvaluationProcess(Process process)
  {
    process.StandardInput.WriteLine("quit");

    process.Close();
  }

  public static float Evaluate(Process process, int depth, Board board)
  {
    bool ready = false;
    bool complete = false;
    float evaluation = 0;

    process.OutputDataReceived += (sender, args) =>
    {
      if (args.Data == "readyok") ready = true;

      if (args.Data.StartsWith("info depth") && args.Data.Split(" ").Length > 9) evaluation = float.Parse(args.Data.Split(" ")[9]) / 100;

      if (!args.Data.StartsWith("bestmove ")) return;

      complete = true;
    };

    process.StandardInput.WriteLine("ucinewgame");

    process.StandardInput.WriteLine("isready");

    while (!ready) { }

    process.StandardInput.WriteLine("position fen " + FenUtility.CurrentFen(board));
    process.StandardInput.WriteLine("go depth " + depth);

    while (!complete) { }

    return evaluation;
  }

  public static void GenerateDataset()
  {
    Process process = CreateEvaluationProcess();

    string[] fens = File.ReadAllLines("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Fens\\Positions Medium.txt");

    string output = "";

    int index = 0;

    foreach (string fen in fens)
    {
      Board board = new Board();
      board.LoadPosition(fen);

      float evaluation = Evaluate(process, 5, board);

      output += fen + " | " + evaluation + "\n";

      Console.WriteLine("Evaluated " + index + " / " + fens.Length);

      index++;

      if (index % 1000 == 0)
      {
        EndEvaluationProcess(process);
        process = CreateEvaluationProcess();
      }
    }

    EndEvaluationProcess(process);

    File.WriteAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Medium.txt", output[..(output.Length - 1)]);
  }

  public void StartTraining(BoardUI boardUI)
  {
    TrainingGame game = new TrainingGame(
      new ChessPlayer(new Frederox.AlphaBeta.AlphaBeta(), ChallengeController.PlayerType.MyBot, 1000 * 60),
      new ChessPlayer(new Frederox.AlphaBeta.AlphaBeta(), ChallengeController.PlayerType.MyBot, 1000 * 60)
    );

    game.Start();

    // new Thread(() =>
    // {
    //   for (int i = 0; i < RoundGameCount; i++)
    //   {
    //     float[] weights = new float[WeightCount];

    //     for (int j = 0; j < WeightCount; j++)
    //     {
    //       weights[j] = (float)(new Random().NextDouble() * 2 - 1);
    //     }

    //     _oldWeightPool.Add(weights);
    //   }

    //   for (int i = 0; i < RoundGameCount; i++)
    //   {
    //     float[] weights = new float[WeightCount];

    //     for (int j = 0; j < WeightCount; j++)
    //     {
    //       weights[j] = (float)(new Random().NextDouble() * 2 - 1);
    //     }

    //     _weightPool.Add(weights);
    //   }

    //   for (int round = 0; round < RoundCount; round++)
    //   {
    //     StartTrainingRound(boardUI, round);
    //   }
    // }).Start();
  }

  private void StartTrainingRound(BoardUI boardUI, int roundNumber)
  {
    List<Thread> gameThreads = new List<Thread>();
    List<TrainingGame.Result> gameResults = new List<TrainingGame.Result>();

    for (int gameIndex = 0; gameIndex < RoundGameCount; gameIndex++)
    {
      List<Func<ChessChallenge.API.IChessBot>> possibleOponents = new List<Func<ChessChallenge.API.IChessBot>>() {
        () => new MyBot() { Weights = _oldWeightPool[new System.Random().Next(_oldWeightPool.Count)] },
        // () => new Frederox.AlphaBeta.AlphaBeta(),
        () => new EvilBot(),
      };

      TrainingGame game = new TrainingGame(
        new ChessPlayer(new MyBot() { Weights = _weightPool[gameIndex] }, ChallengeController.PlayerType.MyBot, 1000 * 60),
        new ChessPlayer(possibleOponents[new System.Random().Next(possibleOponents.Count)](), ChallengeController.PlayerType.MyBot, 1000 * 60)
      );

      Thread gameThread = new Thread(() =>
      {
        TrainingGame.Result result = game.Start();

        gameResults.Add(result);
      });

      gameThreads.Add(gameThread);

      gameThread.Start();
    }

    bool allGamesFinished = false;

    while (!allGamesFinished)
    {
      allGamesFinished = true;

      foreach (Thread thread in gameThreads)
      {
        if (thread.IsAlive) allGamesFinished = false;
      }

      Thread.Sleep(1);
    }

    float averageReward = 0;

    TrainingGame.Result bestResult = gameResults[0];

    foreach (TrainingGame.Result result in gameResults)
    {
      averageReward += result.Reward;

      if (result.Reward <= bestResult.Reward) continue;

      bestResult = result;
    }

    if (roundNumber % 100 == 0)
    {
      string bestResultWeightCheckpoint = "";

      foreach (float weight in bestResult.Weights)
      {
        bestResultWeightCheckpoint += weight + "\n";
      }

      string path = "D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Checkpoints\\Checkpoint " + roundNumber + ".txt";

      File.WriteAllText(path, bestResultWeightCheckpoint);
    }

    DisplayGame(boardUI, bestResult.Board, bestResult.NewBotIsWhite);

    List<float[]> _winnerWeightPool = new List<float[]>();

    gameResults.Sort((a, b) => b.Reward.CompareTo(a.Reward));

    for (int i = 0; i < _weightPool.Count / 4; i++)
    {
      _winnerWeightPool.Add(gameResults[i].Weights);
    }

    averageReward /= gameResults.Count;

    _oldWeightPool = new List<float[]>();

    foreach (float[] weights in _weightPool)
    {
      _oldWeightPool.Add(weights);
    }

    _weightPool = new List<float[]>();

    foreach (float[] weights in _winnerWeightPool)
    {
      _weightPool.Add(weights);

      for (int variation = 0; variation < 2; variation++)
      {
        float[] newWeights = new float[WeightCount];

        for (int i = 0; i < WeightCount; i++)
        {
          newWeights[i] = weights[i];
        }

        for (int splices = 0; splices < new System.Random().Next(0, 4); splices++)
        {
          int spliceStart = new Random().Next(0, WeightCount - 1);
          int spliceEnd = new Random().Next(spliceStart, WeightCount);

          float[] otherWeights = _winnerWeightPool[new Random().Next(0, _winnerWeightPool.Count)];
          newWeights = newWeights[..spliceStart].Concat(otherWeights[spliceStart..spliceEnd]).Concat(newWeights[spliceEnd..]).ToArray();
        }

        for (int i = 0; i < newWeights.Length; i++)
        {
          newWeights[i] += (float)(new Random().NextDouble() * Mutation * 2 - Mutation);
        }

        _weightPool.Add(newWeights);
      }

      float[] randomWeight = new float[WeightCount];
      for (int i = 0; i < WeightCount; i++)
      {
        randomWeight[i] = (float)(new Random().NextDouble() * 2 - 1);
      }

      _weightPool.Add(randomWeight);
    }

    Console.WriteLine("Finished training round " + roundNumber + ". Average reward: " + averageReward);
  }

  private bool _displayingGame = false;

  private void DisplayGame(BoardUI boardUI, Board board, bool white)
  {
    if (_displayingGame) return;

    new Thread(() =>
    {
      _displayingGame = true;

      Board displayBoard = new Board();
      displayBoard.LoadStartPosition();

      boardUI.SetPerspective(white);

      foreach (Move move in board.AllGameMoves)
      {
        displayBoard.MakeMove(move, false);

        boardUI.UpdatePosition(displayBoard, move, true);

        Thread.Sleep(200);
      }

      _displayingGame = false;
    }).Start();
  }
}
