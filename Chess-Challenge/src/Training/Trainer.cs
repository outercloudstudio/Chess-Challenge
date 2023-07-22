using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using ChessChallenge.Application;
using ChessChallenge.Chess;
using Raylib_cs;

public class TrainingGame
{
  public struct Result
  {
    public GameResult GameResult;
    public float Reward;
    public int Moves;
    public float[] Weights;
    public Board Board;
    public bool NewBotIsWhite;
  }

  public Action<Board, Move> OnMoveMade;

  public Board Board = new Board();

  private bool _newBotIsWhite;

  private ChessPlayer _whitePlayer;
  private ChessPlayer _blackPlayer;

  private MoveGenerator _moveGenerator = new MoveGenerator();

  private bool _gameOver;

  private GameResult _result;

  private float _reward;

  private int _totalMoves;

  public TrainingGame(ChessPlayer newBotPlayer, ChessPlayer oldBotPlayer)
  {
    _newBotIsWhite = new Random().Next(0, 2) == 0;

    if (_newBotIsWhite)
    {
      _whitePlayer = newBotPlayer;
      _blackPlayer = oldBotPlayer;
    }
    else
    {
      _whitePlayer = oldBotPlayer;
      _blackPlayer = newBotPlayer;
    }
  }

  public void LoadIntoUI(BoardUI boardUI)
  {
    boardUI.UpdatePosition(Board);
    boardUI.ResetSquareColours();
    boardUI.SetPerspective(true);
  }

  private ChessPlayer PlayerToMove()
  {
    return Board.IsWhiteToMove ? _whitePlayer : _blackPlayer;
  }

  private Move GetMove()
  {
    // Board b = new Board();
    // b.LoadPosition(FenUtility.CurrentFen(board));
    ChessChallenge.API.Board botBoard = new(new(Board));

    try
    {
      ChessChallenge.API.Timer timer = new(PlayerToMove().TimeRemainingMs);
      ChessChallenge.API.Move move = PlayerToMove().Bot.Think(botBoard, timer);

      return new Move(move.RawValue);
    }
    catch (Exception e)
    {
      Console.WriteLine("An error occurred while bot was thinking.\n" + e.ToString());

      GameResult result = PlayerToMove() == _whitePlayer ? GameResult.WhiteTimeout : GameResult.BlackTimeout;

      EndGame(result);
    }

    return Move.NullMove;
  }

  private bool IsLegalMove(Move move)
  {
    Span<Move> moves = _moveGenerator.GenerateMoves(Board);

    foreach (Move legalMove in moves)
    {
      if (move.Value == legalMove.Value)
      {
        return true;
      }
    }

    return false;
  }

  public Result Start()
  {
    Board.LoadStartPosition();

    while (!_gameOver)
    {
      MakeMove();
    }

    if (_newBotIsWhite)
    {
      float resultReward = 0;

      if (Arbiter.IsWhiteWinsResult(_result)) resultReward += Trainer.WinReward;
      if (Arbiter.IsBlackWinsResult(_result)) resultReward -= Trainer.WinReward;
      if (Arbiter.IsDrawResult(_result)) resultReward += Trainer.DrawReward;

      return new Result
      {
        GameResult = _result,
        Reward = _reward / (float)_totalMoves + resultReward,
        Moves = Board.plyCount,
        Weights = _whitePlayer.Bot != null ? ((MyBot)_whitePlayer.Bot).Weights : new float[0],
        Board = Board,
        NewBotIsWhite = _newBotIsWhite,
      };
    }
    else
    {
      float resultReward = 0;

      if (Arbiter.IsBlackWinsResult(_result)) resultReward += Trainer.WinReward;
      if (Arbiter.IsWhiteWinsResult(_result)) resultReward -= Trainer.WinReward;
      if (Arbiter.IsDrawResult(_result)) resultReward += Trainer.DrawReward;

      return new Result
      {
        GameResult = _result,
        Reward = _reward / (float)_totalMoves + resultReward,
        Moves = Board.plyCount,
        Weights = _blackPlayer.Bot != null ? ((MyBot)_blackPlayer.Bot).Weights : new float[0],
        Board = Board,
        NewBotIsWhite = _newBotIsWhite,
      };
    }
  }

  public void MakeMove()
  {
    _totalMoves++;

    double startThinkTime = Raylib.GetTime();

    Move move = GetMove();

    double thinkDuration = Raylib.GetTime() - startThinkTime;

    PlayerToMove().UpdateClock(thinkDuration);

    if (!IsLegalMove(move))
    {
      string moveName = MoveUtility.GetMoveNameUCI(move);

      Console.WriteLine($"Illegal move: {moveName} in position: {FenUtility.CurrentFen(Board)}");

      GameResult illegalResult = PlayerToMove() == _whitePlayer ? GameResult.WhiteIllegalMove : GameResult.BlackIllegalMove;

      EndGame(illegalResult);

      return;
    }

    bool isMyMove = (PlayerToMove() == _whitePlayer && _newBotIsWhite) || (PlayerToMove() == _blackPlayer && !_newBotIsWhite);

    Board.MakeMove(move, false);

    OnMoveMade?.Invoke(new Board(Board), move);

    if (isMyMove)
    {
      _reward += Trainer.Reward(Board, move) - Trainer.OponentReward(Board, move);
    }
    else
    {
      _reward -= Trainer.Reward(Board, move) - Trainer.OponentReward(Board, move);
    }

    GameResult result = Arbiter.GetGameState(Board);

    if (result != GameResult.InProgress) EndGame(result);
  }

  private void EndGame(GameResult result)
  {
    if (_gameOver) return;

    _gameOver = true;

    _result = result;
  }
}

public class Trainer
{
  // public static int WeightCount = 1092;
  public static int WeightCount = 3104;
  public static int RoundGameCount = 20;
  public static int RoundCount = 10000;

  public static float Mutation = 2;

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
  public static float DrawReward = -3;

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

  public void StartTraining(BoardUI boardUI)
  {
    new Thread(() =>
    {
      for (int i = 0; i < RoundGameCount; i++)
      {
        float[] weights = new float[WeightCount];

        for (int j = 0; j < WeightCount; j++)
        {
          weights[j] = (float)(new Random().NextDouble() * 2 - 1);
        }

        _oldWeightPool.Add(weights);
      }

      for (int i = 0; i < RoundGameCount; i++)
      {
        float[] weights = new float[WeightCount];

        for (int j = 0; j < WeightCount; j++)
        {
          weights[j] = (float)(new Random().NextDouble() * 2 - 1);
        }

        _weightPool.Add(weights);
      }

      for (int round = 0; round < RoundCount; round++)
      {
        StartTrainingRound(boardUI, round);
      }
    }).Start();
  }

  private void StartTrainingRound(BoardUI boardUI, int roundNumber)
  {
    List<Thread> gameThreads = new List<Thread>();
    List<TrainingGame.Result> gameResults = new List<TrainingGame.Result>();

    for (int gameIndex = 0; gameIndex < RoundGameCount; gameIndex++)
    {
      TrainingGame game = new TrainingGame(
        new ChessPlayer(new MyBot() { Weights = _weightPool[gameIndex] }, ChallengeController.PlayerType.MyBot, 1000 * 60),
        new ChessPlayer(new MyBot() { Weights = _oldWeightPool[new System.Random().Next(_oldWeightPool.Count)] }, ChallengeController.PlayerType.MyBot, 1000 * 60)
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

    for (int i = 0; i < _weightPool.Count / 2; i++)
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
      int spliceStart = new Random().Next(0, WeightCount - 1);
      int spliceEnd = new Random().Next(spliceStart, WeightCount);

      float[] otherWeights = _winnerWeightPool[new Random().Next(0, _winnerWeightPool.Count)];

      float[] newWeights = weights[..spliceStart].Concat(otherWeights[spliceStart..spliceEnd]).Concat(weights[spliceEnd..]).ToArray();

      for (int i = 0; i < newWeights.Length; i++)
      {
        weights[i] += (float)(new Random().NextDouble() * Mutation * 2 - Mutation);
        newWeights[i] += (float)(new Random().NextDouble() * Mutation * 2 - Mutation);
      }

      _weightPool.Add(weights);
      _weightPool.Add(newWeights);
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

        Thread.Sleep(20);
      }

      _displayingGame = false;
    }).Start();
  }
}
