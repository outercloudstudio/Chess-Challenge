using System;
using System.Collections.Generic;
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
    public float WhiteReward;
    public float BlackReward;
    public int Moves;
    public float[] WhiteWeights;
    public float[] BlackWeights;
    public Board Board;
  }

  public Action<Board, Move> OnMoveMade;

  public Board Board = new Board();

  private ChessPlayer _whitePlayer;
  private ChessPlayer _blackPlayer;

  private MoveGenerator _moveGenerator = new MoveGenerator();

  private bool _gameOver;

  private GameResult _result;

  private float _whiteReward;
  private float _blackReward;

  private int _totalMoves;

  public TrainingGame(ChessPlayer whitePlayer, ChessPlayer blackPlayer)
  {
    _whitePlayer = whitePlayer;
    _blackPlayer = blackPlayer;
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

    return new Result
    {
      GameResult = _result,
      WhiteReward = _whiteReward / (float)_totalMoves + Trainer.WhiteResultRewards[_result],
      BlackReward = _blackReward / (float)_totalMoves + Trainer.BlackResultRewards[_result],
      Moves = Board.plyCount,
      WhiteWeights = _whitePlayer.Bot != null ? ((MyBot)_whitePlayer.Bot).Weights : new float[0],
      BlackWeights = _blackPlayer.Bot != null ? ((MyBot)_blackPlayer.Bot).Weights : new float[0],
      Board = Board
    };
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

    Board.MakeMove(move, false);

    OnMoveMade?.Invoke(new Board(Board), move);

    GameResult result = Arbiter.GetGameState(Board);

    if (result != GameResult.InProgress)
    {
      EndGame(result);
    }
    else
    {
      Board.UndoMove(move);

      if (PlayerToMove() == _whitePlayer)
      {
        _whiteReward += Trainer.Reward(Board, move);
        _blackReward += Trainer.OponentReward(Board, move);
      }
      else
      {
        _blackReward += Trainer.Reward(Board, move);
        _whiteReward += Trainer.OponentReward(Board, move);
      }

      Board.MakeMove(move, true);
    }
  }

  private void EndGame(GameResult result)
  {
    _gameOver = true;

    _result = result;
  }
}

public class Trainer
{
  public static int WeightCount = 1092;

  public static float Mutation = 1;

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

  public static Dictionary<GameResult, float> WhiteResultRewards = new Dictionary<GameResult, float>() {
    { GameResult.WhiteIsMated, -20 },
    { GameResult.WhiteIllegalMove, -20 },
    { GameResult.WhiteTimeout, -20 },
    { GameResult.BlackIsMated, 20 },
    { GameResult.BlackIllegalMove, 20 },
    { GameResult.BlackTimeout, 20 },
    { GameResult.DrawByArbiter, -3 },
    { GameResult.FiftyMoveRule, -3 },
    { GameResult.InsufficientMaterial, -3 },
    { GameResult.Repetition, -3 },
    { GameResult.Stalemate, -3 },
  };

  public static Dictionary<GameResult, float> BlackResultRewards = new Dictionary<GameResult, float>() {
    { GameResult.WhiteIsMated, 20 },
    { GameResult.WhiteIllegalMove, 20 },
    { GameResult.WhiteTimeout, 20 },
    { GameResult.BlackIsMated, -20 },
    { GameResult.BlackIllegalMove, -20 },
    { GameResult.BlackTimeout, -20 },
    { GameResult.DrawByArbiter, -3 },
    { GameResult.FiftyMoveRule, -3 },
    { GameResult.InsufficientMaterial, -3 },
    { GameResult.Repetition, -3 },
    { GameResult.Stalemate, -3 },
  };

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
      for (int i = 0; i < 2 * 100; i++)
      {
        float[] weights = new float[WeightCount];

        for (int j = 0; j < WeightCount; j++)
        {
          weights[j] = (float)(new Random().NextDouble() * 2 - 1);
        }

        _weightPool.Add(weights);
      }

      while (true) StartTrainingRound(boardUI);
    }).Start();
  }

  private void StartTrainingRound(BoardUI boardUI)
  {
    List<Thread> _gameThreads = new List<Thread>();
    List<TrainingGame.Result> _gameResults = new List<TrainingGame.Result>();

    for (int gameIndex = 0; gameIndex < 100; gameIndex++)
    {
      TrainingGame game = new TrainingGame(
        new ChessPlayer(new MyBot() { Weights = _weightPool[gameIndex * 2] }, ChallengeController.PlayerType.MyBot, 1000 * 60),
        new ChessPlayer(new MyBot() { Weights = _weightPool[gameIndex * 2 + 1] }, ChallengeController.PlayerType.MyBot, 1000 * 60)
      );

      Thread gameThread = new Thread(() =>
      {
        TrainingGame.Result result = game.Start();

        _gameResults.Add(result);
      });

      _gameThreads.Add(gameThread);

      gameThread.Start();
    }

    bool allGamesFinished = false;

    while (!allGamesFinished)
    {
      allGamesFinished = true;

      foreach (Thread thread in _gameThreads)
      {
        if (thread.IsAlive) allGamesFinished = false;
      }

      Thread.Sleep(1);
    }

    float averageReward = 0;

    List<float[]> _winnerWeightPool = new List<float[]>();

    TrainingGame.Result bestResult = _gameResults[0];

    foreach (TrainingGame.Result result in _gameResults)
    {
      if (result.WhiteReward > result.BlackReward)
      {
        _winnerWeightPool.Add(result.WhiteWeights);

        averageReward += result.WhiteReward;
      }
      else
      {
        _winnerWeightPool.Add(result.BlackWeights);

        averageReward += result.BlackReward;
      }

      if (result.WhiteReward <= bestResult.WhiteReward) continue;
      if (result.BlackReward <= bestResult.BlackReward) continue;

      bestResult = result;
    }

    DisplayGame(boardUI, bestResult.Board);

    averageReward /= _gameResults.Count;

    _weightPool = new List<float[]>();

    foreach (float[] weights in _winnerWeightPool)
    {
      _weightPool.Add(weights);

      int spliceStart = new Random().Next(0, WeightCount - 1);
      int spliceEnd = new Random().Next(spliceStart, WeightCount);

      float[] otherWeights = _winnerWeightPool[new Random().Next(0, _winnerWeightPool.Count)];

      float[] newWeights = weights[..spliceStart].Concat(otherWeights[spliceStart..spliceEnd]).Concat(weights[spliceEnd..]).ToArray();

      for (int i = 0; i < newWeights.Length; i++)
      {
        newWeights[i] += (float)(new Random().NextDouble() * Mutation * 2 - Mutation);
      }

      _weightPool.Add(weights);
      _weightPool.Add(newWeights);
    }

    Console.WriteLine("Finished training round. Average reward: " + averageReward);
  }

  private bool _displayingGame = false;

  private void DisplayGame(BoardUI boardUI, Board board)
  {
    if (_displayingGame) return;

    new Thread(() =>
    {
      _displayingGame = true;

      Board displayBoard = new Board();
      displayBoard.LoadStartPosition();

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
