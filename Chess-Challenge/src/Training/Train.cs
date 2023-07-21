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

  public TrainingGame(ChessPlayer whitePlayer, ChessPlayer blackPlayer, string startFen)
  {
    _whitePlayer = whitePlayer;
    _blackPlayer = blackPlayer;

    Board.LoadPosition(startFen);
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
    while (!_gameOver)
    {
      MakeMove();
    }

    return new Result
    {
      GameResult = _result,
      WhiteReward = _whiteReward / (float)_totalMoves + Train.WhiteResultRewards[_result],
      BlackReward = _blackReward / (float)_totalMoves + Train.BlackResultRewards[_result],
      Moves = Board.plyCount,
      WhiteWeights = ((MyBot)_whitePlayer.Bot).Weights,
      BlackWeights = ((MyBot)_blackPlayer.Bot).Weights,
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
        _whiteReward += Train.Reward(Board, move);
        _blackReward += Train.OponentReward(Board, move);
      }
      else
      {
        _blackReward += Train.Reward(Board, move);
        _whiteReward += Train.OponentReward(Board, move);
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

public class Train
{
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

  private static int s_GamesInProgress;
  private static Board s_BestGameBoard;
  private static float s_BestGameEvaluation;
  private static GameResult s_BestGameResult;

  public static List<float[]> WeightPool = new List<float[]>();
  public static List<float[]> WinnerWeightPool = new List<float[]>();

  private List<Thread> _gameThreads = new List<Thread>();

  public static void StartTraining(BoardUI boardUI)
  {
    new Thread(() =>
    {
      for (int i = 0; i < 2 * 100; i++)
      {
        WeightPool.Add(new float[] { -1, -1, 4, -1, 6, -2, 5, -2, 1, 5, 2, 1, 5, 0, 2, 5 });
      }

      StartTrainingRound(boardUI);
    }).Start();
  }

  private static void StartTrainingRound(BoardUI boardUI)
  {
    WinnerWeightPool = new List<float[]>();

    WeightsIndex = 0;

    string startFen = FileHelper.ReadResourceFile("Fens.txt").Split('\n')[0];

    s_GamesInProgress = 100;
    s_BestGameEvaluation = -999999999999;

    for (int gameIndex = 0; gameIndex < 100; gameIndex++)
    {
      int originalWeightsIndex = WeightsIndex;

      WeightsIndex = originalWeightsIndex;

      TrainingGame game = new TrainingGame(
        new ChessPlayer(new MyBot(), ChallengeController.PlayerType.MyBot, 1000 * 60),
        new ChessPlayer(new MyBot(), ChallengeController.PlayerType.MyBot, 1000 * 60),
        startFen
      );

      Thread thread = new Thread(() =>
      {
        TrainingGame.Result result = game.Start();

        // Console.WriteLine("Finished game with result " + result.GameResult + " white: " + result.WhiteReward + " black: " + result.BlackReward + " in " + result.Moves + " moves");

        if (s_BestGameEvaluation < MathF.Max(result.WhiteReward, result.BlackReward))
        {
          s_BestGameEvaluation = MathF.Max(result.WhiteReward, result.BlackReward);
          s_BestGameBoard = game.Board;
          s_BestGameResult = result.GameResult;
        }

        if (result.WhiteReward > result.BlackReward)
        {
          WinnerWeightPool.Add(result.WhiteWeights);
        }
        else
        {
          WinnerWeightPool.Add(result.BlackWeights);
        }

        s_GamesInProgress--;

        Console.WriteLine(s_GamesInProgress + " games left in round!");

        if (s_GamesInProgress == 0) EndTrainingRound(boardUI);
      });

      thread.Start();
    }
  }

  private static void EndTrainingRound(BoardUI boardUI)
  {
    Console.WriteLine("Finished training round with best game evaluation: " + s_BestGameEvaluation + " " + s_BestGameResult);

    new Thread(() =>
    {
      Board displayBoard = new Board();
      displayBoard.LoadPosition(s_BestGameBoard.GameStartFen);

      foreach (Move move in s_BestGameBoard.AllGameMoves)
      {
        // displayBoard.MakeMove(move, false);

        // boardUI.UpdatePosition(displayBoard, move, true);

        // Thread.Sleep(50);
      }

      WeightPool = new List<float[]>();

      foreach (float[] weights in WinnerWeightPool)
      {
        WeightPool.Add(weights);

        int spliceStart = new Random().Next(0, 15);
        int spliceEnd = new Random().Next(spliceStart, 16);

        float[] otherWeights = WinnerWeightPool[new Random().Next(0, WinnerWeightPool.Count)];

        float[] newWeights = weights[..spliceStart].Concat(otherWeights[spliceStart..spliceEnd]).Concat(weights[spliceEnd..]).ToArray();

        WeightPool.Add(weights);
        WeightPool.Add(newWeights);
      }

      StartTrainingRound(boardUI);
    }).Start();
  }

  private static int WeightsIndex = 0;

  public static float[] GetWeights()
  {
    if (WeightPool.Count == 0) return new float[] { -1, -1, 4, -1, 6, -2, 5, -2, 1, 5, 2, 1, 5, 0, 2, 5 };

    lock (WeightPool)
    {
      float[] weights = WeightPool[WeightsIndex];

      WeightsIndex++;

      return weights;
    }
  }
}
