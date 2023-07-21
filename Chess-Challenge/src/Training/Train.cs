using System;
using System.Collections.Generic;
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
  }

  public Action<Board, Move> OnMoveMade;

  private ChessPlayer _whitePlayer;
  private ChessPlayer _blackPlayer;

  private Board _board = new Board();

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

    _board.LoadPosition(startFen);
  }

  public void LoadIntoUI(BoardUI boardUI)
  {
    boardUI.UpdatePosition(_board);
    boardUI.ResetSquareColours();
    boardUI.SetPerspective(true);
  }

  private ChessPlayer PlayerToMove()
  {
    return _board.IsWhiteToMove ? _whitePlayer : _blackPlayer;
  }

  private Move GetMove()
  {
    // Board b = new Board();
    // b.LoadPosition(FenUtility.CurrentFen(board));
    ChessChallenge.API.Board botBoard = new(new(_board));

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
    Span<Move> moves = _moveGenerator.GenerateMoves(_board);

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

      Thread.Sleep(50);
    }

    return new Result
    {
      GameResult = _result,
      WhiteReward = _whiteReward / (float)_totalMoves + Train.WhiteResultRewards[_result],
      BlackReward = _blackReward / (float)_totalMoves + Train.BlackResultRewards[_result],
      Moves = _board.plyCount
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

      Console.WriteLine($"Illegal move: {moveName} in position: {FenUtility.CurrentFen(_board)}");

      GameResult illegalResult = PlayerToMove() == _whitePlayer ? GameResult.WhiteIllegalMove : GameResult.BlackIllegalMove;

      EndGame(illegalResult);

      return;
    }

    _board.MakeMove(move, false);

    OnMoveMade?.Invoke(_board, move);

    GameResult result = Arbiter.GetGameState(_board);

    if (result != GameResult.InProgress)
    {
      EndGame(result);
    }
    else
    {
      _board.UndoMove(move);

      if (PlayerToMove() == _whitePlayer)
      {
        _whiteReward += Train.Reward(_board, move);
        _blackReward += Train.OponentReward(_board, move);
      }
      else
      {
        _blackReward += Train.Reward(_board, move);
        _whiteReward += Train.OponentReward(_board, move);
      }

      _board.MakeMove(move, false);
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
    { GameResult.WhiteIsMated, -10000 },
    { GameResult.WhiteIllegalMove, -10000 },
    { GameResult.WhiteTimeout, -10000 },
    { GameResult.BlackIsMated, 10000 },
    { GameResult.BlackIllegalMove, 10000 },
    { GameResult.BlackTimeout, 10000 },
    { GameResult.DrawByArbiter, -100 },
    { GameResult.FiftyMoveRule, -100 },
    { GameResult.InsufficientMaterial, -100 },
    { GameResult.Repetition, -100 },
    { GameResult.Stalemate, -100 },
  };

  public static Dictionary<GameResult, float> BlackResultRewards = new Dictionary<GameResult, float>() {
    { GameResult.WhiteIsMated, 10000 },
    { GameResult.WhiteIllegalMove, 10000 },
    { GameResult.WhiteTimeout, 10000 },
    { GameResult.BlackIsMated, -10000 },
    { GameResult.BlackIllegalMove, -10000 },
    { GameResult.BlackTimeout, -10000 },
    { GameResult.DrawByArbiter, -100 },
    { GameResult.FiftyMoveRule, -100 },
    { GameResult.InsufficientMaterial, -100 },
    { GameResult.Repetition, -100 },
    { GameResult.Stalemate, -100 },
  };

  public static Dictionary<ChessChallenge.API.PieceType, float> PieceWorth = new Dictionary<ChessChallenge.API.PieceType, float>() {
    { ChessChallenge.API.PieceType.Pawn, 1 },
    { ChessChallenge.API.PieceType.Knight, 3 },
    { ChessChallenge.API.PieceType.Bishop, 3 },
    { ChessChallenge.API.PieceType.Rook, 5 },
    { ChessChallenge.API.PieceType.Queen, 9 },
    { ChessChallenge.API.PieceType.King, 0 }
  };

  public static void StartTraining(BoardUI boardUI)
  {
    string startFen = FileHelper.ReadResourceFile("Fens.txt").Split('\n')[0];

    for (int gameIndex = 0; gameIndex < 1; gameIndex++)
    {
      TrainingGame game = new TrainingGame(
        new ChessPlayer(new MyBot(), ChallengeController.PlayerType.MyBot, 1000 * 60),
        new ChessPlayer(new MyBot(), ChallengeController.PlayerType.MyBot, 1000 * 60),
        startFen
      );

      game.OnMoveMade += (Board board, Move move) => boardUI.UpdatePosition(board, move, true);

      Thread thread = new Thread(() =>
      {
        TrainingGame.Result result = game.Start();

        Console.WriteLine("Finished game with result " + result.GameResult + " white: " + result.WhiteReward + " black: " + result.BlackReward + " in " + result.Moves + " moves");
      });

      thread.Start();
    }

    // TrainingGame testingGame = new TrainingGame(
    //   new ChessPlayer(new MyBot(), ChallengeController.PlayerType.MyBot, 1000 * 60),
    //   new ChessPlayer(new MyBot(), ChallengeController.PlayerType.MyBot, 1000 * 60),
    //   startFen
    // );

    // testingGame.LoadIntoUI(boardUI);

    // testingGame.OnMoveMade += (Board board, Move move) => boardUI.UpdatePosition(board, move, true);

    // testingGame.Start();
  }
}