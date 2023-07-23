using System;
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
