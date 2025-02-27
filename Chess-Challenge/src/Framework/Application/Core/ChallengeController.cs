﻿using ChessChallenge.Chess;
using Raylib_cs;
using System;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using static ChessChallenge.Application.Settings;
using static ChessChallenge.Application.ConsoleHelper;
using System.Collections.Generic;

namespace ChessChallenge.Application
{
  public class ChallengeController
  {
    public enum PlayerType
    {
      Human,
      ARCNET1,
      MyBot,
      MyBotOld,
      ARCNET2_Optimized,
      ARCNET2_MoveOrdering,
      EloBot2,
      EvilBot,
      Tyrant,
      MyBotEvil,
      FrederoxQ,
      Theseus,
      Tier2,
    }

    // Game state
    Random rng;
    int gameID;
    bool isPlaying;
    Board board;
    public ChessPlayer PlayerWhite;
    public ChessPlayer PlayerBlack;

    float lastMoveMadeTime;
    bool isWaitingToPlayMove;
    Move moveToPlay;
    float playMoveTime;
    public bool HumanWasWhiteLastGame { get; private set; }

    // Bot match state
    readonly string[] botMatchStartFens;
    int botMatchGameIndex;
    public BotMatchStats BotStatsA { get; private set; }
    public BotMatchStats BotStatsB { get; private set; }
    bool botAPlaysWhite;


    // Bot task
    AutoResetEvent botTaskWaitHandle;
    bool hasBotTaskException;
    ExceptionDispatchInfo botExInfo;

    // Other
    public readonly BoardUI boardUI;
    readonly MoveGenerator moveGenerator;
    readonly int tokenCount;
    readonly int debugTokenCount;
    readonly StringBuilder pgns;

    public ChallengeController()
    {
      Log($"Launching Chess-Challenge version {Settings.Version}");
      (tokenCount, debugTokenCount) = GetTokenCount();
      Warmer.Warm();

      rng = new Random();
      moveGenerator = new();
      boardUI = new BoardUI();
      board = new Board();
      pgns = new();

      BotStatsA = new BotMatchStats("IBot");
      BotStatsB = new BotMatchStats("IBot");
      botMatchStartFens = FileHelper.ReadResourceFile("Fens.txt").Split('\n').Where(fen => fen.Length > 0).ToArray();
      botTaskWaitHandle = new AutoResetEvent(false);

      StartNewGame(PlayerType.Human, PlayerType.MyBot);

      int[] compressedParameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_10.txt")[0..4927].Select(line =>
      {
        float value = float.Parse(line);
        int quantized = (int)(MathF.Min(MathF.Max(MathF.Pow(MathF.Abs(value) / 6f, 1 / 3f) * (value < 0 ? -1 : 1) + 0.5f, 0f), 1f) * 64f);
        return quantized;
      }).ToArray();

      int compressedTokenCount = (int)MathF.Ceiling(compressedParameters.Length / 16f);
      Console.WriteLine($"Param Count: 4927 Compressed Tokens: {compressedTokenCount}"); //#DEBUG

      List<decimal> decimals = new List<decimal>();

      for (int readIndex = 0; readIndex < 4927; readIndex += 16)
      {
        byte[] bytes = new byte[16];

        for (int offset = 0; offset < Math.Min(16, compressedParameters.Length - readIndex); offset++)
        {
          int bits = offset * 6;
          int byteIndex = bits / 8;
          int bitsOffset = bits - byteIndex * 8;

          bytes[byteIndex] |= (byte)(compressedParameters[readIndex + offset] << bitsOffset);

          if (bitsOffset > 2) bytes[byteIndex + 1] |= (byte)(compressedParameters[readIndex + offset] >> 8 - bitsOffset);
        }

        decimals.Add(ByteArrayToDecimal(bytes, 0));
      }

      string output = "";
      foreach (decimal value in decimals)
      {
        output += value.ToString() + "M, ";
      }

      Console.WriteLine("Output " + decimals.Count + " decimals");

      File.WriteAllText("D:/Chess-Challenge/Training/Models/Lila_10_Compressed.txt", output);
    }

    public static decimal ByteArrayToDecimal(byte[] src, int offset)
    {
      using (MemoryStream stream = new MemoryStream(src))
      {
        stream.Position = offset;
        using (BinaryReader reader = new BinaryReader(stream))
          return reader.ReadDecimal();
      }
    }

    public void StartNewGame(PlayerType whiteType, PlayerType blackType)
    {
      // End any ongoing game
      EndGame(GameResult.DrawByArbiter, log: false, autoStartNextBotMatch: false);
      gameID = rng.Next();

      // Stop prev task and create a new one
      if (RunBotsOnSeparateThread)
      {
        // Allow task to terminate
        botTaskWaitHandle.Set();
        // Create new task
        botTaskWaitHandle = new AutoResetEvent(false);
        Task.Factory.StartNew(BotThinkerThread, TaskCreationOptions.LongRunning);
      }
      // Board Setup
      board = new Board();
      bool isGameWithHuman = whiteType is PlayerType.Human || blackType is PlayerType.Human;
      int fenIndex = isGameWithHuman ? 0 : botMatchGameIndex / 2;
      board.LoadPosition(botMatchStartFens[fenIndex]);
      // board.LoadPosition("8/8/7K/1p1r2r1/8/8/8/5k2 w - - 4 116");

      // Player Setup
      PlayerWhite = CreatePlayer(whiteType);
      PlayerBlack = CreatePlayer(blackType);
      PlayerWhite.SubscribeToMoveChosenEventIfHuman(OnMoveChosen);
      PlayerBlack.SubscribeToMoveChosenEventIfHuman(OnMoveChosen);

      // UI Setup
      boardUI.UpdatePosition(board);
      boardUI.ResetSquareColours();
      SetBoardPerspective();

      // Start
      isPlaying = true;
      NotifyTurnToMove();
    }

    void BotThinkerThread()
    {
      int threadID = gameID;
      //Console.WriteLine("Starting thread: " + threadID);

      while (true)
      {
        // Sleep thread until notified
        botTaskWaitHandle.WaitOne();
        // Get bot move
        if (threadID == gameID)
        {
          var move = GetBotMove();

          if (threadID == gameID)
          {
            OnMoveChosen(move);
          }
        }
        // Terminate if no longer playing this game
        if (threadID != gameID)
        {
          break;
        }
      }
      //Console.WriteLine("Exitting thread: " + threadID);
    }

    Move GetBotMove()
    {
      API.Board botBoard = new(board);
      try
      {
        API.Timer timer = new(PlayerToMove.TimeRemainingMs, PlayerNotOnMove.TimeRemainingMs, GameDurationMilliseconds);
        API.Move move = PlayerToMove.Bot.Think(botBoard, timer);
        return new Move(move.RawValue);
      }
      catch (Exception e)
      {
        Log("An error occurred while bot was thinking.\n" + e.ToString(), true, ConsoleColor.Red);
        hasBotTaskException = true;
        botExInfo = ExceptionDispatchInfo.Capture(e);
      }
      return Move.NullMove;
    }



    void NotifyTurnToMove()
    {
      //playerToMove.NotifyTurnToMove(board);
      if (PlayerToMove.IsHuman)
      {
        PlayerToMove.Human.SetPosition(FenUtility.CurrentFen(board));
        PlayerToMove.Human.NotifyTurnToMove();
      }
      else
      {
        if (RunBotsOnSeparateThread)
        {
          botTaskWaitHandle.Set();
        }
        else
        {
          double startThinkTime = Raylib.GetTime();
          var move = GetBotMove();
          double thinkDuration = Raylib.GetTime() - startThinkTime;
          PlayerToMove.UpdateClock(thinkDuration);
          OnMoveChosen(move);
        }
      }
    }

    void SetBoardPerspective()
    {
      // Board perspective
      if (PlayerWhite.IsHuman || PlayerBlack.IsHuman)
      {
        boardUI.SetPerspective(PlayerWhite.IsHuman);
        HumanWasWhiteLastGame = PlayerWhite.IsHuman;
      }
      else if (PlayerWhite.Bot != null && PlayerBlack.Bot != null)
      {
        boardUI.SetPerspective(PlayerWhite.PlayerType == PlayerType.MyBot);
      }
      else
      {
        boardUI.SetPerspective(PlayerWhite.Bot is MyBot);
      }
    }

    ChessPlayer CreatePlayer(PlayerType type)
    {
      return type switch
      {
        PlayerType.ARCNET1 => new ChessPlayer(new ARCNET1(), type, GameDurationMilliseconds),
        PlayerType.MyBot => new ChessPlayer(new MyBot(), type, GameDurationMilliseconds),
        PlayerType.MyBotOld => new ChessPlayer(new MyBotOld(), type, GameDurationMilliseconds),
        PlayerType.MyBotEvil => new ChessPlayer(new MyBotEvil(), type, GameDurationMilliseconds),
        PlayerType.ARCNET2_Optimized => new ChessPlayer(new ARCNET2_Optimized(), type, GameDurationMilliseconds),
        PlayerType.ARCNET2_MoveOrdering => new ChessPlayer(new ARCNET2_MoveOrdering(), type, GameDurationMilliseconds),
        PlayerType.EloBot2 => new ChessPlayer(new EloBot2(), type, GameDurationMilliseconds),
        PlayerType.EvilBot => new ChessPlayer(new EvilBot(), type, GameDurationMilliseconds),
        PlayerType.Tyrant => new ChessPlayer(new Tyrant(), type, GameDurationMilliseconds),
        PlayerType.FrederoxQ => new ChessPlayer(new Frederox.Quiescence.Quiescence(), type, GameDurationMilliseconds),
        PlayerType.Theseus => new ChessPlayer(new Theseus(), type, GameDurationMilliseconds),
        PlayerType.Tier2 => new ChessPlayer(new Tier2(), type, GameDurationMilliseconds),
        _ => new ChessPlayer(new HumanPlayer(boardUI), type)
      };
    }

    static (int totalTokenCount, int debugTokenCount) GetTokenCount()
    {
      string path = Path.Combine(Directory.GetCurrentDirectory(), "src", "My Bot", "MyBot.cs");

      using StreamReader reader = new(path);
      string txt = reader.ReadToEnd();
      return TokenCounter.CountTokens(txt);
    }

    void OnMoveChosen(Move chosenMove)
    {
      if (IsLegal(chosenMove))
      {
        if (PlayerToMove.IsBot)
        {
          moveToPlay = chosenMove;
          isWaitingToPlayMove = true;
          playMoveTime = lastMoveMadeTime + MinMoveDelay;
        }
        else
        {
          PlayMove(chosenMove);
        }
      }
      else
      {
        string moveName = MoveUtility.GetMoveNameUCI(chosenMove);
        string log = $"Illegal move: {moveName} in position: {FenUtility.CurrentFen(board)}";
        Log(log, true, ConsoleColor.Red);
        GameResult result = PlayerToMove == PlayerWhite ? GameResult.WhiteIllegalMove : GameResult.BlackIllegalMove;
        EndGame(result);
      }
    }

    void PlayMove(Move move)
    {
      if (isPlaying)
      {
        bool animate = PlayerToMove.IsBot;
        lastMoveMadeTime = (float)Raylib.GetTime();

        board.MakeMove(move, false);
        boardUI.UpdatePosition(board, move, animate);

        GameResult result = Arbiter.GetGameState(board);
        if (result == GameResult.InProgress)
        {
          NotifyTurnToMove();
        }
        else
        {
          EndGame(result);
        }
      }
    }

    void EndGame(GameResult result, bool log = true, bool autoStartNextBotMatch = true)
    {
      if (isPlaying)
      {
        isPlaying = false;
        isWaitingToPlayMove = false;
        gameID = -1;

        if (log)
        {
          Log("Game Over: " + result, false, ConsoleColor.Blue);
        }

        string pgn = PGNCreator.CreatePGN(board, result, GetPlayerName(PlayerWhite), GetPlayerName(PlayerBlack));
        pgns.AppendLine(pgn);

        // If 2 bots playing each other, start next game automatically.
        if (PlayerWhite.IsBot && PlayerBlack.IsBot)
        {
          UpdateBotMatchStats(result);
          botMatchGameIndex++;
          int numGamesToPlay = botMatchStartFens.Length * 2;

          if (botMatchGameIndex < numGamesToPlay && autoStartNextBotMatch)
          {
            botAPlaysWhite = !botAPlaysWhite;
            const int startNextGameDelayMs = 600;
            System.Timers.Timer autoNextTimer = new(startNextGameDelayMs);
            int originalGameID = gameID;
            autoNextTimer.Elapsed += (s, e) => AutoStartNextBotMatchGame(originalGameID, autoNextTimer);
            autoNextTimer.AutoReset = false;
            autoNextTimer.Start();

          }
          else if (autoStartNextBotMatch)
          {
            Log("Match finished", false, ConsoleColor.Blue);
          }
        }
      }
    }

    private void AutoStartNextBotMatchGame(int originalGameID, System.Timers.Timer timer)
    {
      if (originalGameID == gameID)
      {
        StartNewGame(PlayerBlack.PlayerType, PlayerWhite.PlayerType);
      }
      timer.Close();
    }


    void UpdateBotMatchStats(GameResult result)
    {
      UpdateStats(BotStatsA, botAPlaysWhite);
      UpdateStats(BotStatsB, !botAPlaysWhite);

      void UpdateStats(BotMatchStats stats, bool isWhiteStats)
      {
        // Draw
        if (Arbiter.IsDrawResult(result))
        {
          stats.NumDraws++;
        }
        // Win
        else if (Arbiter.IsWhiteWinsResult(result) == isWhiteStats)
        {
          stats.NumWins++;
        }
        // Loss
        else
        {
          stats.NumLosses++;
          stats.NumTimeouts += (result is GameResult.WhiteTimeout or GameResult.BlackTimeout) ? 1 : 0;
          stats.NumIllegalMoves += (result is GameResult.WhiteIllegalMove or GameResult.BlackIllegalMove) ? 1 : 0;
        }
      }
    }

    public void Update()
    {
      if (isPlaying)
      {
        PlayerWhite.Update();
        PlayerBlack.Update();

        PlayerToMove.UpdateClock(Raylib.GetFrameTime());
        if (PlayerToMove.TimeRemainingMs <= 0)
        {
          EndGame(PlayerToMove == PlayerWhite ? GameResult.WhiteTimeout : GameResult.BlackTimeout);
        }
        else
        {
          if (isWaitingToPlayMove && Raylib.GetTime() > playMoveTime)
          {
            isWaitingToPlayMove = false;
            PlayMove(moveToPlay);
          }
        }
      }

      if (hasBotTaskException)
      {
        hasBotTaskException = false;
        botExInfo.Throw();
      }
    }

    public void Draw()
    {
      boardUI.Draw();
      string nameW = GetPlayerName(PlayerWhite);
      string nameB = GetPlayerName(PlayerBlack);
      boardUI.DrawPlayerNames(nameW, nameB, PlayerWhite.TimeRemainingMs, PlayerBlack.TimeRemainingMs, isPlaying);
    }

    public void DrawOverlay()
    {
      BotBrainCapacityUI.Draw(tokenCount, debugTokenCount, MaxTokenCount);
      MenuUI.DrawButtons(this);
      MatchStatsUI.DrawMatchStats(this);
    }

    static string GetPlayerName(ChessPlayer player) => GetPlayerName(player.PlayerType);
    static string GetPlayerName(PlayerType type) => type.ToString();

    public void StartNewBotMatch(PlayerType botTypeA, PlayerType botTypeB)
    {
      EndGame(GameResult.DrawByArbiter, log: false, autoStartNextBotMatch: false);
      botMatchGameIndex = 0;
      string nameA = GetPlayerName(botTypeA);
      string nameB = GetPlayerName(botTypeB);
      if (nameA == nameB)
      {
        nameA += " (A)";
        nameB += " (B)";
      }
      BotStatsA = new BotMatchStats(nameA);
      BotStatsB = new BotMatchStats(nameB);
      botAPlaysWhite = true;
      Log($"Starting new match: {nameA} vs {nameB}", false, ConsoleColor.Blue);
      StartNewGame(botTypeA, botTypeB);
    }


    ChessPlayer PlayerToMove => board.IsWhiteToMove ? PlayerWhite : PlayerBlack;
    ChessPlayer PlayerNotOnMove => board.IsWhiteToMove ? PlayerBlack : PlayerWhite;

    public int TotalGameCount => botMatchStartFens.Length * 2;
    public int CurrGameNumber => Math.Min(TotalGameCount, botMatchGameIndex + 1);
    public string AllPGNs => pgns.ToString();


    bool IsLegal(Move givenMove)
    {
      var moves = moveGenerator.GenerateMoves(board);
      foreach (var legalMove in moves)
      {
        if (givenMove.Value == legalMove.Value)
        {
          return true;
        }
      }

      return false;
    }

    public class BotMatchStats
    {
      public string BotName;
      public int NumWins;
      public int NumLosses;
      public int NumDraws;
      public int NumTimeouts;
      public int NumIllegalMoves;

      public BotMatchStats(string name) => BotName = name;
    }

    public void Release()
    {
      boardUI.Release();
    }
  }
}
