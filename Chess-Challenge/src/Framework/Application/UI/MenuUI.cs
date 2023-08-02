﻿using Raylib_cs;
using System.Numerics;
using System;
using System.IO;

namespace ChessChallenge.Application
{
  public static class MenuUI
  {
    public static Trainer Trainer;

    public static void DrawButtons(ChallengeController controller)
    {
      Vector2 buttonPos = UIHelper.Scale(new Vector2(130, 100));
      Vector2 buttonSize = UIHelper.Scale(new Vector2(220, 55));
      float spacing = buttonSize.Y * 1.2f;
      float breakSpacing = spacing * 0.6f;

      // Game Buttons
      if (NextButtonInRow("Human vs My Bot", ref buttonPos, spacing, buttonSize))
      {
        var whiteType = controller.HumanWasWhiteLastGame ? ChallengeController.PlayerType.MyBot : ChallengeController.PlayerType.Human;
        var blackType = !controller.HumanWasWhiteLastGame ? ChallengeController.PlayerType.MyBot : ChallengeController.PlayerType.Human;
        controller.StartNewGame(whiteType, blackType);
      }
      if (NextButtonInRow("Human vs ARCNET 2 Move Ordering", ref buttonPos, spacing, buttonSize))
      {
        var whiteType = controller.HumanWasWhiteLastGame ? ChallengeController.PlayerType.ARCNET2_MoveOrdering : ChallengeController.PlayerType.Human;
        var blackType = !controller.HumanWasWhiteLastGame ? ChallengeController.PlayerType.ARCNET2_MoveOrdering : ChallengeController.PlayerType.Human;
        controller.StartNewGame(whiteType, blackType);
      }
      if (NextButtonInRow("My Bot vs My Bot", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.MyBot);
      }
      if (NextButtonInRow("My Bot vs My Bot No Transposition", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.MyBotNoTransposition);
      }
      if (NextButtonInRow("v T0", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.EvilBot);
      }
      if (NextButtonInRow("v T1", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.ARCNET2_MoveOrdering);
      }
      if (NextButtonInRow("v T2", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.EloBot2);
      }
      if (NextButtonInRow("v Tyrant", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.Tyrant);
      }
      // if (NextButtonInRow("ARCNET 2 vs ARCNET 2", ref buttonPos, spacing, buttonSize))
      // {
      //   controller.StartNewBotMatch(ChallengeController.PlayerType.ARCNET2, ChallengeController.PlayerType.ARCNET2);
      // }
      // if (NextButtonInRow("ARCNET 2 vs ARCNET 1", ref buttonPos, spacing, buttonSize))
      // {
      //   controller.StartNewBotMatch(ChallengeController.PlayerType.ARCNET2, ChallengeController.PlayerType.ARCNET1);
      // }
      // if (NextButtonInRow("ARCNET 2 vs ARCNET 2 Optimized", ref buttonPos, spacing, buttonSize))
      // {
      //   controller.StartNewBotMatch(ChallengeController.PlayerType.ARCNET2, ChallengeController.PlayerType.ARCNET2_Optimized);
      // }
      // if (NextButtonInRow("ARCNET 2 vs Test Evil Bot", ref buttonPos, spacing, buttonSize))
      // {
      //   controller.StartNewBotMatch(ChallengeController.PlayerType.ARCNET2, ChallengeController.PlayerType.EvilBot);
      // }
      // if (NextButtonInRow("ARCNET 2 vs Elo Bot 2", ref buttonPos, spacing, buttonSize))
      // {
      //   controller.StartNewBotMatch(ChallengeController.PlayerType.ARCNET2, ChallengeController.PlayerType.EloBot2);
      // }

      buttonPos = UIHelper.Scale(new Vector2(390, 100));


      // Page buttons
      if (NextButtonInRow("Start Training Server", ref buttonPos, spacing, buttonSize))
      {
        if (Trainer != null)
          Trainer.StopServer();

        Trainer = new Trainer(controller);

        Trainer.StartServer();
      }
      if (NextButtonInRow("Save Games", ref buttonPos, spacing, buttonSize))
      {
        string pgns = controller.AllPGNs;
        string directoryPath = Path.Combine(FileHelper.AppDataPath, "Games");
        Directory.CreateDirectory(directoryPath);
        string fileName = FileHelper.GetUniqueFileName(directoryPath, "games", ".txt");
        string fullPath = Path.Combine(directoryPath, fileName);
        File.WriteAllText(fullPath, pgns);
        ConsoleHelper.Log("Saved games to " + fullPath, false, ConsoleColor.Blue);
      }
      if (NextButtonInRow("Rules & Help", ref buttonPos, spacing, buttonSize))
      {
        FileHelper.OpenUrl("https://github.com/SebLague/Chess-Challenge");
      }
      if (NextButtonInRow("Documentation", ref buttonPos, spacing, buttonSize))
      {
        FileHelper.OpenUrl("https://seblague.github.io/chess-coding-challenge/documentation/");
      }
      if (NextButtonInRow("Submission Page", ref buttonPos, spacing, buttonSize))
      {
        FileHelper.OpenUrl("https://forms.gle/6jjj8jxNQ5Ln53ie6");
      }

      // Window and quit buttons
      buttonPos.Y += breakSpacing;

      bool isBigWindow = Raylib.GetScreenWidth() > Settings.ScreenSizeSmall.X;
      string windowButtonName = isBigWindow ? "Smaller Window" : "Bigger Window";
      if (NextButtonInRow(windowButtonName, ref buttonPos, spacing, buttonSize))
      {
        Program.SetWindowSize(isBigWindow ? Settings.ScreenSizeSmall : Settings.ScreenSizeBig);
      }
      if (NextButtonInRow("Exit (ESC)", ref buttonPos, spacing, buttonSize))
      {
        Environment.Exit(0);
      }

      bool NextButtonInRow(string name, ref Vector2 pos, float spacingY, Vector2 size)
      {
        bool pressed = UIHelper.Button(name, pos, size);
        pos.Y += spacingY;
        return pressed;
      }
    }
  }
}