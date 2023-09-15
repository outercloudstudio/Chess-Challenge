using Raylib_cs;
using System.Numerics;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ChessChallenge.Application
{
  public static class MenuUI
  {
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
      if (NextButtonInRow("Human vs Theseus", ref buttonPos, spacing, buttonSize))
      {
        var whiteType = controller.HumanWasWhiteLastGame ? ChallengeController.PlayerType.Theseus : ChallengeController.PlayerType.Human;
        var blackType = !controller.HumanWasWhiteLastGame ? ChallengeController.PlayerType.Theseus : ChallengeController.PlayerType.Human;
        controller.StartNewGame(whiteType, blackType);
      }
      if (NextButtonInRow("My Bot vs My Bot", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.MyBot);
      }
      if (NextButtonInRow("My Bot vs Evil Bot", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.EvilBot);
      }
      if (NextButtonInRow("My Bot vs My Bot Evil", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.MyBotEvil);
      }
      if (NextButtonInRow("My Bot vs Tier2", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.Tier2);
      }
      if (NextButtonInRow("My Bot vs FrederoxQ", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.FrederoxQ);
      }
      if (NextButtonInRow("v T2", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.EloBot2);
      }
      if (NextButtonInRow("v Tyrant", ref buttonPos, spacing, buttonSize))
      {
        controller.StartNewBotMatch(ChallengeController.PlayerType.MyBot, ChallengeController.PlayerType.Tyrant);
      }

      buttonPos = UIHelper.Scale(new Vector2(390, 100));


      // Page buttons
      if (NextButtonInRow("Random PSQT Test", ref buttonPos, spacing, buttonSize))
      {
        int[] targets = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        for (int seed = 0; ; seed++)
        {
          Random random = new Random(seed);

          if (seed % 10000 == 0) Console.WriteLine($"Trying seed {seed}");

          bool failed = false;

          for (int index = 0; index < 16; index++)
          {
            int item = random.Next(0, targets.Length);

            if (targets[index] != item)
            {
              failed = true;

              continue;
            }
          }

          if (failed) continue;

          Console.WriteLine($"Found Seed {seed}");

          break;
        }
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