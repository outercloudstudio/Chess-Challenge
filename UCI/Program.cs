using ChessChallenge.Application;

namespace Chess_Challenge.Cli
{
  internal class Program
  {
    static (int, int) GetTokenCount()
    {
      string path = Path.Combine(Directory.GetCurrentDirectory(), "src", "My Bot", "MyBot.cs");
      string txt = File.ReadAllText(path);
      return TokenCounter.CountTokens(txt);
    }

    static void Main(string[] args)
    {
      var uci = new Uci(args);
      uci.Run();
    }
  }
}