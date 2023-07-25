using System.Diagnostics;
using ChessChallenge.Chess;

public class Evaluation
{
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
}