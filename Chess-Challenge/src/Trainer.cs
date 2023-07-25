using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using ChessChallenge.Chess;

public class Trainer
{
  private TcpListener? _server;
  private TcpClient? _client;
  private NetworkStream? _stream;
  private Thread _listenThread;

  public void StartServer()
  {
    Console.WriteLine("Starting server...");

    _server = new TcpListener(IPAddress.Parse("127.0.0.1"), 8080);
    _server.Start();

    _listenThread = new Thread(new ThreadStart(Listen));
    _listenThread.Start();
  }

  public void Listen()
  {
    Console.WriteLine("Waiting for a connection... ");

    _client = _server.AcceptTcpClient();
    _stream = _client.GetStream();

    Console.WriteLine("Connected!");

    while (true)
    {
      if (_server == null) return;

      byte[] bytes = new byte[1024];

      int bytesRead;
      while ((bytesRead = _stream.Read(bytes, 0, bytes.Length)) != 0)
      {
        string message = Encoding.UTF8.GetString(bytes[..bytesRead]);

        Console.WriteLine("Received: " + message);

        byte[] sendMessage = Encoding.UTF8.GetBytes("Hello Client!");

        _stream.Write(sendMessage, 0, sendMessage.Length);

        bytes = new byte[1024];
      }
    }
  }

  public void StopServer()
  {
    if (_server == null) return;

    TcpListener? server = _server;

    _server = null;

    _stream.Close();
    _client.Close();
    server.Stop();

    _stream = null;
    _client = null;
  }

  public static void GenerateDataset()
  {
    Process process = Evaluation.CreateEvaluationProcess();

    string[] fens = File.ReadAllLines("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Fens\\Positions Medium.txt");

    string output = "";

    int index = 0;

    foreach (string fen in fens)
    {
      Board board = new Board();
      board.LoadPosition(fen);

      float evaluation = Evaluation.Evaluate(process, 5, board);

      output += fen + " | " + evaluation + "\n";

      Console.WriteLine("Evaluated " + index + " / " + fens.Length);

      index++;

      if (index % 1000 == 0)
      {
        Evaluation.EndEvaluationProcess(process);
        process = Evaluation.CreateEvaluationProcess();
      }
    }

    Evaluation.EndEvaluationProcess(process);

    File.WriteAllText("D:\\Chess-Challenge\\Chess-Challenge\\src\\Training\\Datasets\\Evaluations Medium.txt", output[..(output.Length - 1)]);
  }
}
