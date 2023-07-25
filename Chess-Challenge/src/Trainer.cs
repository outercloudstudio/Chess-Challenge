using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using ChessChallenge.Application;
using ChessChallenge.Chess;

public class Trainer
{
  private TcpListener? _server;
  private TcpClient? _client;
  private NetworkStream? _stream;
  private Thread _listenThread;

  private ChallengeController _controller;
  private BoardUI _boardUI;

  public Trainer(ChallengeController controller)
  {
    _controller = controller;
    _boardUI = controller.boardUI;
  }

  public void StartServer()
  {
    Console.WriteLine("Starting server...");

    _server = new TcpListener(IPAddress.Parse("127.0.0.1"), 8080);
    _server.Start();

    _listenThread = new Thread(new ThreadStart(Listen));
    _listenThread.Start();
  }

  private string ReadString()
  {
    byte[] lengthBytes = new byte[4];
    _stream.Read(lengthBytes, 0, lengthBytes.Length);
    int length = BitConverter.ToInt32(lengthBytes, 0);

    byte[] bytes = new byte[length];
    _stream.Read(bytes, 0, bytes.Length);

    return Encoding.UTF8.GetString(bytes, 0, length);
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

      string fen = ReadString();
      string uci = ReadString();

      SendMoveState(fen, uci);
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

  List<int> _PieceValues = new List<int>() { 0, 1, 3, 3, 5, 9, 0 };

  public int GetMaterial(ChessChallenge.API.Board board, bool white)
  {
    int material = 0;

    for (int squareIndex = 0; squareIndex < 64; squareIndex++)
    {
      ChessChallenge.API.Square square = new ChessChallenge.API.Square(squareIndex);
      ChessChallenge.API.Piece piece = board.GetPiece(square);

      if (piece.IsWhite == white) material += _PieceValues[(int)piece.PieceType];
    }

    return material;
  }

  public void SendMoveState(string fen, string moveUci)
  {
    Board board = new Board();
    board.LoadPosition(fen);

    _controller.PlayerWhite = new ChessPlayer(null, ChallengeController.PlayerType.ARCNET, -1);
    _controller.PlayerBlack = new ChessPlayer(null, ChallengeController.PlayerType.ARCNET, -1);
    _boardUI.ResetSquareColours();
    _boardUI.SetPerspective(true);
    _boardUI.UpdatePosition(board);

    ChessChallenge.API.Board apiBoard = new ChessChallenge.API.Board(board);
    ChessChallenge.API.Move apiMove = new ChessChallenge.API.Move(moveUci, apiBoard);

    _stream.Write(BitConverter.GetBytes(GetMaterial(apiBoard, true)), 0, 4);
    _stream.Write(BitConverter.GetBytes(GetMaterial(apiBoard, false)), 0, 4);

    apiBoard.MakeMove(apiMove);

    _stream.Write(BitConverter.GetBytes(GetMaterial(apiBoard, true)), 0, 4);
    _stream.Write(BitConverter.GetBytes(GetMaterial(apiBoard, false)), 0, 4);

    _stream.Write(BitConverter.GetBytes(apiBoard.GameMoveHistory.Length), 0, 4);

    _stream.Write(BitConverter.GetBytes(apiBoard.GetKingSquare(true).File), 0, 4);
    _stream.Write(BitConverter.GetBytes(apiBoard.GetKingSquare(true).Rank), 0, 4);
    _stream.Write(BitConverter.GetBytes(apiBoard.GetKingSquare(false).File), 0, 4);
    _stream.Write(BitConverter.GetBytes(apiBoard.GetKingSquare(false).Rank), 0, 4);

    _stream.Write(BitConverter.GetBytes(apiMove.IsCapture), 0, 1);
    _stream.Write(BitConverter.GetBytes(apiBoard.IsDraw()), 0, 1);
    _stream.Write(BitConverter.GetBytes(apiBoard.IsInCheck()), 0, 1);

    _stream.Write(BitConverter.GetBytes(_PieceValues[(int)apiMove.MovePieceType]), 0, 4);

    _stream.Write(BitConverter.GetBytes(apiMove.StartSquare.File), 0, 4);
    _stream.Write(BitConverter.GetBytes(apiMove.StartSquare.Rank), 0, 4);

    _stream.Write(BitConverter.GetBytes(apiMove.TargetSquare.File), 0, 4);
    _stream.Write(BitConverter.GetBytes(apiMove.TargetSquare.Rank), 0, 4);

    apiBoard.UndoMove(apiMove);
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
