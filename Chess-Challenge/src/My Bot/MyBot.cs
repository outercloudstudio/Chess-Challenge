using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  Board _board;

  class Node
  {
    public int Score;
    public int Depth;
    public Move Move;
    public Node[] ChildNodes;
    public bool WhiteMove;

    MyBot _bot;

    public Node(Move move, bool whiteMove, MyBot bot)
    {
      Move = move;
      WhiteMove = whiteMove;
      _bot = bot;
    }

    public void Expand()
    {
      // Console.WriteLine("Expanding node with " + Move.ToString());

      _bot._board.MakeMove(Move);

      if (ChildNodes != null)
      {
        ChildNodes.MinBy(node => node.Depth).Expand();

        Depth = ChildNodes.Min(node => node.Depth) + 1;
      }
      else
      {
        ChildNodes = _bot._board.GetLegalMoves().Select(move => new Node(move, !WhiteMove, _bot)).ToArray();

        foreach (Node node in ChildNodes) node.Simulate(3);

        Depth = 1;
      }

      if (WhiteMove)
      {
        Score = ChildNodes.Min(node => node.Score);
      }
      else
      {
        Score = ChildNodes.Max(node => node.Score);
      }

      _bot._board.UndoMove(Move);
    }

    public void Simulate(int moves)
    {
      // Console.WriteLine("Simulating node with " + Move.ToString());

      _bot._board.MakeMove(Move);

      List<Move> simulatedMoves = new List<Move>();

      for (int i = 0; i < moves; i++)
      {
        Move[] legalMoves = _bot._board.GetLegalMoves();

        if (legalMoves.Length == 0) break;

        Move nextMove = legalMoves[0];

        if (WhiteMove)
        {
          nextMove = legalMoves.MinBy(Evaluate);
        }
        else
        {
          nextMove = legalMoves.MaxBy(Evaluate);
        }

        // Console.WriteLine("Next simulated " + nextMove);

        simulatedMoves.Add(nextMove);

        _bot._board.MakeMove(nextMove);
      }

      Score = Evaluate();

      for (int i = simulatedMoves.Count - 1; i >= 0; i--)
      {
        _bot._board.UndoMove(simulatedMoves[i]);
      }

      _bot._board.UndoMove(Move);
    }

    public int Evaluate()
    {
      int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 0 };

      int score = 0;

      for (int typeIndex = 1; typeIndex < 7; typeIndex++)
      {
        score += _bot._board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
        score -= _bot._board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
      }

      return score;
    }

    public int Evaluate(Move move)
    {
      _bot._board.MakeMove(move);

      int score = 0;

      _bot._board.UndoMove(move);

      return score;
    }

    public void Debug(int maxDepth = 2, int depth = 0)
    {
      if (depth > maxDepth) return;

      Console.WriteLine(new string('\t', depth) + "Node with " + Move.ToString() + " has score " + Score + " and depth " + Depth + " and white move " + WhiteMove);

      if (ChildNodes != null) foreach (Node node in ChildNodes) node.Debug(maxDepth, depth + 1);
    }
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    Node rootNode = new Node(Move.NullMove, !board.IsWhiteToMove, this);

    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60) rootNode.Expand();

    rootNode.Debug(0);

    if (board.IsWhiteToMove)
    {
      return rootNode.ChildNodes.MaxBy(node => node.Score).Move;
    }
    else
    {
      return rootNode.ChildNodes.MinBy(node => node.Score).Move;
    }
  }
}