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

    MyBot _bot;

    public Node(Move move, MyBot bot)
    {
      Move = move;
      _bot = bot;
    }

    public void Expand()
    {
      _bot._board.MakeMove(Move);

      if (ChildNodes != null)
      {
        ChildNodes.MinBy(node => node.Depth).Expand();

        Depth = ChildNodes.Min(node => node.Depth) + 1;
      }
      else
      {
        ChildNodes = _bot._board.GetLegalMoves().Select(move => new Node(move, _bot)).ToArray();

        foreach (Node node in ChildNodes) node.Simulate(3);

        Depth = 1;
      }

      if (ChildNodes.Length == 0)
      {
        Depth = 999999999;

        _bot._board.UndoMove(Move);

        return;
      }

      Score = ChildNodes.Min(node => -node.Score);

      _bot._board.UndoMove(Move);
    }

    public void Simulate(int moves)
    {
      _bot._board.MakeMove(Move);

      var simulatedMoves = new List<Move>();

      for (int i = 0; i < moves; i++)
      {
        Move[] legalMoves = _bot._board.GetLegalMoves();

        if (legalMoves.Length == 0) break;

        Move nextMove = legalMoves[0];

        nextMove = legalMoves.MaxBy(Evaluate);

        simulatedMoves.Add(nextMove);

        _bot._board.MakeMove(nextMove);
      }

      Score = Evaluate() * (moves % 2 == 0 ? -1 : 1);

      simulatedMoves.Reverse();

      foreach (Move move in simulatedMoves) _bot._board.UndoMove(move);

      _bot._board.UndoMove(Move);
    }

    public int Evaluate()
    {
      if (_bot._board.IsInCheckmate()) return -999999999;

      int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 0 };

      int score = 0;

      for (int typeIndex = 1; typeIndex < 7; typeIndex++)
      {
        score += _bot._board.GetPieceList((PieceType)typeIndex, true).Count * pieceValues[typeIndex];
        score -= _bot._board.GetPieceList((PieceType)typeIndex, false).Count * pieceValues[typeIndex];
      }

      return score * (_bot._board.IsWhiteToMove ? 1 : -1);
    }

    public int Evaluate(Move move)
    {
      _bot._board.MakeMove(move);

      int score = 0;

      _bot._board.UndoMove(move);

      return score;
    }

    public void Debug(int maxDepth = 2, int depth = 0) // #DEBUG
    {
      if (depth > maxDepth) return; // #DEBUG

      Console.WriteLine(new string('\t', depth) + "Node with " + Move.ToString() + " has score " + Score + " and depth " + Depth); // #DEBUG

      if (ChildNodes != null) foreach (Node node in ChildNodes) node.Debug(maxDepth, depth + 1); // #DEBUG
    } // #DEBUG
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    Node rootNode = new Node(Move.NullMove, this);

    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60) rootNode.Expand();

    rootNode.Debug(1); // #DEBUG

    return rootNode.ChildNodes.MaxBy(node => node.Score).Move;
  }
}