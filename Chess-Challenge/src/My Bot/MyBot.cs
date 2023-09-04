using System;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  int _positionsSearched = 0;

  Board _board;

  int[] pieceValues = new int[] { 1, 3, 3, 5, 9, 10, -1, -3, -3, -5, -9, -10 };

  public int Evaluate()
  {
    int score = 0;

    var lists = _board.GetAllPieceLists();

    for (int index = 0; index < lists.Length; index++) score += pieceValues[index] * lists[index].Count;

    return score * (_board.IsWhiteToMove ? 1 : -1);
  }

  public class Node
  {
    public Move Move;
    public int Score;
    public Node[] Children;
    public ulong Hash;
    public int Depth;
  }

  Node[] _transpositionTable = new Node[100000];

  public void Search(Node node, int depth, int lowerBound, int upperBound)
  {
    _positionsSearched++;

    if (node.Depth >= depth) return;

    if (depth == 0)
    {
      node.Score = Evaluate();

      return;
    }

    var moves = _board.GetLegalMoves();

    if (moves.Length == 0)
    {
      node.Score = _board.IsInCheck() ? -999999999 : 0;

      return;
    }

    var children = new Node[moves.Length];

    int max = -999999999;

    for (int moveIndex = 0; moveIndex < moves.Length; moveIndex++)
    {
      Move move = moves[moveIndex];

      _board.MakeMove(move);

      ulong hash = _board.ZobristKey;

      Node child = new Node() { Move = move, Hash = hash };

      Node transposition = _transpositionTable[hash % 100000];

      if (transposition != null && transposition.Hash == hash) child = transposition;

      Search(child, depth - 1, -upperBound, -lowerBound);

      _transpositionTable[child.Hash % 100000] = child;

      _board.UndoMove(move);

      node.Depth = Math.Max(node.Depth, child.Depth + 1);

      int score = -child.Score;

      children[moveIndex] = child;

      if (score > upperBound)
      {
        max = score;

        break;
      }

      if (score > max)
      {
        max = score;

        lowerBound = Math.Max(lowerBound, score);
      }
    }

    node.Score = max;

    node.Children = children;
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;

    Node root = new Node() { Move = Move.NullMove, Score = Evaluate() };

    int depth = 1;

    while (true)
    {
      _positionsSearched = 0;

      Search(root, depth, -999999999, 999999999);

      string positionsPerSecond = timer.MillisecondsElapsedThisTurn == 0 ? "a lot" : (_positionsSearched / timer.MillisecondsElapsedThisTurn).ToString();

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms positions: {_positionsSearched} pps: {positionsPerSecond}"); // #DEBUG

      if (timer.MillisecondsElapsedThisTurn > timer.MillisecondsRemaining / 120f) break;

      depth++;
    }

    return root.Children.MaxBy(child => child != null ? -child.Score : -999999999).Move;
  }
}