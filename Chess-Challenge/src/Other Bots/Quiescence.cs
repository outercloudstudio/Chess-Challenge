using ChessChallenge.API;
using Frederox.AlphaBeta;
using System;
using System.Linq;

// Things to do:
// - PeSTO piece values
// - PeSTO Tables
// - Iterative deepening
namespace Frederox.Quiescence
{
  public class Quiescence : IChessBot
  {
    // 256 * 1024 * 1024 = 268435456 bits
    // 268435456 / 112 (Sizeof TEntry) = 2396745
    TEntry[] transpositions = new TEntry[2396745];

    // Using an array is probably better than dict
    // index can be created 

    int positionsEvaluated;

    int negativeInfinity = -10000000;
    Move bestMove;
    byte mDepth = 5;

    public Move Think(Board board, Timer timer)
    {
      if (board.PlyCount == 0 && board.IsWhiteToMove) return new Move("d2d4", board);
      positionsEvaluated = 0;
      bestMove = Move.NullMove;

      int score = Negamax(board, mDepth, negativeInfinity, -negativeInfinity, board.IsWhiteToMove);
      //Console.WriteLine($"Depth: {mDepth}, Evaluated: {positionsEvaluated}, {bestMove}, score: {score}");

      // The bot occasionally doesnt have a move weird bug...
      if (bestMove == Move.NullMove)
      {
        //Console.WriteLine("NULL MOVE!");
        return board.GetLegalMoves().OrderByDescending(move => ScoreMovePotential(move)).ToArray()[0];
      }

      return bestMove;
    }

    int Negamax(Board board, byte depth, int alpha, int beta, bool asWhite)
    {
      if (board.IsInCheckmate()) return negativeInfinity;
      if (board.IsDraw()) return 0;

      int alphaOriginal = alpha;
      TEntry? ttEntry = getTransposition(board.ZobristKey);

      // (* Transposition Table Lookup; board.ZobristKey is the lookup key for ttEntry *)
      if (ttEntry != null && ttEntry.Depth >= depth)
      {
        // if ttEntry.flag = EXACT then
        if (ttEntry.Flag == 2) return ttEntry.Value;

        // else if ttEntry.flag = LOWERBOUND then
        else if (ttEntry.Flag == 0) alpha = Math.Max(alpha, ttEntry.Value);

        // else if ttEntry.flag = UPPERBOUND then
        else beta = Math.Min(beta, ttEntry.Value);

        // if α ≥ β then
        if (alpha >= beta) return ttEntry.Value;
      }

      positionsEvaluated++;

      // if depth = 0 or node is a terminal node then
      if (depth == 0) return Quiesce(board, alpha, beta);

      var moves = board.GetLegalMoves()
          .OrderByDescending(move => ScoreMovePotential(move));

      //for (int i = 0; i < moves.Length; i++)
      foreach (Move move in moves)
      {
        board.MakeMove(move);
        int moveScore = -Negamax(board, (byte)(depth - 1), -beta, -alpha, !asWhite);
        board.UndoMove(move);

        if (moveScore > alpha)
        {
          if (depth == mDepth) bestMove = move;
          if (moveScore >= beta) return beta;
          alpha = moveScore;
        }
      }

      // (* Transposition Table Store; board.ZobristKey is the lookup key for ttEntry *)

      // ttEntry.flag := UPPERBOUND
      byte flag = 2;

      // ttEntry.flag := UPPERBOUND
      if (alpha <= alphaOriginal) flag = 1;

      // ttEntry.flag := UPPERBOUND
      else if (alpha >= beta) flag = 0;

      setTransposition(board.ZobristKey, depth, alpha, flag);
      return alpha;
    }

    int Quiesce(Board board, int alpha, int beta)
    {
      positionsEvaluated++;
      if (board.IsInCheckmate()) return negativeInfinity;
      if (board.IsDraw()) return 0;

      int standPat = EvaluateHeuristicValue(board, board.IsWhiteToMove);
      if (standPat >= beta) return beta;
      if (alpha < standPat) alpha = standPat;

      Move[] legalCaptures = board.GetLegalMoves(true)
          .OrderByDescending(move => ScoreMovePotential(move))
          .ToArray();

      foreach (Move move in legalCaptures)
      {
        board.MakeMove(move);
        int score = -Quiesce(board, -beta, -alpha);
        board.UndoMove(move);

        if (score >= beta) return beta;
        alpha = Math.Max(alpha, score);
      }
      return alpha;
    }

    int ScoreMovePotential(Move move)
    {
      int scoreGuess = 0;

      // Prioritise taking high-value pieces with the lowest-value piece
      if (move.IsCapture)
        scoreGuess += 10 * GetPieceValue(move.CapturePieceType) - GetPieceValue(move.MovePieceType);

      // Promoting a pawn
      if (move.IsPromotion)
        scoreGuess += GetPieceValue(move.PromotionPieceType);

      return scoreGuess;
    }

    int GetPieceValue(PieceType type)
    {
      int[] values = { 0, 100, 300, 300, 500, 900, 10000 };
      return values[(int)type];
    }

    int EvaluateHeuristicValue(Board board, bool asWhite)
        => EvaluateSide(board, asWhite) - EvaluateSide(board, !asWhite);

    int EvaluateSide(Board board, bool asWhite)
        => board.GetPieceList(PieceType.Pawn, asWhite).Count * 100 +
            board.GetPieceList(PieceType.Knight, asWhite).Count * 300 +
            board.GetPieceList(PieceType.Bishop, asWhite).Count * 300 +
            board.GetPieceList(PieceType.Rook, asWhite).Count * 500 +
            board.GetPieceList(PieceType.Queen, asWhite).Count * 900 +
            board.GetPieceList(PieceType.King, asWhite).Count * 100000 +
            // Mobility (the number of legal moves)
            10 * board.GetLegalMoves().Length;

    TEntry? getTransposition(ulong zobristKey)
    {
      ///                                                 
      TEntry? entry = transpositions[(int)(zobristKey % 2080895)];

      // only risk is occasional data collisions which is why checking key
      if (entry != null && entry.ZobristKey == zobristKey) return entry;
      return null;
    }

    void setTransposition(ulong zobristKey, byte depth, int value, byte flag)
    {
      transpositions[(int)(zobristKey % 2396745)] = new TEntry
      {
        ZobristKey = zobristKey,
        Depth = depth,
        Value = value,
        Flag = flag
      };
    }
  }

  // Size 112 bits
  class TEntry
  {
    public ulong ZobristKey;
    public int Value;
    public byte Depth;
    /**
     * 0 Lowerbound, 1 Upperbound, 2 Exact
     */
    public byte Flag;
  }
}