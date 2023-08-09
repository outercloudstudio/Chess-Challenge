using System;
using System.Collections.Generic;
using System.Linq;
using ChessChallenge.API;

public class Classic : IChessBot
{
  Board _board;
  Timer _timer;
  Move _bestMove;
  int _searchedMoves;
  bool _cancelledSearchEarly;
  long outOfTimeSum = 0;
  int outOfTimeCount = 0;
  long transpositionSum = 0;
  int transpositionCount = 0;
  long evalSum = 0;
  int evalCount = 0;
  long moveOrderSum = 0;
  int moveOrderCount = 0;
  long transpositionSetSum = 0;
  int transpositionSetCount = 0;
  // System.Text.StringBuilder _log = new System.Text.StringBuilder("Search:\n"); // #DEBUG

  record struct TranspositionEntry(ulong Hash, int Depth, int Score, int Bound);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  int Search(int depth, int ply, int alpha, int beta, bool isLoud, bool initial)
  {
    System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
    stopwatch.Start();

    if (!initial && _timer.MillisecondsElapsedThisTurn > _timer.MillisecondsRemaining / 60)
    {
      _cancelledSearchEarly = true;

      return 0;
    }

    outOfTimeSum += stopwatch.ElapsedTicks;
    outOfTimeCount++;
    stopwatch.Restart();

    bool qSearch = isLoud && depth <= 0;

    _searchedMoves++;

    ulong hash = _board.ZobristKey;
    int key = (int)(hash % 100000);
    TranspositionEntry entry = _transpositionTable[key];

    if (entry.Depth > 0 && entry.Depth >= depth && entry.Hash == hash && !qSearch)
    {
      if (entry.Bound == 0) return entry.Score;

      if (entry.Bound == 1) alpha = Math.Max(alpha, entry.Score);

      if (entry.Bound == -1) beta = Math.Min(beta, entry.Score);

      if (alpha >= beta) return entry.Score;
    }

    transpositionSum += stopwatch.ElapsedTicks;
    transpositionCount++;
    stopwatch.Restart();

    if (depth <= 0 && !(qSearch && ply < 12)) return Evaluate();

    evalSum += stopwatch.ElapsedTicks;
    evalCount++;
    stopwatch.Restart();

    Move[] moves = _board.GetLegalMoves();
    int[] interest = new int[moves.Length];

    moveOrderSum += stopwatch.ElapsedTicks;
    moveOrderCount++;
    stopwatch.Restart();

    if (moves.Length == 0) return Evaluate();

    if (ply == 0) _bestMove = moves[0];

    for (int moveIndex = 0; moveIndex < moves.Length; moveIndex++) interest[moveIndex] = Interest(moves[moveIndex]);

    int max = -999999995;

    for (int moveIndex = 0; moveIndex < moves.Length; moveIndex++)
    {
      for (int otherMoveIndex = moveIndex + 1; otherMoveIndex < moves.Length; otherMoveIndex++)
      {
        if (interest[otherMoveIndex] > interest[moveIndex])
          (interest[moveIndex], interest[otherMoveIndex], moves[moveIndex], moves[otherMoveIndex]) = (interest[otherMoveIndex], interest[moveIndex], moves[otherMoveIndex], moves[moveIndex]);
      }

      Move move = moves[moveIndex];

      _board.MakeMove(move);

      // _log.AppendLine(new string('\t', ply * 2 + 1) + "- " + move + ":"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Alpha: {alpha}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Beta: {beta}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Loud: {isLoud}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Depth: {depth}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Other Moves: {moveChoices.Length}"); // #DEBUG
      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Children:"); // #DEBUG

      int score;

      if (ply == 3)
      {
        score = -Search(depth - 3, ply + 1, -beta, -alpha, move.IsCapture, initial);

        if (score >= beta) score = -Search(depth - 1, ply + 1, -beta, -alpha, move.IsCapture, initial);
      }
      else
      {
        score = -Search(depth - 1, ply + 1, -beta, -alpha, move.IsCapture, initial);
      }

      // _log.AppendLine(new string('\t', ply * 2 + 2) + $"Score: {score}"); // #DEBUG

      _board.UndoMove(move);

      if (score >= beta)
      {
        if (ply == 0) _bestMove = move;

        if (depth > entry.Depth) _transpositionTable[key] = new TranspositionEntry(hash, depth, max, 1);

        return score;
      }

      if (score > max)
      {
        max = score;

        if (ply == 0) _bestMove = move;

        if (score > alpha) alpha = score;
      };
    }

    stopwatch.Restart();
    transpositionSetSum += stopwatch.ElapsedTicks;
    transpositionSetCount++;

    if (depth > entry.Depth && !_cancelledSearchEarly) _transpositionTable[key] = new TranspositionEntry(hash, depth, max, max <= alpha ? -1 : 0);

    return max;
  }

  int[] pieceValues = new int[] { 0, 100, 300, 300, 500, 900, 10000 };
  int[] piecePhases = { 0, 0, 1, 1, 2, 4, 0 };
  ulong[] pieceSquareTables = { 657614902731556116, 420894446315227099, 384592972471695068, 312245244820264086, 364876803783607569, 366006824779723922, 366006826859316500, 786039115310605588, 421220596516513823, 366011295806342421, 366006826859316436, 366006896669578452, 162218943720801556, 440575073001255824, 657087419459913430, 402634039558223453, 347425219986941203, 365698755348489557, 311382605788951956, 147850316371514514, 329107007234708689, 402598430990222677, 402611905376114006, 329415149680141460, 257053881053295759, 291134268204721362, 492947507967247313, 367159395376767958, 384021229732455700, 384307098409076181, 402035762391246293, 328847661003244824, 365712019230110867, 366002427738801364, 384307168185238804, 347996828560606484, 329692156834174227, 365439338182165780, 386018218798040211, 456959123538409047, 347157285952386452, 365711880701965780, 365997890021704981, 221896035722130452, 384289231362147538, 384307167128540502, 366006826859320596, 366006826876093716, 366002360093332756, 366006824694793492, 347992428333053139, 457508666683233428, 329723156783776785, 329401687190893908, 366002356855326100, 366288301819245844, 329978030930875600, 420621693221156179, 422042614449657239, 384602117564867863, 419505151144195476, 366274972473194070, 329406075454444949, 275354286769374224, 366855645423297932, 329991151972070674, 311105941360174354, 256772197720318995, 365993560693875923, 258219435335676691, 383730812414424149, 384601907111998612, 401758895947998613, 420612834953622999, 402607438610388375, 329978099633296596, 67159620133902 };

  int ColorEvaluationFactor(bool white) => white ? 1 : -1;

  int Interest(Move move)
  {
    if (move == _bestMove) return 999;

    if (move.IsCapture) return pieceValues[(int)move.CapturePieceType] - pieceValues[(int)move.MovePieceType] / 100;

    return 0;
  }

  public int getPieceSquareTableValue(int index)
  {
    return (int)(((pieceSquareTables[index / 10] >> (6 * (index % 10))) & 63) - 20) * 8;
  }

  int Evaluate()
  {
    if (_board.IsInCheckmate()) return -999993;

    if (_board.IsDraw()) return 0;

    int middleGameEvaluation = 0;
    int endGameEvaluation = 0;
    int phase = 0;

    foreach (bool white in new[] { true, false })
    {
      for (var pieceType = PieceType.Pawn; pieceType <= PieceType.King; pieceType++)
      {
        int piece = (int)pieceType;
        ulong mask = _board.GetPieceBitboard(pieceType, white);

        while (mask != 0)
        {
          phase += piecePhases[piece];

          int index = 128 * (piece - 1) + BitboardHelper.ClearAndGetIndexOfLSB(ref mask) ^ (white ? 56 : 0);

          middleGameEvaluation += getPieceSquareTableValue(index) + pieceValues[piece];
          endGameEvaluation += getPieceSquareTableValue(index + 64) + pieceValues[piece];
        }
      }

      middleGameEvaluation = -middleGameEvaluation;
      endGameEvaluation = -endGameEvaluation;
    }

    return (middleGameEvaluation * phase + endGameEvaluation * (24 - phase)) / 24 * ColorEvaluationFactor(_board.IsWhiteToMove);
  }

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    _cancelledSearchEarly = false;

    int depth = 2;
    int bestMoveScore = 0;
    Move lastSearchBestMove = Move.NullMove;
    while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining / 60 || depth == 2)
    {
      // _log = new System.Text.StringBuilder("Search:\n"); // #DEBUG

      int score = Search(depth, 0, bestMoveScore - 100, bestMoveScore + 100, false, depth == 2);

      if (score <= bestMoveScore - 100 || score >= bestMoveScore + 100)
      {
        bestMoveScore = Search(depth, 0, -999999991, 999999992, false, depth == 2);
      }
      else
      {
        bestMoveScore = score;
      }

      if (!_cancelledSearchEarly) lastSearchBestMove = _bestMove;

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn} ms, cancelled early {_cancelledSearchEarly}"); // #DEBUG

      // System.IO.File.WriteAllText(@"D:\Chess-Challenge\Chess-Challenge\log.yml", _log.ToString()); // #DEBUG

      depth++;
    }

    Console.WriteLine($"Searched {_searchedMoves / (float)timer.MillisecondsElapsedThisTurn} moves / sec");

    Console.WriteLine($"Average out of time {outOfTimeSum / (float)outOfTimeCount}");
    Console.WriteLine($"Average transposition {transpositionSum / (float)transpositionCount}");
    Console.WriteLine($"Average eval {evalSum / (float)evalCount}");
    Console.WriteLine($"Average move order {moveOrderSum / (float)moveOrderCount}");
    Console.WriteLine($"Average transposition {transpositionSetSum / (float)transpositionSetCount}");


    return lastSearchBestMove;
  }
}