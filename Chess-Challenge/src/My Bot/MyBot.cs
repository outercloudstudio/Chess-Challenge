using System;
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  record struct TranspositionEntry(ulong Hash, int Depth, int LowerBound, int UpperBound, Move BestMove);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  int[,,] _historyTable;

  Board _board;
  Move _bestMove;
  bool _white;
  Timer _timer;
  bool _initialSearch;

  int nodesSearched = 0;

  int[] phase_weight = { 0, 1, 1, 2, 4, 0 };
  int[] pieceValues = { 82, 337, 365, 477, 1025, 20000, 94, 281, 297, 512, 936, 20000 };

  decimal[] packedPieceTables = {
    63746705523041458768562654720m, 71818693703096985528394040064m, 75532537544690978830456252672m, 75536154932036771593352371712m, 76774085526445040292133284352m, 3110608541636285947269332480m, 936945638387574698250991104m, 75531285965747665584902616832m,
    77047302762000299964198997571m, 3730792265775293618620982364m, 3121489077029470166123295018m, 3747712412930601838683035969m, 3763381335243474116535455791m, 8067176012614548496052660822m, 4977175895537975520060507415m, 2475894077091727551177487608m,
    2458978764687427073924784380m, 3718684080556872886692423941m, 4959037324412353051075877138m, 3135972447545098299460234261m, 4371494653131335197311645996m, 9624249097030609585804826662m, 9301461106541282841985626641m, 2793818196182115168911564530m,
    77683174186957799541255830262m, 4660418590176711545920359433m, 4971145620211324499469864196m, 5608211711321183125202150414m, 5617883191736004891949734160m, 7150801075091790966455611144m, 5619082524459738931006868492m, 649197923531967450704711664m,
    75809334407291469990832437230m, 78322691297526401047122740223m, 4348529951871323093202439165m, 4990460191572192980035045640m, 5597312470813537077508379404m, 4980755617409140165251173636m, 1890741055734852330174483975m, 76772801025035254361275759599m,
    75502243563200070682362835182m, 78896921543467230670583692029m, 2489164206166677455700101373m, 4338830174078735659125311481m, 4960199192571758553533648130m, 3420013420025511569771334658m, 1557077491473974933188251927m, 77376040767919248347203368440m,
    73949978050619586491881614568m, 77043619187199676893167803647m, 1212557245150259869494540530m, 3081561358716686153294085872m, 3392217589357453836837847030m, 1219782446916489227407330320m, 78580145051212187267589731866m, 75798434925965430405537592305m,
    68369566912511282590874449920m, 72396532057599326246617936384m, 75186737388538008131054524416m, 77027917484951889231108827392m, 73655004947793353634062267392m, 76417372019396591550492896512m, 74568981255592060493492515584m, 70529879645288096380279255040m,
  };

  int[][] pieceTables;

  public MyBot()
  {
    pieceTables = new int[64][];
    pieceTables = packedPieceTables.Select(packedTable =>
    {
      int pieceType = 0;
      return decimal.GetBits(packedTable).Take(3)
        .SelectMany(c => BitConverter.GetBytes(c)
          .Select((byte square) => (int)((sbyte)square * 1.461) + pieceValues[pieceType++]))
        .ToArray();
    }).ToArray();
  }

  bool hasTime => _timer.MillisecondsElapsedThisTurn < _timer.MillisecondsRemaining / 60;

  int Interest(Move move, Move bestHashMove)
  {
    if (move == bestHashMove) return 99999999;

    if (move.IsCapture) return pieceValues[(int)move.CapturePieceType - 1] - pieceValues[(int)move.MovePieceType - 1] / 100;

    return _historyTable[_board.IsWhiteToMove ? 0 : 1, (int)move.MovePieceType - 1, move.TargetSquare.Index];
  }

  int Evaluate()
  {
    int middleGame = 0, endGame = 0, phase = 0;

    foreach (bool white in new[] { true, false })
    {
      for (int piece = -1; ++piece < 6;)
      {
        ulong bitBoard = _board.GetPieceBitboard((PieceType)(piece + 1), white);

        while (bitBoard != 0)
        {
          int sq = BitboardHelper.ClearAndGetIndexOfLSB(ref bitBoard) ^ (white ? 56 : 0);

          middleGame += pieceTables[sq][piece];
          endGame += pieceTables[sq][piece + 6];

          phase += phase_weight[piece];
        }
      }
      middleGame = -middleGame;
      endGame = -endGame;
    }

    phase = Math.Min(phase, 24);

    return (middleGame * phase + endGame * (24 - phase)) / 24 * (_board.IsWhiteToMove ? 1 : -1);
  }

  record struct OrderedMove(Move Move, int Interest);

  int AlphaBetaWM(int lowerBound, int upperBound, int ply, int depth, bool qSearch)
  {
    nodesSearched++;

    Move bestMove = Move.NullMove;

    ulong hash = _board.ZobristKey;
    ulong key = hash % 100000L;
    TranspositionEntry transpositionEntry = _transpositionTable[key];

    if (!qSearch && transpositionEntry.Depth > 0 && transpositionEntry.Hash == hash)
    {
      if (transpositionEntry.Depth >= depth)
      {
        if (transpositionEntry.LowerBound >= upperBound) return transpositionEntry.LowerBound;
        if (transpositionEntry.UpperBound <= lowerBound) return transpositionEntry.UpperBound;

        lowerBound = Math.Max(lowerBound, transpositionEntry.LowerBound);
        upperBound = Math.Min(upperBound, transpositionEntry.UpperBound);

        if (ply == 0) _bestMove = transpositionEntry.BestMove;
      }

      bestMove = transpositionEntry.BestMove;
    }

    int max = -99999999 + ply;

    if (depth <= 0 && !qSearch) return Evaluate();
    else
    {
      Move[] moves = _board.GetLegalMoves(qSearch);

      // if (_board.IsRepeatedPosition() || _board.IsFiftyMoveDraw()) return _white == _board.IsWhiteToMove ? -5 : 5;

      if (qSearch)
      {
        if (moves.Length == 0) return Evaluate();

        int standingScore = Evaluate();

        if (standingScore >= upperBound) return standingScore;

        if (standingScore > lowerBound) lowerBound = standingScore;
      }

      var orderedMoves = moves.Select(move => new OrderedMove(move, Interest(move, bestMove))).OrderByDescending(orderedMove => orderedMove.Interest);

      foreach (OrderedMove orderedMove in orderedMoves)
      {
        if (!_initialSearch && !hasTime) break;

        Move move = orderedMove.Move;

        _board.MakeMove(move);

        int score = -AlphaBetaWM(-upperBound, -lowerBound, ply + 1, depth - 1, depth <= 1 && move.IsCapture);

        _board.UndoMove(move);

        if (score >= upperBound)
        {
          bestMove = move;

          max = score;

          upperBound = score;

          if (!move.IsCapture)
            _historyTable[_board.IsWhiteToMove ? 0 : 1, (int)move.MovePieceType - 1, move.TargetSquare.Index] += depth * depth;

          break;
        }

        if (score > max)
        {
          max = score;

          bestMove = move;

          if (score > lowerBound) lowerBound = score;
        }
      }
    }

    if (!qSearch && depth >= transpositionEntry.Depth) _transpositionTable[key] = new TranspositionEntry(hash, depth, lowerBound, upperBound, bestMove);

    if (ply == 0) _bestMove = bestMove;

    return max;
  }

  int MTDF(int initialGuess, int depth)
  {
    int max = initialGuess;

    int upperBound = 99999999;
    int lowerBound = -99999999;

    while (lowerBound < upperBound)
    {
      int beta = max;
      if (max == lowerBound) beta++;

      max = AlphaBetaWM(beta - 1, beta, 0, depth, false);

      if (max < beta) upperBound = max;
      else lowerBound = max;
    }

    return max;
  }

  int bestMoveGuess = 0;

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _white = board.IsWhiteToMove;
    _timer = timer;

    _historyTable = new int[2, 6, 64];

    int depth = 1;

    while (_initialSearch || hasTime)
    {
      _initialSearch = depth == 1;

      Move lastBestMove = _bestMove;

      bestMoveGuess = MTDF(bestMoveGuess, depth);

      if (!_initialSearch && !hasTime)
      {
        _bestMove = lastBestMove;

        break;
      }

      Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms"); // #DEBUG

      depth++;
    }

    depth--;

    Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms {nodesSearched / (timer.MillisecondsElapsedThisTurn / (float)1000)} nodes/sec"); // #DEBUG

    return _bestMove;
  }
}