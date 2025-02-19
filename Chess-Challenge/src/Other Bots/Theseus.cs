﻿using System;
using System.Linq;
using ChessChallenge.API;

public class Theseus : IChessBot
{
  // Bounds:
  // 0 = Exact
  // 1 = Lower, Never found a move greater than alpha
  // 2 = Upper, found a move better than oponent reposonses
  record struct TranspositionEntry(ulong Hash, int Score, int Bound, Move BestMove, int Depth = -1);
  TranspositionEntry[] _transpositionTable = new TranspositionEntry[400000];

  int[,,] _historyTable;

  Board _board;
  Move _bestMove;
  int _evaluation = 0;
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

  public Theseus()
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

  int Interest(Move move, Move bestMove)
  {
    if (move == bestMove) return 999999999;

    if (move.IsCapture) return 100 * pieceValues[(int)move.CapturePieceType - 1] - pieceValues[(int)move.MovePieceType - 1];

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

  int Search(int lowerBound, int upperBound, int ply, int depth, bool isLoud)
  {
    nodesSearched++;

    bool qSearch = depth <= 0 && isLoud;

    Move bestMove = Move.NullMove;

    ulong hash = _board.ZobristKey;
    ulong key = hash % 400000L;

    TranspositionEntry transpositionEntry = _transpositionTable[key];

    // Don't get transposition table if qSearch since we are going to search to an undefined depth
    if (!qSearch && transpositionEntry.Depth > -1 && transpositionEntry.Hash == hash)
    {
      bestMove = transpositionEntry.BestMove;

      if (depth <= transpositionEntry.Depth)
      {
        if (transpositionEntry.Bound == 0) return transpositionEntry.Score;
        if (transpositionEntry.Bound == 1 && transpositionEntry.Score >= upperBound) return transpositionEntry.Score;
        if (transpositionEntry.Bound == 2 && transpositionEntry.Score <= lowerBound) return transpositionEntry.Score;

        lowerBound = Math.Max(lowerBound, transpositionEntry.Score);
        upperBound = Math.Min(upperBound, transpositionEntry.Score);
      }
      // losing move prune (my custom idea)
      else if (_evaluation - transpositionEntry.Score > Math.Pow(50, depth - transpositionEntry.Depth)) return transpositionEntry.Score;
    }

    // we can't return cause we need to keep q searching
    if (!qSearch && depth <= 0) return Evaluate();

    if (qSearch)
    {
      int standingEvaluation = Evaluate();

      if (standingEvaluation >= upperBound) return standingEvaluation;

      lowerBound = Math.Max(lowerBound, standingEvaluation);
    }

    int originalLowerBound = lowerBound;

    var moves = _board.GetLegalMoves(qSearch);
    var interest = moves.Select(move => -Interest(move, bestMove)).ToArray();

    Array.Sort(interest, moves);

    bool principalVariation = true;

    foreach (Move move in moves)
    {
      if (!_initialSearch && !hasTime) break;

      _board.MakeMove(move);

      int score;

      if (principalVariation)
      {
        score = -Search(-upperBound, -lowerBound, ply + 1, depth - 1, move.IsCapture);
      }
      else
      {
        score = -Search(-lowerBound - 1, -lowerBound, ply + 1, depth - 1, move.IsCapture);
        if (score > lowerBound && score < upperBound) score = -Search(-upperBound, -lowerBound, ply + 1, depth - 1, move.IsCapture);
      }

      _board.UndoMove(move);

      if (bestMove == Move.NullMove) bestMove = move;

      if (score >= upperBound)
      {
        bestMove = move;

        lowerBound = score;

        if (!move.IsCapture && !qSearch)
          _historyTable[_board.IsWhiteToMove ? 0 : 1, (int)move.MovePieceType - 1, move.TargetSquare.Index] += depth * depth;

        break;
      }

      if (score > lowerBound)
      {
        bestMove = move;

        lowerBound = score;
      }

      principalVariation = false;
    }

    if (ply == 0) _bestMove = bestMove;

    int bound = 0;
    if (lowerBound <= originalLowerBound) bound = 1;
    if (lowerBound >= upperBound) bound = 2;

    // Can't save qSearch results cause it's incomplete information
    if (!qSearch && depth >= transpositionEntry.Depth) transpositionEntry = new TranspositionEntry(hash, lowerBound, bound, bestMove, depth);

    return lowerBound;
  }

  public Move Think(Board board, Timer timer)
  {
    _historyTable = new int[2, 6, 64];

    _board = board;
    _timer = timer;

    int depth = 4;
    _initialSearch = true;

    while (_initialSearch || hasTime)
    {
      Move lastBestMove = _bestMove;
      int lastEvaluation = _evaluation;

      int score = Search(-1000000, 1000000, 0, depth, false);

      if (!_initialSearch && !hasTime)
      {
        _bestMove = lastBestMove;
        _evaluation = lastEvaluation;

        break;
      }

      depth++;

      _initialSearch = false;

      if (score > 999000) break;
    }

    depth--;

    // Console.WriteLine($"Searched to depth {depth} in {timer.MillisecondsElapsedThisTurn}ms {nodesSearched / (timer.MillisecondsElapsedThisTurn / (float)1000)} nodes/sec"); // #DEBUG

    return _bestMove;
  }
}