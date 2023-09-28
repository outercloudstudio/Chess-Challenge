using System;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters = new float[1418];

  public MyBot()
  {
    for (int parameter = 0; parameter < 1418; parameter++)
    {
      var ints = decimal.GetBits(_compressedParameters[parameter / 16]);

      int bits = parameter % 16 * 6, bitsOffset = bits % 32, intIndex = bits / 32, quantized = ints[intIndex] >> bitsOffset & 0b111111;
      if (bitsOffset > 27) quantized |= ints[intIndex + 1] << 32 - bitsOffset & 0b111111;

      _parameters[parameter] = MathF.Pow(quantized / 64f - 0.5f, 3) * 6f;
    }
  }

  int parameterOffset = 0;

  float[] _layerInput = new float[54], _layerOutput = new float[16], _evaluationTensor = new float[37], _sightTensor = new float[54], _emptyTensor = new float[54];

  void Layer(int previousLayerSize, int layerSize)
  {
    Array.Copy(_emptyTensor, _layerOutput, 16);

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        _layerOutput[nodeIndex] += _layerInput[weightIndex] * _parameters[parameterOffset + nodeIndex * previousLayerSize + weightIndex];
      }

      _layerOutput[nodeIndex] = MathF.Max(MathF.Min(_layerOutput[nodeIndex] + _parameters[parameterOffset + layerSize * previousLayerSize + nodeIndex], 1), -1);
    }

    parameterOffset += layerSize * previousLayerSize + layerSize;

    Array.Copy(_layerOutput, _layerInput, layerSize);
  }

  int[] pieceValues = new int[] { 0, 1, 3, 3, 5, 9, 1000 };

  float Inference()
  {
    if (_board.IsInCheckmate()) return -100000 * WhiteToMoveFactor;

    int evaluation = 0;

    for (int type = 1; type < 7; type++)
    {
      evaluation += _board.GetPieceList((PieceType)type, true).Count * pieceValues[type];
      evaluation -= _board.GetPieceList((PieceType)type, false).Count * pieceValues[type];
    }

    for (int x = 0; x < 6; x++)
    {
      for (int y = 0; y < 6; y++)
      {
        Array.Copy(_emptyTensor, _sightTensor, 54);

        for (int kernelX = 0; kernelX < 3; kernelX++)
        {
          for (int kernelY = 0; kernelY < 3; kernelY++)
          {
            Piece piece = _board.GetPiece(new Square(x + kernelX, y + kernelY));

            if (piece.PieceType != PieceType.None) _sightTensor[kernelX * 18 + kernelY * 6 + (int)piece.PieceType - 1] = piece.IsWhite ? 1 : -1;
          }
        }

        parameterOffset = 0;

        Array.Copy(_sightTensor, _layerInput, 54);

        Layer(54, 8);
        Layer(8, 8);
        Layer(8, 1);

        _evaluationTensor[x * 6 + y] = _layerOutput[0];
      }
    }

    _evaluationTensor[36] = WhiteToMoveFactor;

    Array.Copy(_evaluationTensor, _layerInput, 37);
    Layer(37, 16);
    Layer(16, 16);
    Layer(16, 1);

    return _layerOutput[0] + evaluation;
  }

  Board _board;
  Timer _timer;
  Move _bestMove;

  int _nodes; //#DEBUG

  int[] MoveScores = new int[218];

  int WhiteToMoveFactor => _board.IsWhiteToMove ? 1 : -1;

  // Hash, Move, Score, Depth, Bound
  (ulong, Move, float, int, int)[] _transpositionTable = new (ulong, Move, float, int, int)[40000];

  float Search(int ply, int depth, float alpha, float beta)
  {
    _nodes++; //#DEBUG

    bool qSearch = depth <= 0;

    if (qSearch)
    {
      alpha = MathF.Max(alpha, Inference() * WhiteToMoveFactor);

      if (alpha >= beta) return alpha;
    }

    bool isCheck = _board.IsInCheck();

    ulong zobristKey = _board.ZobristKey;
    var (transpositionHash, transpositionMove, transpositionScore, transpositionDepth, transpositionFlag) = _transpositionTable[zobristKey % 40000];

    if (transpositionHash == zobristKey && transpositionDepth >= depth && (
      transpositionFlag == 1 ||
      transpositionFlag == 2 && transpositionScore <= alpha ||
      transpositionFlag == 3 && transpositionScore >= beta)
    ) return transpositionScore;

    Span<Move> moves = stackalloc Move[218];
    _board.GetLegalMovesNonAlloc(ref moves, qSearch && !isCheck);

    if (moves.Length == 0) return Inference() * WhiteToMoveFactor;

    int index = 0;

    // Scores are sorted low to high
    foreach (Move move in moves)
    {
      MoveScores[index++] = move == transpositionMove ? -1000000 : move.IsCapture
        ? (int)move.MovePieceType - 100 * (int)move.CapturePieceType
        : 1000000;
    }

    MoveScores.AsSpan(0, moves.Length).Sort(moves);

    Move bestMove = moves[0];
    int newTranspositionFlag = 1;

    foreach (Move move in moves)
    {
      if (outOfTime && ply > 0) return 100000000f;

      _board.MakeMove(move);

      float score = -Search(ply + 1, depth - 1, -beta, -alpha);

      _board.UndoMove(move);

      if (score > alpha)
      {
        newTranspositionFlag = 0;

        bestMove = move;

        if (ply == 0) _bestMove = move;

        alpha = score;

        if (score >= beta)
        {
          newTranspositionFlag = 2;

          break;
        }
      }
    }

    if (!outOfTime && !qSearch) _transpositionTable[zobristKey % 40000] = (zobristKey, bestMove, alpha, depth, newTranspositionFlag);

    return alpha;
  }

  bool outOfTime => _timer.MillisecondsElapsedThisTurn >= _timer.MillisecondsRemaining / 30f;

  public Move Think(Board board, Timer timer)
  {
    _board = board;
    _timer = timer;

    _nodes = 0; //#DEBUG

    int depth = 1;

    Move lastBestMove = Move.NullMove;

    while (true)
    {
      Search(0, depth++, -1000000f, 1000000f);

      if (outOfTime) break;

      lastBestMove = _bestMove;
    }

    Console.WriteLine($"My Bot: Nodes per second {_nodes / (timer.MillisecondsElapsedThisTurn / 1000f + 0.00001f)} Depth: {depth} Seconds {timer.MillisecondsElapsedThisTurn / 1000f}"); //#DEBUG

    return lastBestMove;
  }

  decimal[] _compressedParameters = { 71643580751471398296591240685M, 57808770151261461419226477691M, 19406506132741603410010081711M, 24482052838937212115927924269M, 35488450030206109575574123883M, 44505961270809876616670397460M, 6386205757416436081818797591M, 25518927372226980421006337218M, 23759813701360984535944704917M, 5148567674429100055487624011M, 65430025144897957756759659907M, 54785293903891415770164608693M, 58621329350839794016438352743M, 59677419043468099767816636657M, 55847011185427554312254960426M, 2712260948636043033985346640M, 9079033211120423414756572096M, 57004819721906195134032396679M, 32501453695806711460947358866M, 7829413217067279613767263334M, 16191685549497778701553119624M, 47317279480554206307608112134M, 8758087394345427328314942091M, 27155457833350119556620767723M, 51658818453113445286514981541M, 69028348657143427241524763433M, 51934205895145868172533681783M, 65207431665110062407286114012M, 16476965894721585970540000915M, 19881998466025449883699163916M, 63988824764756138895881190247M, 36549338497560467735161175413M, 8802884638218255110629284083M, 24030207099822022709986796679M, 8807360347731295831221629391M, 22663927820590848568716981592M, 26366101318163985896470115411M, 52888618345146642650996756934M, 20104505686152296584988134376M, 71662972483713349080852569136M, 57824030357890306811820739374M, 67754147866867210581916188216M, 51030800339494603283610414055M, 63893763633431852281605842083M, 52344139399671199669525982190M, 12586505396258988918759951409M, 22401126729864773123343548940M, 8981512959445864384542040530M, 46591620132819955298537907143M, 25255132075702536855666703318M, 28738200987018938277227709873M, 47472132678762306041403175404M, 7565236942929439305419269290M, 18853874505270817498542445830M, 19163371283611636837956121613M, 12035211327720050287397643090M, 30026316197745443247095341769M, 18766064531419282377772729687M, 23978051483937635908706803256M, 72900912677490329324072131309M, 14463343690081442275149454189M, 20023035597168653870810170503M, 8807124290459376578932286167M, 42540065527031475380029301191M, 58591526226367785436314367123M, 17085341290331636776627969263M, 14600123241326501917554563605M, 18206198034425516280857937383M, 61039374671395954943592143699M, 31729039200191393546303501651M, 26389446575821012298297677272M, 49085833787666536873660695955M, 323068096965598306092542738M, 50340991670837829630733660457M, 28910973119991200164092401367M, 50323158810991832396098070229M, 52414080967812762904973884011M, 52836800429546975705412285929M, 75971341946353016808913594775M, 54107988951112108751504891428M, 23875792211356679989254777364M, 26436004261411981641454114154M, 47460192405019022231898249639M, 52446638114096207455089683116M, 74689578046324202366055843304M, 23902695263491844496948022808M, 34128460711275439826764924200M, 34147809207135683504639466150M, 404248125950292647M };
}