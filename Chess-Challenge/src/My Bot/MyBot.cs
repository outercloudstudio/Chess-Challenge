using System;
using System.Collections.Generic;
using System.IO; //#DEBUG
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters = new float[2930];

  public MyBot()
  {
    // var values = new List<int>();

    // int valueIndex = 0;
    // while (true)
    // {
    //   var bytes = decimal.GetBits(_compressedParameters[valueIndex / 16]);

    //   int quantized = 0;

    //   int byteIndex = (6 * (valueIndex % 16)) / 8;

    //   quantized |= (bytes[0]) << 0;

    //   _parameters[valueIndex] = (float)_compressedParameters[valueIndex];
    // }

    int uncompressedParameterCount = 2930;

    List<int> rawParameters = new List<int>() { 0 };

    bool countingPrunedNodes = false;

    int lastCountIndex = 0;

    foreach (string parameter in File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_7.txt")[0..2930])
    {
      float value = float.Parse(parameter);

      bool shouldPrune = MathF.Abs(value) < 0.04f;

      if (shouldPrune && !countingPrunedNodes)
      {
        countingPrunedNodes = true;

        lastCountIndex = rawParameters.Count;
        rawParameters.Add(0);
      }
      else if (!shouldPrune && countingPrunedNodes)
      {
        countingPrunedNodes = false;

        lastCountIndex = rawParameters.Count;
        rawParameters.Add(0);
      }
      else if (rawParameters[lastCountIndex] == 63)
      {
        countingPrunedNodes = !countingPrunedNodes;

        lastCountIndex = rawParameters.Count;
        rawParameters.Add(0);
      }

      if (!shouldPrune) rawParameters.Add((int)(MathF.Min(MathF.Max(MathF.Pow(value / 4.8f, 1 / 3f) + 0.5f, -0.5f), 0.5f) * 64f));

      rawParameters[lastCountIndex]++;
    }

    int compressedTokenCount = (int)MathF.Ceiling(rawParameters.Count / 16f);

    Console.WriteLine($"Param Count: {uncompressedParameterCount} Compressed Tokens: {compressedTokenCount} Raw Parameters: {rawParameters.Count}"); //#DEBUG

    List<decimal> decimals = new List<decimal>();

    for (int readIndex = 0; readIndex < rawParameters.Count; readIndex += 16)
    {
      byte[] bytes = new byte[16];

      for (int offset = 0; offset < Math.Min(16, rawParameters.Count - readIndex); offset++)
      {
        for (int byteIndex = 0; byteIndex < 16; byteIndex++)
        {
          int bitOffset = offset * 6 - byteIndex * 8;

          if (bitOffset < 0 || bitOffset > 7) continue;

          // Console.WriteLine(Convert.ToString(BitConverter.GetBytes(rawParameters[readIndex + offset])[0], toBase: 2).PadLeft(8, '0'));
          // Console.WriteLine(bitOffset);
          // Console.WriteLine(Convert.ToString((byte)(BitConverter.GetBytes(rawParameters[readIndex + offset])[0] << bitOffset), toBase: 2).PadLeft(8, '0'));
          // Console.WriteLine('\n');

          bytes[byteIndex] = bytes[byteIndex] |= (byte)(BitConverter.GetBytes(rawParameters[readIndex + offset])[0] << bitOffset);
        }
      }

      decimals.Add(ByteArrayToDecimal(bytes, 0));
    }

    string output = "";

    foreach (decimal value in decimals)
    {
      output += value.ToString() + "M, ";
    }

    File.WriteAllText("D:/Chess-Challenge/Training/Models/Lila_7_Compressed.txt", output);
  }

  public static decimal ByteArrayToDecimal(byte[] src, int offset)
  {
    using (MemoryStream stream = new MemoryStream(src))
    {
      stream.Position = offset;
      using (BinaryReader reader = new BinaryReader(stream))
        return reader.ReadDecimal();
    }
  }

  int parameterOffset = 0;

  float[] _layerInput = new float[54];
  float[] _layerOutput = new float[32];
  float[] _evaluationTensor = new float[37];
  float[] _sightTensor = new float[54];
  float[] _emptyTensor = new float[54];

  void Layer(int previousLayerSize, int layerSize)
  {
    Array.Copy(_emptyTensor, _layerOutput, 32);

    for (int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
    {
      for (int weightIndex = 0; weightIndex < previousLayerSize; weightIndex++)
      {
        _layerOutput[nodeIndex] += _layerInput[weightIndex] * _parameters[parameterOffset + nodeIndex * previousLayerSize + weightIndex];
      }

      _layerOutput[nodeIndex] = MathF.Tanh(_layerOutput[nodeIndex] + _parameters[parameterOffset + layerSize * previousLayerSize + nodeIndex]);
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

        Layer(54, 16);
        Layer(16, 16);
        Layer(16, 1);

        _evaluationTensor[x * 6 + y] = _layerOutput[0];
      }
    }

    _evaluationTensor[36] = WhiteToMoveFactor;

    Array.Copy(_evaluationTensor, _layerInput, 37);
    Layer(37, 32);
    Layer(32, 16);
    Layer(16, 1);

    return _layerOutput[0] + evaluation;
    // return _layerOutput[0];
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

    ulong zobristKey = _board.ZobristKey;
    var (transpositionHash, transpositionMove, transpositionScore, transpositionDepth, transpositionFlag) = _transpositionTable[zobristKey % 40000];

    if (transpositionHash == zobristKey && transpositionDepth >= depth && (
      transpositionFlag == 1 ||
      transpositionFlag == 2 && transpositionScore <= alpha ||
      transpositionFlag == 3 && transpositionScore >= beta)
    ) return transpositionScore;

    bool qSearch = depth <= 0;

    if (qSearch)
    {
      alpha = MathF.Max(alpha, Inference() * WhiteToMoveFactor);

      if (alpha >= beta) return alpha;
    }

    bool isCheck = _board.IsInCheck();

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
      if (outOfTime && ply > 0) return -100000f;

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

    _transpositionTable[zobristKey % 40000] = (zobristKey, bestMove, alpha, depth, newTranspositionFlag);

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
    _bestMove = Move.NullMove;

    while (true)
    {
      Search(0, depth++, -1000000f, 1000000f);

      if (lastBestMove == Move.NullMove) lastBestMove = _bestMove;

      if (outOfTime) break;

      lastBestMove = _bestMove;

      if (depth > 50) break;
    }

    Console.WriteLine($"Nodes per second {_nodes / (timer.MillisecondsElapsedThisTurn / 1000f + 0.00001f)} Depth: {depth} Seconds {timer.MillisecondsElapsedThisTurn / 1000f}"); //#DEBUG

    return lastBestMove;
  }

  decimal[] _compressedParameters = new decimal[] { 39614083618324417431394582547M, 1237949557814329509398183968M, 39614234734052009997529776144M, 9147936759877696M, 39614081269832324212722761760M, 232567106881233361228005376M, 75631650702209161625600M, 290444428162414659223158784M, 16112563323823777640492302336M, 77673483910239925020131328M, 56742193777929936437248M, 7446983052303035366008422400M, 1238246993106784462743535616M, 826244333572M, 39614081257132168797258776576M, 39614234734052009997529776160M, 39614383490911070599815372832M, 1238091155021980040834318368M, 39614081257141316734058500097M, 213224291434530221225672704M, 960383613337511919648M, 2363516106041800523776M, 19645044568878461577330688M, 39633877417428358349507788993M, 25165953M, 1624466399991048966176M, 0M, 19826534706134244388112961536M, 39614232372859620625418813440M, 39614081257132168796973563969M, 4971102970255381623390351424M, 232567104549494586527973376M, 137022414979514549403648M, 39614234734052009996984516608M, 39614232742956571541505048608M, 39614234734052009998058266784M, 39614233259456543541644230688M, 39730593844235235526845661216M, 77675845102629709439565856M, 153476919841200414130208M, 2361192389372110962784M, 39614081257132168796990341120M, 1237949557805335571887296512M, 39614392933320038193358180353M, 1237940039294387474153865216M, 7446985410009357151012651021M, 4722375507730342346752M, 96825742721056M, 16112716800743618841241714688M, 39614990312689128403275612192M, 1238093516205080738168569888M, 39614234734052009997529776148M, 1257436329319056024562302976M, 12398745567159885450409193568M, 1237940261799371402054205472M, 911416731193842071502859M, 6935543426150563840M, 140739661008960M, 23685619390644154793984M, 39614085980669728448219189248M, 2475882444392850338839007232M, 1276630461685366398577016835M, 39614081553432995480807149632M, 1257434119168390680870387906M, 4951769749449006777108545600M, 1315316235486846451984896001M, 151412037426449425109056M, 39614992673872387498466869249M, 39633877494674398226419814528M, 4971107766409390475001987138M, 1237940039294528212732739616M, 2476033555490461012530823182M, 3714150684672011141279907842M, 1238254077836438881609711621M, 4952069472183163093273296897M, 39653522538089924242998558721M, 1237949557823758922174959712M, 39614388212133502892864962562M, 3714143599961075014498254880M, 1238091155031128047907504132M, 39614395294512427567086043139M, 39614081265229645393567154208M, 153476910834622654591008M, 39614105016539107480755241025M, 1237949705379275176266698784M, 1238249354326318510828945411M, 1257285213591462975813713923M, 39634182010120968849625993312M, 1238249354290861457683775553M, 296309833883231191043M, 39653220230543327112336769090M, 2222985344M, 152223685017755860336640M, 39614100220385076570191036416M, 14240886992251773796416M, 3714131795259982259721076929M, 39614234734052009997219921922M, 39614232390180605529841729568M, 13636683395141960626864128032M, 1237949631601305803129094176M, 3733163080858918427127451649M, 39614095424232479422857609248M, 3714134156398192025986953216M, 2476031198927917331404824577M, 613907679083396203544580M, 39614695165946018494534385664M, 1237949557805463322467762370M, 1238256439010809181741514760M, 1257282855912026517826437122M, 33130370652326716252224M, 4971102970256771338467278880M, 39633879778611599787568398336M, 1238249354326459246689910914M, 39614234734045694402228649986M, 1237940262952438041684475904M, 39615002119794285879014457352M, 39692210449417153579298324512M, 39614234734052009997236437024M, 1237949557823618184686600224M, 1257287796126639358713266181M, 39614109665136760397647319264M, 2475880301084905608069775392M, 33277935456908875137026M, 3713971233583733391931473952M, 38987857683702089936535555M, 1034170875506514202720M, 297453748188566519808M, 193M, 1238249354290857060736176192M, 1238249354290025898103541761M, 39885487467133885427274223617M, 39692210449417153579298324512M, 39614234734052009997236437024M, 39614234734052009999157166112M, 39614082365098882660736958496M, 23685628538580352630816M, 2475889597091269932464480256M, 20251871012669946297581573M, 7466477126394561580037570689M, 1238091159660551385766690816M, 1276630461666648310469885962M, 1276776929967374538660388928M, 153476910974809849995424M, 39614095498027614300635930624M, 39633877419752220223996952577M, 59086326679405809597677697M, 58937494943920015084097568M, 39614100220385089833587445792M, 616268826863312792719424M, 39672563118605072219370684481M, 39633877492368256148954431488M, 39614234734052009997220184130M, 39614232390180605529841729568M, 1237940261808378601308946464M, 2476189393576237334570991618M, 39672865350050968677350637571M, 2495227835432131901381705728M, 19796162647068697273065504M, 3714131795205943391927402564M, 1276630978193919054023237633M, 39634030896671916223260266560M, 1257288165080225658227916933M, 1354002083057292663477768416M, 1276630756815002091841917024M, 20249582418781488910700640M, 19796242153924243111542851M, 19645125273251911398326343M, 4952067112124840797182623810M, 616268862606306540003329M, 39633726382405732901834457218M, 20098472464614596593582151M, 3714145962323542673381654658M, 11141769668610486679697108993M, 39614995036208538205842571266M, 39634481957584231929286234208M, 21991339851906M };
}