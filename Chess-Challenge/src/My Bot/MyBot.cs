﻿using System;
using System.Collections.Generic;
using System.IO; //#DEBUG
using System.Linq;
using ChessChallenge.API;

public class MyBot : IChessBot
{
  float[] _parameters = new float[2930];

  public MyBot()
  {
    // float[] parameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_7.txt")[0..2930].Select(line =>
    // {
    //   return float.Parse(line);
    // }).ToArray();

    // int[] compressedParameters = File.ReadAllLines("D:/Chess-Challenge/Training/Models/Lila_7.txt")[0..2930].Select(line =>
    // {
    //   float value = float.Parse(line);
    //   int quantized = (int)(MathF.Min(MathF.Max(MathF.Pow(MathF.Abs(value) / 4.8f, 1 / 3f) * (value < 0 ? -1 : 1) + 0.5f, 0f), 1f) * 64f);

    //   return quantized;
    // }).ToArray();

    for (int parameter = 0; parameter < 2930; parameter++)
    {
      var ints = decimal.GetBits(_compressedParameters[parameter / 16]);
      int bitsOffset = parameter % 16 * 6 % 32;
      int intIndex = parameter % 16 * 6 / 32;

      int quantized = ints[intIndex] >> bitsOffset & 0b111111;
      if (bitsOffset > 27) quantized |= ints[intIndex + 1] << 32 - bitsOffset & 0b111111;

      _parameters[parameter] = MathF.Pow(quantized / 64f - 0.5f, 3) * 4.8f;
    }

    // int compressedTokenCount = (int)MathF.Ceiling(compressedParameters.Length / 16f);

    // Console.WriteLine($"Param Count: 2930 Compressed Tokens: {compressedTokenCount}"); //#DEBUG

    // List<decimal> decimals = new List<decimal>();

    // for (int readIndex = 0; readIndex < _parameters.Length; readIndex += 16)
    // {
    //   byte[] bytes = new byte[16];

    //   // Console.WriteLine("Values:");

    //   for (int offset = 0; offset < Math.Min(16, compressedParameters.Length - readIndex); offset++)
    //   {
    //     // Console.WriteLine(offset + ": " + compressedParameters[readIndex + offset] + " " + Convert.ToString(compressedParameters[readIndex + offset], toBase: 2).PadLeft(6, '0'));

    //     int bits = offset * 6;
    //     int byteIndex = bits / 8;
    //     int bitsOffset = bits - byteIndex * 8;

    //     bytes[byteIndex] |= (byte)(compressedParameters[readIndex + offset] << bitsOffset);
    //     if (bitsOffset > 2) bytes[byteIndex + 1] |= (byte)(compressedParameters[readIndex + offset] >> 8 - bitsOffset);
    //   }

    //   // Console.WriteLine("Bytes:"); //#DEBUG

    //   // int i = 0;
    //   // foreach (byte b in bytes) Console.WriteLine((i++) + " " + Convert.ToString(b, toBase: 2).PadLeft(8, '0'));

    //   decimals.Add(ByteArrayToDecimal(bytes, 0));

    //   // break;
    // }

    // string output = "";

    // foreach (decimal value in decimals)
    // {
    //   output += value.ToString() + "M, ";
    // }

    // File.WriteAllText("D:/Chess-Challenge/Training/Models/Lila_7_Compressed.txt", output);
  }

  // public static decimal ByteArrayToDecimal(byte[] src, int offset)
  // {
  //   using (MemoryStream stream = new MemoryStream(src))
  //   {
  //     stream.Position = offset;
  //     using (BinaryReader reader = new BinaryReader(stream))
  //       return reader.ReadDecimal();
  //   }
  // }

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

  decimal[] _compressedParameters = new decimal[] { 72960421518591874409191427830M, 66649327480537848462537358834M, 4491720803646012485575323118M, 11378442063772113375485205996M, 28226034062655981920052872656M, 63173680543351021664714155313M, 2614627042801101075777350084M, 21220327085053658059128787718M, 5089323294427310403598332426M, 332355006225497801881588116M, 21188607328842369211390514056M, 67124933917816446672450109449M, 12546552319959475909807092375M, 60906254047422084302303433481M, 74602719790179323976890174391M, 76653674021945064816943199344M, 2723048658277002223384833265M, 12593701399663984498379878530M, 11531689410357723875793820552M, 9012658805121287152279032656M, 46096701591358463893976170628M, 52985954047029403007081288968M, 31060340800448918757203951119M, 17467179250878697663462679364M, 10258982889556608018356143026M, 7758943187610098178175865071M, 17563610209823133666313822341M, 198298605524596424813347461M, 12879358496481584798932839620M, 15032433782514213991772877644M, 56786030976185987355378250441M, 24409496171408627391656047356M, 62863509359248057382992388373M, 77852078041400552512986254858M, 68556355663850417597770076989M, 60539320764787767916782656821M, 77697014418846976059637718392M, 215251018136654911488979776M, 10974972151489187050861371463M, 7141542234108885723981202283M, 15258466754267192048510082124M, 57123158088270483035873403601M, 7488435262600048672362236935M, 65451253540068406278244608643M, 9786805984194056997104496503M, 61680661703926066127531355792M, 67929196541408081964929881205M, 68953425587841785646124105460M, 60470725454676494439140777717M, 53023174398344531263611260172M, 52012843584555143500735139535M, 11376654498141929249351713614M, 5169410064813586906880049865M, 44624200725228946788048453702M, 33935324925519973289286019302M, 12159716548253285993275607728M, 10980399745071286416234445678M, 16938379691211110531132146924M, 68247762695406074882471302289M, 12160023502074672382775224880M, 59815442749007077048181779093M, 17769803814982401000728519916M, 12218339858143642985971772206M, 15640880725426633162178004141M, 62659138583117195198395939603M, 18727728844240550094102079033M, 21952510003842713418146926857M, 14092870474152782173816055213M, 28825843197381138617067731754M, 57357984549505970670858836628M, 62682693523466977025200671443M, 44032538118540855301085735140M, 28945201759297848476138110192M, 48601591134153878550321650724M, 55255854054095927137356547374M, 54775459595246177456359670830M, 61573213730991749224841970537M, 14039431402692751031056938030M, 5276447989075634156904172683M, 56455077704008391988694956875M, 62850304635544039118418294354M, 29406611889010273785925257880M, 51068583543407996492253299980M, 19397382918739686206605039028M, 50972913891712277788300600079M, 23679736458321933479666217838M, 9103412920862739593786866534M, 16486793093112167988473430852M, 21124198824699983861140734866M, 65372891274632345084660834639M, 54734322400910025400656534763M, 71663534907636482413128494231M, 64732177520312258755056577658M, 20221371232503999736287309423M, 60207758620285401769399227630M, 66002587883381556117439207690M, 21444030802987393963431900333M, 59000324591270238235928765233M, 20624065136081619572736206035M, 30139166045356383676496833645M, 52405027609002939055472686254M, 50470958801771833184696233265M, 20564524201414514313311493808M, 13255622935304622665863600943M, 44824170079130442753862863403M, 62173296497692441208192031596M, 59195577881754404756326124846M, 62936487553890635506476026791M, 22603776017578068741554513005M, 27686902250728966013665350894M, 51718115280039886505816866321M, 71663242119787094511087705259M, 20033335150348290582238661563M, 49142453695854739065169318993M, 57849581070048073085914880332M, 15738581104502528475646069074M, 57830249793667268507098352681M, 16350841748724339012414532269M, 55275364006061694810955113906M, 7683352109280131419222142056M, 16191824386900297999657264071M, 57209437826554249470855901582M, 25699266106554767599560259952M, 69819181650537724316367516944M, 62741369421152169605070519483M, 58103963689564353732403215469M, 70442529862559173615825952432M, 61644004708223696588711566448M, 30093560380887238478591401529M, 14543300701285741239716467824M, 32508857073444205900329569683M, 36756382965084303967004850998M, 54802788085309401615309548558M, 47830457357334032231182546164M, 18902078027766995101308335536M, 51660806513862595014320339759M, 22683928482133014189100831922M, 62135853374851075053434381135M, 27500248957905952362176531377M, 59195134555816720547538326546M, 33069791612825804841196374833M, 54230538392582963191692346093M, 61798441065788068962070574317M, 71487498209894296793676746354M, 51852426548379117010792422586M, 52491458536578790597690491760M, 51727129078593769625503060687M, 48697805145955458046201861675M, 49061933840043468766019144148M, 45347843435084758266954267033M, 70086456234060918832662146587M, 24028790565456402893847057495M, 49176794161304649649898744278M, 24357270307555690403286674006M, 49143345736171742095185177043M, 50344088045196302532784483285M, 25095593893914484925540427051M, 45347843435084758266954267047M, 70086456234060918832662146587M, 25519407218742668038253753943M, 53671264720970790408976689901M, 28130921062902450070720255574M, 26815389518640402608617796946M, 33880319079179579326589683284M, 9141706280203418760881803748M, 33880319079179579326589683304M, 9141706280203418760881803748M, 23933195896776346939474135784M, 51620316907481318433533278805M, 23120632121615209055511307734M, 31807345036165786040397297232M, 45347843435084758266954267025M, 70086456234060918832662146587M, 33071296240077997172072450775M, 25093766041802802224830690581M, 56436976650424240482496080233M, 25195343963858666262872347927M, 25151283402553358702145616093M, 53690683083823857898963155692M, 51659295987900988107639573223M, 25189370570267556370238338322M, 47746097460395903391610022232M, 48770733752080986767439326498M, 1433M };
}