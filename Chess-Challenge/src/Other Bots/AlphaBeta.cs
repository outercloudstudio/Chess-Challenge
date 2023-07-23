using ChessChallenge.API;
using System;
using System.Linq;

namespace Frederox.AlphaBeta
{
    public class AlphaBeta : IChessBot
    {
        int positionsEvaluated;
        bool botIsWhite;
        int transpositionTableSize = 1048576;
        Transposition[] transpositions;
        int negativeInfinity = -100000;
        int positiveInfinity = 100000;

        public AlphaBeta()
        {
            transpositions = new Transposition[transpositionTableSize];
        }

        public Move Think(Board board, Timer timer)
        {
            if (board.PlyCount == 0 && board.IsWhiteToMove) return new Move("e2e4", board);
            botIsWhite = board.IsWhiteToMove;
            positionsEvaluated = 0;

            Move[] moves = board.GetLegalMoves();
            Move bestMove = moves[0];
            int bestScore = negativeInfinity;

            int depthToSearch = 4;

            for (int i = 0; i < moves.Length; i++)
            {
                int score = AlphaBetaEval(board, moves[i], depthToSearch, negativeInfinity, positiveInfinity);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = moves[i];
                }
            }

            return bestMove;
        }

        // fail-soft alpha-beta -- https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
        int AlphaBetaEval(Board board, Move move, int depth, int alpha, int beta)
        {
            board.MakeMove(move);

            ulong zobristKey = board.ZobristKey;
            Transposition? transposition = getTransposition(zobristKey);

            // If this move has already been evaluated re-use its result
            if (transposition != null && transposition.Depth >= depth)
            {
                board.UndoMove(move);
                return transposition.Score;
            }

            positionsEvaluated++;

            // Evaluate from the perspective of the bot's color
            int heuristic = EvaluateBoard(board, botIsWhite);

            if (depth == 0)
            {
                board.UndoMove(move);
                return heuristic;
            }

            //Move[] legalResponses = board.GetLegalMoves();
            Move[] legalResponses = getMovesWithCapturesFirst(board);
            int value;

            // maximize score
            if (botIsWhite == board.IsWhiteToMove)
            {
                value = negativeInfinity;
                for (int i = 0; i < legalResponses.Length; i++)
                {
                    value = Math.Max(value, AlphaBetaEval(board, legalResponses[i], depth - 1, alpha, beta));
                    alpha = Math.Max(alpha, value);

                    if (value >= beta) break;
                }
            }
            else
            {
                value = positiveInfinity;
                for (int i = 0; i < legalResponses.Length; i++)
                {
                    value = Math.Min(value, AlphaBetaEval(board, legalResponses[i], depth - 1, alpha, beta));
                    beta = Math.Min(beta, value);

                    if (value <= alpha) break;
                }
            }

            setTransposition(zobristKey, depth, value);
            board.UndoMove(move);
            return value;
        }

        int EvaluateBoard(Board board, bool asWhite)
        {
            int whiteScore = EvaluateSide(board, true);
            int blackScore = EvaluateSide(board, false);

            return (asWhite ? 1 : -1) * (whiteScore - blackScore);
        }

        int EvaluateSide(Board board, bool asWhite)
        {
            // Sum of all pieces values
            int pieceValues = board.GetPieceList(PieceType.Pawn, asWhite).Count * 1 +
                board.GetPieceList(PieceType.Knight, asWhite).Count * 3 +
                board.GetPieceList(PieceType.Bishop, asWhite).Count * 3 +
                board.GetPieceList(PieceType.Rook, asWhite).Count * 5 +
                board.GetPieceList(PieceType.Queen, asWhite).Count * 9 +
                board.GetPieceList(PieceType.King, asWhite).Count * 1000;

            // Mobility (the number of legal moves)
            int mobilityScore = (int)Math.Floor(0.1f * board.GetLegalMoves().Length);

            return pieceValues + mobilityScore;
        }

        Transposition? getTransposition(ulong zobristKey)
        {
            Transposition entry = transpositions[(int)(zobristKey % (ulong)transpositionTableSize)];
            if (entry != null && entry.ZobristKey == zobristKey) return entry;
            return null;
        }

        void setTransposition(ulong zobristKey, int depth, int score)
        {
            transpositions[(int)(zobristKey % (ulong)transpositionTableSize)] = new Transposition
            {
                ZobristKey = zobristKey,
                Depth = depth,
                Score = score
            };
        }

        Move[] getMovesWithCapturesFirst(Board board)
        {
            Move[] legalMoves = board.GetLegalMoves();
            return legalMoves.Where(m => m.IsCapture).ToArray()
                .Concat(legalMoves.Where(m => !m.IsCapture).ToArray()
            ).ToArray();
        }
    }

    public class Transposition
    {
        public ulong ZobristKey { get; set; }
        public int Depth { get; set; }
        public int Score { get; set; }
    }
}