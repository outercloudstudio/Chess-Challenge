using System.Threading;
using ChessChallenge.Application;
using ChessChallenge.Chess;

public class UIInterface
{
  private static bool s_DisplayingGame = false;

  private void DisplayGame(BoardUI boardUI, Board board, bool white)
  {
    if (s_DisplayingGame) return;

    new Thread(() =>
    {
      s_DisplayingGame = true;

      Board displayBoard = new Board();
      displayBoard.LoadStartPosition();

      boardUI.SetPerspective(white);

      foreach (Move move in board.AllGameMoves)
      {
        displayBoard.MakeMove(move, false);

        boardUI.UpdatePosition(displayBoard, move, true);

        Thread.Sleep(200);
      }

      s_DisplayingGame = false;
    }).Start();
  }
}