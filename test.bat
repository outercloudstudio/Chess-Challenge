cd Chess-Challenge
cutechess-cli.exe -engine name="MyBot" arg="MyBot" cmd="D:\Chess-Challenge\UCI\bin\Debug\net6.0\Chess-Challenge.CLI.exe" -engine name="MyBotEvil" arg="MyBotEvil" cmd="D:\Chess-Challenge\UCI\bin\Release\net6.0\Chess-Challenge.CLI.exe" -each proto=uci tc=8+0.08 -concurrency 6 -maxmoves 200 -games 2 -repeat -rounds 10000 -ratinginterval 20 -pgnout games.pgn -recover -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 -openings file=Pohl.epd format=epd policy=round