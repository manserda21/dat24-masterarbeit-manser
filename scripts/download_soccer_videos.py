import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="./data/soccernet")

mySoccerNetDownloader.password = "s0cc3rn3t"

# Hole Listen der Spiele per Split
splits = ["train", "valid", "test", "challenge"]
real_madrid_games = []

for split in splits:
    games = getListGames(split=split)
    for game in games:
        if "Real Madrid" in game:
            real_madrid_games.append((split, game))

print("Real Madrid Spiele:")
for split, game in real_madrid_games:
    print(f"[{split}] {game}")
    
print(len(real_madrid_games))