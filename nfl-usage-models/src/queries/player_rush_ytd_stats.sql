select eventmetadata.gameCode gamecode,
       teamid,
       totalrushingattempts game_totalRushingAttempts
from datalakefootball.team_aggregated_game_stats
where season>='2017' and eventmetadata.eventtypeid in (1,2)