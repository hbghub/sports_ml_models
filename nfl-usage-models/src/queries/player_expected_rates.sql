select
    eventmetadata.gamecode,
    player.playerid,
    targetpercentage exp_targetShare
from datalakefootball.player_expected_rates
where
    season >= '2017' and
    eventmetadata.eventtypeid in (1,2) and
    player.positionid in (1,7,9) and version='override'
order by season