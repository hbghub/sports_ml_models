select
    eventmetadata.gamecode, -- may not be complete for some game
    player.playerid,
    rushingpercentage exp_rushingShare
    -- overriderushingpercentage exp_rushingShare
from datalakefootball.player_expected_rates
where
    season >= '2018' and
    eventmetadata.eventtypeid in (1,2) and
    player.positionid in (1,8,9) and version='override'
order by season