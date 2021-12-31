select
       cast(season as integer) season,
       teamid,
       eventmetadata.week as week,
       eventmetadata.gameCode as gamecode,
       gametimeinseconds,
       timeofpossessioninseconds,
       totalplaysonfield,
       totaloffensiveplays,
       totaloffensiveplays * 3600 / gametimeinseconds as totaloffensiveplays_normalized,
       totaldesignedpassplays as totaldesignedpassplays,
       totalpassattempts as totalpassattempts,
       -- totalexpectedpassingplays, -- no value
       -- passingpercentage,
       totalrushingattempts,
       totalsacksallowed,
       totalthrowawaysandspikes,
       totalscrambles
       -- totalsacks,
       -- timeofpossessionpergameinminutes, -- normalized into a 60-min game?
       -- timeofpossessioninminutes,
from datalakefootball.team_aggregated_game_stats
where
    eventmetadata.week is not null
    and eventmetadata.eventtypeid < 3
order by season, teamid, week