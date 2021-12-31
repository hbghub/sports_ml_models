select
   teamid,
   eventmetadata.gameCode as gamecode,
   gametimeinseconds,
   timeofpossessioninseconds,
   totalpoints,
   case
       when gametimeinminutes > 60 then True
       else False
       end
       overTime

from datalakefootball.team_aggregated_game_stats
where
    season>='2016' and
    eventmetadata.week is not null
    and eventmetadata.eventtypeid in (1,2)