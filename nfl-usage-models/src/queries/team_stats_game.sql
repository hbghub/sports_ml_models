select cast(season as integer) season,
       teamid, eventmetadata.gameCode gamecode, eventmetadata.week week,
       passing.attempts passingAttempts,
       passing.yards passingYards,
       rushing.attempts rushingAttempts,
       rushing.yards rushingYards
from datalakefootball.team_stats_game
where season>='2013' and (eventmetadata.eventtypeid=1 or eventmetadata.eventtypeid=2)
order by season, teamid, week