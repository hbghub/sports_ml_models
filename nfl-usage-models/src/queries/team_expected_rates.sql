select
   cast(season as integer) season,
   team.teamid as teamid,
   opponentteamid,
   eventmetadata.week as week,
   totalplays as exp_totalPlays
   -- passingplays as exp_passingPlays, -- no values
   -- rushingplays as exp_rushingPlays
from datalakefootball.team_expected_rates
where
    season >= '2015'
    and version = '1'
    and (eventmetadata.eventtypeid = 1 or eventmetadata.eventtypeid = 2)
order by season, teamid, week
