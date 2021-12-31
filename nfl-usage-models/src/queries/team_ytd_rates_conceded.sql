select
    cast(season as integer) season,
    teamid,
    eventmetadata.week as week,
    pace as ytd_pace_conceded
from datalakefootball.team_aggregated_ytd_stats_conceded
where
    season >= '2013'
    and (eventmetadata.eventtypeid = 1 or eventmetadata.eventtypeid = 2)
    and eventmetadata.week is not null
order by season, teamid, week