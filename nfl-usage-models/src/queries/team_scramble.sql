select cast(season as integer) season,
        teamid, eventmetadata.week week,
       sum(totalscrambles) game_scrambles, 1 idx
from datalakefootball.player_aggregated_game_stats
where season >= '2013' and leagueid='8'
group by season, teamid, eventmetadata.week
order by season, teamid, week