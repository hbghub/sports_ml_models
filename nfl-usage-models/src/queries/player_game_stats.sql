with teamquery as
(
select season,
       eventmetadata.week week,
       eventmetadata.gameCode gamecode,
       teamid,
       totaltruepassattempts
from datalakefootball.team_aggregated_game_stats
where season>='2017' and eventmetadata.eventtypeid in (1,2)
order by season, week, gamecode, teamid
)

select
    cast(t1.season as integer) season,
    t1.eventmetadata.week week,
    t1.eventmetadata.gamecode gamecode,
    t1.eventmetadata.eventtypeid eventType,
    t1.teamid teamid,
    t1.player.playerid,
    t1.player.positionid,
    t1.receivertotaltargetsontruepassattempts,
    t2.totaltruepassattempts,
    case when t2.totaltruepassattempts = 0
        then null
        else t1.receivertotaltargetsontruepassattempts / cast (t2.totaltruepassattempts as double)
        end as targetShare,
    row_number() over (PARTITION BY
                         t1.season, t1.teamid, t1.eventmetadata.week, t1.player.positionid
                     ORDER BY
                        t1.receivertotaltargetsontruepassattempts DESC) Rank
from
    datalakefootball.player_aggregated_game_stats as t1
    left join
    teamquery t2
    on t1.season = t2.season and
       t1.eventmetadata.gamecode = t2.gamecode and
       t1.teamid = t2.teamid
where t1.season >= '2017' and
     t1.eventmetadata.eventtypeid in (1,2) and
     t1.player.positionid in (1,7,9) -- 1:WR, 7:TE, 9:RB
order by season, week, gamecode, teamid
