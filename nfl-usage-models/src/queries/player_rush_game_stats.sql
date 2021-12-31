with teamquery as
(
    select season,
           eventmetadata.week week,
           eventmetadata.gameCode gamecode,
           teamid,
           totalrushingattempts
    from datalakefootball.team_aggregated_game_stats
    where season>='2017' and eventmetadata.eventtypeid in (1,2)
    order by season, week, gamecode, teamid
),
--playerquery as
--(
--  select playerid,
--           positions[1].positionid positionid
--  from datalakefootball.players
--)

playerquery as
(
  select nflid as playerid,
           positionabbr as positionid
  from datalakefootball.not_stats_data_players
)

select
    cast(t1.season as integer) season,
    t1.eventmetadata.week week,
    t1.eventmetadata.gamecode gamecode,
    t1.eventmetadata.eventtypeid eventType,
    t1.teamid teamid,
    t1.playerid,
    t3.positionid,
    t1.playerstats.rushingstats.attempts rushertotalrushingattempts,
    t2.totalrushingattempts,

    case when t2.totalrushingattempts = 0
        then null
        else t1.playerstats.rushingstats.attempts / cast (t2.totalrushingattempts as double)
        end as rushingShare,
    row_number() over (PARTITION BY
                         t1.season, t1.teamid, t1.eventmetadata.week, t3.positionid
                     ORDER BY
                        t1.playerstats.rushingstats.attempts DESC) Rank,
    if (playerstats.inactives is not null, False, True) as isActive
from
    datalakefootball.player_stats_game as t1
    left join teamquery t2
    on t1.season = t2.season and
       t1.eventmetadata.gamecode = t2.gamecode and
       t1.teamid = t2.teamid
    left join playerquery t3
    on t1.playerid = t3.playerid
where t1.season >= '2017' and
     t1.eventmetadata.eventtypeid in (1,2) and
     t3.positionid in ('WR','TE','QB','RB') --(1,8,9) -- 1:WR, 7:TE, 8:QB, 9:RB
order by season, week, gamecode, teamid
