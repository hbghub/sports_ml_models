with teamquery as
(
select eventmetadata.gameCode gamecode,
       teamid,
       totaltruepassattempts game_totalTruePassAttempts
from datalakefootball.team_aggregated_game_stats
where season>='2017' and eventmetadata.eventtypeid in (1,2)
)

select
  cast(t1.season as integer) season,
  t1.eventmetadata.week,
  t1.eventmetadata.gamecode,
  t1.teamid,
  t1.player.playerid,
  t1.player.positionid,
  t1.onfieldtotaltruepassattempts ytd_onFieldTotalTruePassAttempts,
  t1.receivertotaltargetsontruepassattempts ytd_totalTargetsOnTruePassAttempts,
  t2.game_totaltruepassattempts,
  row_number() over (PARTITION BY
                         t1.season, t1.teamid, t1.eventmetadata.week, t1.player.positionid
                     ORDER BY
                         t1.receivertotaltargetsontruepassattempts DESC) ytd_rank
from datalakefootball.player_aggregated_ytd_stats t1
    left join teamquery t2
    on t1.eventmetadata.gamecode = t2.gamecode and
       t1.teamid=t2.teamid
where
    t1.season >='2017' and
    t1.player.positionid in (1,7,9) and
    t1.eventmetadata.eventtypeid in (1,2) and
    t1.eventmetadata.week is not null -- when a week is missing, the player may not be active
order by season, week, gamecode, teamid
