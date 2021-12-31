WITH game_opponents AS
(
    select
      eventmetadata.gamecode game_code,
      event.teams[1].teamid team1_id, -- team1 is home team
      event.teams[2].teamid team2_id
    FROM datalakebasketball.api_events
    where leagueid = '1'
        and season >= '2004'
)

select
  cast(t1.season as integer) season,
  t1.eventmetadata.gamecode game_code,
  t1.eventmetadata.eventtypeid event_type_id,
  DATE_FORMAT(from_unixtime(t1.eventmetadata.gamedateutcepoch), '%Y-%m-%d') date,
  t1.teamid team_id,
  if(t1.teamid = t2.team1_id, 1, 0) at_home,
  if(t1.teamid = t2.team1_id, t2.team2_id, t2.team1_id) opp_team_id,
  t1.points,
  t1.fieldgoals.attempted fg_attempted,
  t1.fieldgoals.made fg_made,
  t1.freethrows.attempted ft_attempted,
  t1.freethrows.made ft_made,
  t1.rebounds.offensive offensive_rebounds,
  t1.rebounds.defensive defensive_rebounds,
  t1.turnovers.total + t1.turnovers.team turnovers,
  t1.assists assists
from
  datalakebasketball.team_stats_game t1
  LEFT JOIN game_opponents t2
    ON t1.eventmetadata.gamecode=t2.game_code
where
  t1.season>='2004' and t1.season<='2018'
  and t1.leagueid='1'
  and t1.teamid not in (53, 54)   -- all_start teams
  and t1.eventmetadata.eventtypeid in (1,2)
  and t1.points is not null       -- filter out postponed games
order by t1.season, t1.teamid, t1.eventmetadata.gamedateutcepoch -- eventmetadata.gamecode