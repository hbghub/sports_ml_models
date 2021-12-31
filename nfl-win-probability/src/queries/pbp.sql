select
    cast(season as integer) season,
    eventmetadata.week, eventmetadata.eventid game_code, eventmetadata.eventtypeid event_type_id,
    if(teammetadata[1].ishometeam, teammetadata[1].teamid, teammetadata[2].teamid) home_team,
    if(teammetadata[2].ishometeam, teammetadata[1].teamid, teammetadata[2].teamid) away_team,

    pbp.startpossessionteamid offense_team,
    if(pbp.startpossessionteamid = teammetadata[1].teamid, teammetadata[2].teamid, teammetadata[1].teamid) defense_team,

    cast(pbp.playid as integer) play_id, pbp.driveid drive_id, pbp.period,
    pbp.secondsremaininginperiod seconds_remaining_in_period,

    case
        when pbp.period in (1,2) then 1
        when pbp.period in (3,4) then 2
        else pbp.period - 2
        end half_game,

    case
        when pbp.down is null then 0
        else pbp.down
        end down,

    case
        when pbp.distance is null then 0.0
        else cast(pbp.distance as double)
        end yards_to_go,

    cast(pbp.startyardsfromgoal as double) yards_from_goal,

    pbp.awayscorebefore away_score,
    pbp.homescorebefore home_score,
    pbp.awayscoreafter  away_score_after,
    pbp.homescoreafter  home_score_after,
    if(teammetadata[1].ishometeam, teammetadata[1].score, teammetadata[2].score) home_final_score,
    if(teammetadata[2].ishometeam, teammetadata[1].score, teammetadata[2].score) away_final_score,

    case
        when teammetadata[1].teamid = pbp.startpossessionteamid and teammetadata[1].ishometeam then pbp.homescorebefore
        when teammetadata[2].teamid = pbp.startpossessionteamid and teammetadata[2].ishometeam then pbp.homescorebefore
        else pbp.awayscorebefore
        end as offense_score,

    pbp.playtype.playtypeid play_type_id, pbp.playtype.name play_name,

    --ceate known playtype pre-play
    case
        when pbp.playtype.playtypeid in (22) then 1
        when pbp.playtype.playtypeid in (52, 53, 54, 55, 56) then 2
        when pbp.playtype.playtypeid in (42, 35, 36) then 3
        else 0
        end play_design,

    case
        when pbp.playtype.playtypeid = 57 then 1
        else 0
        end offense_timeout,

    case
        when pbp.playtype.playtypeid = 58 then 1
        else 0
        end defense_timeout

from datalakefootball.pbp
where
    leagueid='16' -- CFB
    and season>='2015' and season<='2019'
    and eventmetadata.eventtypeid in (1,2)
    -- and pbp.period <= 4
    -- and pbp.down is not null -- this will elliminate field goal attempts and a few other actions
    -- and pbp.playtype.playtypeid not in (5,13) -- 2-min warning, kickoff
order by season, eventmetadata.week, eventmetadata.eventid, pbp.period, play_id --, secondsremaininginperiod desc