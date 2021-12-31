select
    cast(season as integer) season,
    teamid,
    eventmetadata.week as week,
    gametimeinseconds as ytd_gameTime,
    totaloffensiveplays as ytd_totalPlays,
    totaldesignedpassplays / floor(gametimeinseconds / 3600) as ytd_passPlaysPerGame, -- overtime is estimated as well
    pace as ytd_pace,
    timeofpossessioninseconds as ytd_TOP,
    timeofpossessioninseconds / floor(gametimeinseconds / 3600) as ytd_TOPperGame, -- ot
    passingpercentage as ytd_passingpercentage,
    -- percentageofpointsfrompassingtdsytd as ytd_perofpointsfrompassingtds,
    totalpoints / floor(gametimeinseconds / 3600) as ytd_totalPointsPerGame,  -- ot
    totalsacks / floor(gametimeinseconds / 3600) as ytd_totalSacksPerGame,     -- ot

    timeofpossessioninseconds / floor(gametimeinseconds / 3600) / pace as ytd_offensivePlaysPerGame  -- ot
from datalakefootball.team_aggregated_ytd_stats
where
    season >= '2013'
    and (eventmetadata.eventtypeid = 1 or eventmetadata.eventtypeid = 2)
    and eventmetadata.week is not null
order by season, teamid, week
