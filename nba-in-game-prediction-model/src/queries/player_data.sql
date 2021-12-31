WITH players AS (
    SELECT
        playerid player_id,
        draft.year
    FROM datalakebasketball.players
    where draft.year is not null
),
teams AS (
    SELECT
        api.eventmetadata.gamecode game_code,
        teams.teamid team_id,
        if(teams.teamlocationtype.teamlocationtypeid = 1,
            1,
            0) AS is_home_team
    FROM datalakebasketball.api_events api, UNNEST(api.event.teams) t(teams)
    WHERE leagueid = '1'
                AND season >= '2004'
                AND season <= '2018'
                and eventmetadata.eventtypeid=1
)

SELECT
    CAST(season AS INTEGER) season,
    eventmetadata.gamecode game_code,
    eventmetadata.gamedateutcepoch game_time,
    player.playerid player_id,
    --player.firstname first_name,
    --player.lastname last_name,
    team.teamid team_id,
    t3.is_home_team at_home,
    positionid position_id,
    t2.year draft_year,
    --isgameplayed game_played,
    isgamestarted game_started,
    minutesplayed minutes,
    points,
    fieldgoals.attempted fg_attempt,
    fieldgoals.made fg_made,
    freethrows.attempted ft_attempt,
    freethrows.made ft_made,
    threepointfieldgoals.attempted point_3_attempt,
    threepointfieldgoals.made point_3_made,
    rebounds.offensive offensive_rebounds,
    rebounds.defensive defensive_rebounds,
    assists,
    blockedshots blocks,
    turnovers
FROM
    datalakebasketball.player_stats_game t1
    LEFT JOIN players t2
        ON t1.player.playerid=t2.player_id
    LEFT JOIN teams t3
        ON t1.eventmetadata.gamecode=t3.game_code AND t1.team.teamid=t3.team_id
WHERE
    leagueid='1'
    AND eventmetadata.eventtypeid=1
    AND season>='2004'
    AND season<='2018'
    AND isgameplayed -- does this mean eligibility to play?
ORDER BY
    season, player_id, game_time