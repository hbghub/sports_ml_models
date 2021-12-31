WITH subQuery AS
(
    SELECT
        event.eventId AS eventId,
        lineList.scope.name AS lineScope,
        lineList.line AS lineData,
        cast(season as integer) season,
        eventmetadata.gamecode gamecode
    FROM datalakefootball.odds,
        UNNEST(event.lines) t(lineList)
    WHERE leagueid='8'
)
SELECT
    --season,
    gamecode game_code,
    --eventId,
    lineScope line_scope,
    sublineList.lineType.name AS line_type,
    --sublineList.total AS overUnderPoints,
    sublineList.favoritePoints favorite_points,
    sublineList.favoriteTeamId favorite_team_id
    --sublineList
FROM subQuery,
    UNNEST(lineData) t(sublineList)
WHERE sublineList.lineType.name = 'current' and season >= 2016 and season <= 2019