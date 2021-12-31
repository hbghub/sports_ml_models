WITH mainQuery AS
(
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
        season,
        gamecode,
        eventId,
        lineScope,
        sublineList.lineType.name AS lineType,
        sublineList.total AS overUnderPoints,
        sublineList.favoritePoints,
        sublineList.favoriteTeamId,
        IF(sublineList.favoriteMoney='EVEN', 100.0, 1.0 * CAST(sublineList.favoriteMoney AS INTEGER)) AS favoriteMoney,
        IF(sublineList.underdogMoney='EVEN', 100.0, 1.0 * CAST(sublineList.underdogMoney AS INTEGER)) AS underdogMoney,
        IF(sublineList.homeMoney='EVEN', 100.0, 1.0 * CAST(sublineList.homeMoney AS INTEGER)) AS homeMoney,
        IF(sublineList.awayMoney='EVEN', 100.0, 1.0 * CAST(sublineList.awayMoney AS INTEGER)) AS awayMoney,
        IF(sublineList.overMoney='EVEN', 100.0, 1.0 * CAST(sublineList.overMoney AS INTEGER)) AS overMoney,
        IF(sublineList.underMoney='EVEN', 100.0, 1.0 * CAST(sublineList.underMoney AS INTEGER)) AS underMoney
    FROM subQuery,
        UNNEST(lineData) t(sublineList)
)
SELECT
    season,
    gamecode, -- same as eventId
    -- eventId,
    -- lineScope,
    -- lineType,
    overUnderPoints,
    favoritePoints,
    favoriteTeamId,
    -- favoriteMoney,
    CASE
        WHEN favoriteMoney < 0 THEN - (100 - favoriteMoney) / favoriteMoney
        WHEN favoriteMoney > 0 THEN (100 + favoriteMoney) / 100
    END AS favoriteMoneyDecimal,
    -- underdogMoney,
    CASE
        WHEN underdogMoney < 0 THEN - (100 - underdogMoney) / underdogMoney
        WHEN underdogMoney > 0 THEN (100 + underdogMoney) / 100
    END AS underdogMoneyDecimal,
    -- homeMoney,
    CASE
        WHEN homeMoney < 0 THEN - (100 - homeMoney) / homeMoney
        WHEN homeMoney > 0 THEN (100 + homeMoney) / 100
    END AS homeMoneyDecimal,
    -- awayMoney,
    CASE
        WHEN awayMoney < 0 THEN - (100 - awayMoney) / awayMoney
        WHEN awayMoney > 0 THEN (100 + awayMoney) / 100
    END AS awayMoneyDecimal,
    -- overMoney,
    CASE
        WHEN overMoney < 0 THEN - (100 - overMoney) / overMoney
        WHEN overMoney > 0 THEN (100 + overMoney) / 100
    END AS overMoneyDecimal,
    -- underMoney,
    CASE
        WHEN underMoney < 0 THEN - (100 - underMoney) / underMoney
        WHEN underMoney > 0 THEN (100 + underMoney) / 100
    END AS underMoneyDecimal
FROM mainQuery
WHERE lineType = 'current' and season >= 2016
order by season, gamecode
