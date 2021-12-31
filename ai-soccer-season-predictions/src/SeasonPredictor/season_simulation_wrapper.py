import json
#import boto3
import datetime
#import requests
#import aws4_requests
import numpy as np
import pandas as pd


def aggregate_match_results_by_team(results, return_detail=False):
    results_df = pd.DataFrame(results)
    results_df['outcome'] = np.sign(results_df['score_home'] - results_df['score_away']) + 1
    home_df = results_df[['home_team_id', 'away_team_id', 'game_id', 'outcome', 'score_home', 'score_away']]\
        .rename(columns={'home_team_id': 'team_id', 'away_team_id': 'opponent_id', 'score_home': 'GF', 'score_away': 'GA'})
    home_df['is_home'] = 1
    away_df = results_df[['away_team_id', 'home_team_id', 'game_id', 'outcome', 'score_home', 'score_away']]\
        .rename(columns={'away_team_id': 'team_id', 'home_team_id': 'opponent_id', 'score_away': 'GF', 'score_home': 'GA'})
    away_df['outcome'] = away_df['outcome'].map({0: 2, 1: 1, 2: 0})
    away_df['is_home'] = 0
    results_df = pd.concat([home_df, away_df], axis=0)
    results_df['G'] = 1
    results_df['W'] = (results_df['outcome'] == 2).astype(int)
    results_df['D'] = (results_df['outcome'] == 1).astype(int)
    results_df['L'] = (results_df['outcome'] == 0).astype(int)
    results_df['points'] = results_df['outcome'].map({2: 3, 1: 1, 0: 0})
    agg_df = results_df.groupby('team_id')[['G', 'W', 'D', 'L', 'points', 'GF', 'GA']].sum()
    if return_detail:
        return agg_df, results_df
    else:
        return agg_df


def simulate_season_standing_with_tiebreakers(match_preds, match_results, team_names, comp_id):
    n_runs = match_preds['n_runs']
    match_ids = match_preds['game_id']
    run_list = []

    # for all predicted matches, one loop is for one game
    for ind_match, match_id in enumerate(match_ids):
        exact_score_probs = np.array(match_preds['pred_exact_home_away'][ind_match]) / \
                            np.sum(match_preds['pred_exact_home_away'][ind_match])
        scores = np.random.multinomial(
            n=1,
            pvals=exact_score_probs,
            size=(match_preds['n_runs'])
        )
        ind_scores = np.argmax(scores, axis=1)

        # both 10*11=11*10= 110, but what is the purpose?
        aux_home = np.tile(np.arange(0, match_preds['n_exact_home']), (match_preds['n_exact_away'], 1)).flatten(order='F')
        aux_away = np.tile(np.arange(0, match_preds['n_exact_away']), (match_preds['n_exact_home']))

        # to handle in-game logic
        home_scores = aux_home[ind_scores] + match_preds['current_score_home'][ind_match]
        away_scores = aux_away[ind_scores] + match_preds['current_score_away'][ind_match]

        # one-hot encoding: 0=home_lose, 1=draw, 2=home_win; array shape 10000 * 3
        away_outcomes = np.zeros((n_runs, 3)).astype(int)
        away_outcomes[np.arange(n_runs), np.sign(home_scores - away_scores) + 1] = 1
        home_outcomes = np.fliplr(away_outcomes)

        home_df = pd.DataFrame(
            data=np.hstack((np.arange(1, n_runs + 1).reshape(-1, 1), home_outcomes, home_scores.reshape(-1, 1), away_scores.reshape(-1, 1))),
            columns=['run', 'W', 'D', 'L', 'GF', 'GA']
        )
        home_df['team_id'] = match_preds['home_team_id'][ind_match]
        home_df['opponent_id'] = match_preds['away_team_id'][ind_match]
        home_df['is_home'] = 1
        away_df = pd.DataFrame(
            data=np.hstack((np.arange(1, n_runs + 1).reshape(-1, 1), away_outcomes, away_scores.reshape(-1, 1), home_scores.reshape(-1, 1))),
            columns=['run', 'W', 'D', 'L', 'GF', 'GA']
        )
        away_df['team_id'] = match_preds['away_team_id'][ind_match]
        away_df['opponent_id'] = match_preds['home_team_id'][ind_match]
        away_df['is_home'] = 0
        run_df = pd.concat([home_df, away_df], axis=0)
        run_list.append(run_df)

    all_runs_df = pd.concat(run_list, axis=0).reset_index(drop=True)
    all_runs_df['G'] = 1
    all_runs_df['points'] = (all_runs_df[['W', 'D', 'L']] * np.array([3, 1, 0])).sum(axis=1)
    # 'GF_away' not needed for Euro tournament
    all_runs_df['GF_away'] = all_runs_df['GF'].where(all_runs_df['is_home'] == 0, 0)

    # Aggregate simulated game results by run and team, each team played 10 games in this period
    agg_runs_df = all_runs_df.groupby(['run', 'team_id']).sum().reset_index(drop=False)

    # Ranking criteria
    try:
        ranking_criteria = {
            '2kwbbcootiqqgmrzs6o5inle5': ['points', 'GD', 'GF', 'H2H_points', 'H2H_GF_away'],  # Premier League,
            '34pl8szyvrbwcmfkuocjm3r6t': ['points', 'H2H_points', 'H2H_GD', 'GD', 'GF'],  # Primera División,
            '1r097lpxe0xn03ihb7wi98kao': ['points', 'H2H_points', 'H2H_GD', 'GD', 'GF'],  # Serie A,
            '6by3h89i2eykc341oz7lv1ddd': ['points', 'GD', 'GF', 'H2H_points', 'H2H_GF', 'H2H_GF_away', 'GF_away'],  # Bundesliga,
            'dm5ka0os1e3dxcp3vh05kmp33': ['points', 'GD', 'GF'],  # Ligue 1,
        }[comp_id]
    except KeyError:
        ranking_criteria = ['points', 'GD', 'GF', 'H2H_points', 'H2H_GF_away']
    # Preliminary criteria (anything that does not involve head-to-head subsetting)
    preliminary_criteria = []
    for i in ranking_criteria:
        if i.find('H2H_') == 0:
            break
        else:
            preliminary_criteria.append(i)
    breakpoint()
    # Add observed results from past games
    agg_match_results, match_team_results = aggregate_match_results_by_team(match_results, return_detail=True)
    sum_cols = ['G', 'W', 'D', 'L', 'GF', 'GA', 'points']
    prev_df = agg_match_results.reindex(agg_runs_df.team_id.values, columns=sum_cols).fillna(0).astype(int)
    agg_runs_df[sum_cols] = agg_runs_df[sum_cols].values + prev_df.values
    agg_runs_df['GD'] = agg_runs_df['GF'] - agg_runs_df['GA']
    agg_runs_df = agg_runs_df.sort_values(by=['run'] + preliminary_criteria, ascending=False).reset_index(drop=True)

    # Identify ties (2 or more teams with identical values in the (preliminary) sorting features)
    aux = agg_runs_df.drop_duplicates(subset=['run'] + preliminary_criteria).index.values
    agg_runs_df['tie_id'] = 0
    agg_runs_df.loc[aux, 'tie_id'] = 1
    agg_runs_df['tie_id'] = np.cumsum(agg_runs_df['tie_id'])
    tie_size = agg_runs_df.groupby(['tie_id'])['team_id'].count().rename('tie_size').reset_index(drop=False)
    agg_runs_df = agg_runs_df.merge(right=tie_size, on='tie_id', how='left')

    # Head-to-head data
    h2h_df = agg_runs_df[['run', 'tie_id', 'team_id']]
    h2h_df = h2h_df.merge(right=h2h_df, on=['run', 'tie_id'], how='left')
    h2h_df = h2h_df.loc[h2h_df.team_id_x != h2h_df.team_id_y]
    h2h_df.rename(columns={'team_id_x': 'team_id', 'team_id_y': 'opponent_id'}, inplace=True)
    # Relevant results
    h2h_past_df = h2h_df.merge(
        right=match_team_results[['team_id', 'opponent_id', 'is_home'] + sum_cols],
        on=['team_id', 'opponent_id'],
        how='inner'
    )

    # Relevant simulations
    h2h_pred_df = h2h_df.merge(
        right=all_runs_df[['run', 'team_id', 'opponent_id', 'is_home'] + sum_cols],
        on=['run', 'team_id', 'opponent_id'],
        how='inner'
    )

    # Join results and predictions
    h2h_df = pd.concat([h2h_past_df, h2h_pred_df], axis=0).reset_index(drop=True)
    h2h_df['GD'] = h2h_df['GF'] - h2h_df['GA']
    h2h_df['GF_away'] = h2h_df['GF'].where(h2h_df['is_home'] == 0, 0)
    h2h_df.drop(columns='opponent_id', inplace=True)

    # Aggregate "head-to-head runs"
    agg_h2h_df = h2h_df.groupby(['run', 'tie_id', 'team_id']).sum().reset_index(drop=False)
    agg_h2h_df.rename(columns={i: 'H2H_' + i for i in ['points', 'GD', 'GF', 'GF_away']}, inplace=True)

    # Add tie breaker values to the main data frame
    h2h_cols = [i for i in agg_h2h_df.columns if i.find('H2H_') == 0]
    agg_runs_df = agg_runs_df.merge(
        right=agg_h2h_df[['run', 'tie_id', 'team_id'] + h2h_cols],
        on=['run', 'tie_id', 'team_id'],
        how='left'
    )

    agg_runs_df[h2h_cols] = agg_runs_df[h2h_cols].fillna(0)

    # Sort main data frame again, now with all tie-breaking data included
    agg_runs_df = agg_runs_df.sort_values(by=['run'] + ranking_criteria, ascending=False).reset_index(drop=True)

    # Calculate rank ("cheap" method, repeating a [1, 2, 3, ..., N] array as many times as runs)
    agg_runs_df['rank'] = np.arange(1, team_names.size + 1).tolist() * n_runs

    # Distribution of simulated end-of-season rankings
    rank_dist = agg_runs_df.groupby(['team_id', 'rank'])[['run']].count().rename(columns={'run': 'n'})
    rank_dist['p'] = rank_dist['n'] / n_runs

    # Distribution of simulated end-of-season league points
    points_dist = agg_runs_df.groupby(['team_id', 'points'])[['run']].count().rename(columns={'run': 'n'})
    points_dist['p'] = points_dist['n'] / n_runs

    # Average number of points per team
    avg_points_df = agg_runs_df.groupby('team_id')[['points']].mean().sort_values(by='points', ascending=False)
    avg_points_df['team_name'] = team_names.loc[avg_points_df.index].values
    # ordered_team_ids = avg_points_df.index.values

    # Merge results into a single dictionary
    aux = agg_match_results[['points', 'GF', 'GA']].copy()
    aux['GD'] = aux['GF'] - aux['GA']
    ordered_team_ids = aux.sort_values(by=['points', 'GD', 'GF'], ascending=False).index.values
    final_list = [{'id': team_id,
                   'name': avg_points_df.loc[team_id, 'team_name'],
                   'current': agg_match_results.loc[team_id].to_dict(),
                   'predicted': {'average': avg_points_df.loc[team_id, 'points'], 'rank': {}, 'points': {}}}
                  for team_id in ordered_team_ids]
    breakpoint()
    for (team_id, rank), sim_results in rank_dist.to_dict(orient='index').items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]['predicted']['rank'][rank] = sim_results
    breakpoint()
    for (team_id, rank), sim_results in points_dist.to_dict(orient='index').items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]['predicted']['points'][rank] = sim_results

    breakpoint()
    return final_list


if __name__=='__main__':
    print('hello')

    with open('local_match_preds.json') as json_file:
        match_preds = json.load(json_file)

    match_results = pd.read_csv('local_match_results.csv')

    team_names = pd.read_csv('local_team_names.csv', index_col=0, squeeze=True)

    comp_id = "34pl8szyvrbwcmfkuocjm3r6t"

    final_list = simulate_season_standing_with_tiebreakers(match_preds, match_results, team_names, comp_id)

    breakpoint()