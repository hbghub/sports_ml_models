import json
import boto3
import datetime
import requests
import aws4_requests
import numpy as np
import pandas as pd

boto3.setup_default_session(profile_name='innovation-playground')
dynamo = boto3.client(service_name='dynamodb', region_name='eu-west-1')
ssm = boto3.client(service_name='ssm', region_name='eu-west-1')
parameter = ssm.get_parameter(Name='/adp/opta/feeds/auth/key', WithDecryption=True)
OUTLET_AUTH_KEY = parameter['Parameter']['Value']
IP_OR_DNS = ['api.performfeeds.com', '23.214.191.97', '35.176.250.248', '96.6.246.25'][1]


MODEL_ENDPOINTS = {
    'aggregator': 'https://vhirrc05ld.execute-api.us-east-1.amazonaws.com/prod/soccer-prediction-aggregator-0-01'
}
EURO_2020_GROUPS = {
    '9s2kpeunkes0g17l95r3t91j6': 'A',  # Turkey
    'eks08q2vbr45w563zrctsl5xo': 'A',  # Italy
    'eyhp0bgsz2deg91xaw4zotn5c': 'A',  # Wales
    '339wxbt07ltnva7hop6m5o477': 'A',  # Switzerland
    'akblgvrthxxgta4ou9lnslxgb': 'B',  # Denmark
    '7j1jtn1skeji5iavxu9yty4os': 'B',  # Finland
    'pfvmhackqyhhk02majd2tb8l':  'B',  # Belgium
    'f0frccyrlq2jjihdraoie2e2d': 'B',  # Russia
    '657oha8nkne1ybba2bouzkbo7': 'C',  # Netherlands
    '7a8no9dyo0r9g31sfi9h7ffbc': 'C',  # Ukraine
    '61g4epojd5f198lv1sje27yh3': 'C',  # Austria
    '4u2lrl06blm7j4r3udbkb5wx9': 'C',  # North Macedonia
    'ck8m1cn23sukwsurgx5qakttk': 'D',  # England
    '4njsfszcgd9m765d6suktsz2a': 'D',  # Croatia
    'bcllqlzvj9j7opjene2zvran7': 'D',  # Scotland
    '70tnqyqn871jwlk26gtjw7knm': 'D',  # Czech Republic
    'eh7yt2x2wck51oixw8012ux5j': 'E',  # Spain
    '4w1ktgu1nbbgqfe3ssbb4fg5v': 'E',  # Sweden
    'ba25ib8pofxr85hkbs8lt7kw7': 'E',  # Poland
    '7ntk6fzpvcrgl55evfr1z7xmv': 'E',  # Slovakia
    'ajqq56th1sg5f7kcx9yon9z6x': 'F',  # Hungary
    '8gxg8f7p9299jbrz30u8bsc7g': 'F',  # Portugal
    '4pz87gsel7183b7kcadw1dwzv': 'F',  # France
    '3l2t2db0c5ow2f7s7bhr6mij4': 'F',  # Germany
}


def lambda_handler(event, context):
    if 'Records' in event:  # Feature generation lambda trigger event
        event_json = event['Records'][0]['body']
        event = json.loads(event_json)
        print('SoccerSeasonPredictor lambda triggered with these parameters:\n{0}'.format(event))

    predictions = wrapper(
        tcal_ids=event['tcal_ids'],
        team_ids=event['team_ids'],
        player_id='N/A' if 'player_id' not in event else event['player_id'],
        prediction_types=event['prediction_types'],
        target_list=event['targets'] if 'targets' in event else None,
        n_runs=event['n_runs'],
        what_if_data=event['what_if'] if 'what_if' in event else None,
    )
    # print(predictions)

    return {
        'statusCode': 200,
        'body': json.dumps('SoccerLivePredictor executed successfully!'),
        'data': json.dumps(predictions),
    }


def wrapper(tcal_ids, team_ids, player_id, prediction_types, target_list=None, what_if_data=None, n_runs=1000):
    prediction_list = []
    for prediction_type in prediction_types:
        for tcal_id in tcal_ids:
            # Get game predictions from DynamoDB
            game_predictions, team_names = get_game_predictions(tcal_id, team_ids, player_id, prediction_type)

            # Format game predictions into the inputs that the aggregator expects
            features, results = format_game_predictions(game_predictions, prediction_type, target_list, n_runs)

            # Get season predictions from the Cortex aggregator
            # predictions = get_season_predictions(features)

            # Generate league standing predictions
            if prediction_type == 'league':
                if what_if_data is not None:
                    features, results = add_what_if_results(what_if_data, features, results)
                #  Simulation without realistic tie-breakers (faster)
                # agg_match_results = aggregate_match_results_by_team(results)
                # standing_predictions = simulate_season_standing(features, agg_match_results, team_names, sim_mode='score')
                #  Simulation with realistic tie-breakers
                comp_id = get_competition_uuid(tcal_id)
                if tcal_id == 'cnqwzc1jx33qoyfgyoorl0yqx':
                    standing_predictions = simulate_euro_2020(features, pd.DataFrame(results), team_names, comp_id)
                else:
                    standing_predictions = simulate_season_standing_with_tiebreakers(features, pd.DataFrame(results), team_names, comp_id)

            elif prediction_type == 'team':
                agg_team_stats = aggregate_match_stats_by_team(results, target_list)
                standing_predictions = simulate_team_season(features, agg_team_stats, target_list, team_names)
            elif prediction_type == 'player':
                standing_predictions = simulate_player_season(results, features, target_list)

            # 6: Send outputs
            # publish_to_kinesis(season_predictions, prediction_type)
            # for prediction in predictions:
            #     write_predictions_to_dynamo(prediction, prediction_type)

            export_game_list = []
            for game in game_predictions:
                export_game_list.append({
                    'game_id': game['game_id'],
                    'game_description': game['predictions']['match_description'],
                    'game_status': game['predictions']['match_status'],
                    'home_team_id': game['home_team_id'],
                    'home_team_name_id': game['predictions']['home_team_name'],
                    'away_team_id': game['away_team_id'],
                    'away_team_name_id': game['predictions']['away_team_name'],
                    'home_team_score': int(game['predictions']['current.score'][0]),
                    'away_team_score': int(game['predictions']['current.score'][1]),
                })

            prediction_list.append({
                tcal_id: {'season_predictions': standing_predictions,
                          'game_list': export_game_list}})
    return prediction_list


def get_competition_uuid(tcal_id):
    url = opta_feed_url_builder(query='MFL', identifier=tcal_id, entity_type='tournament_calendar')
    with open_requests_session() as s:
        r = s.get(url)
        r.raise_for_status()
        data = json.loads(r.content.decode())
    return data['fixtures'][0]['competition']['competitionUUID']


def add_what_if_results(what_if_data, features, results):
    for game_id, game_data in what_if_data.items():
        if game_id in results['game_id']:
            try:
                ind_game = results['game_id'].index(game_id)
            except ValueError:
                continue
            results['score_home'][ind_game] = game_data['score'][0]
            results['score_away'][ind_game] = game_data['score'][1]
        elif game_id in features['game_id']:
            try:
                ind_game = features['game_id'].index(game_id)
            except ValueError:
                continue
            results['game_id'].append(features['game_id'].pop(ind_game))
            results['score_home'].append(game_data['score'][0])
            results['score_away'].append(game_data['score'][1])
            results['home_team_id'].append(features['home_team_id'].pop(ind_game))
            results['away_team_id'].append(features['away_team_id'].pop(ind_game))
            for k in ['pred_1X2_home', 'pred_1X2_draw', 'pred_1X2_away', 'pred_exact_home_away',
                      'current_score_home', 'current_score_away']:
                features[k].pop(ind_game)
    return features, results


def get_game_predictions(tcal_id, team_ids, player_id, prediction_type):
    if prediction_type in ['league', 'team']:
        game_list, team_names = get_game_ids(tcal_id, team_ids)
        # Retrieve predictions game by game
        # for ind in range(len(game_list)):
        #     dynamo_key = {'game_id': {'S': game_list[ind]['game_id']}, 'player_id': {'S': 'N/A'}}
        #     game_list[ind]['predictions'] = get_item_from_dynamo('SoccerLivePredictionStore', dynamo_key)
        # Retrieve predictions in batches (of size 100)
        dynamo_key_list = [{'game_id': {'S': game_list[ind]['game_id']}, 'player_id': {'S': 'N/A'}}
                           for ind in range(len(game_list))]
        batch_predictions = get_item_batch_from_dynamo('SoccerLivePredictionStore', dynamo_key_list)
        for ind in range(len(game_list)):
            game_list[ind]['predictions'] = batch_predictions[game_list[ind]['game_id']]

    elif prediction_type == 'player':
        game_list, team_names = get_game_ids(tcal_id, team_ids)
        dynamo_key_list = [{'game_id': {'S': game_list[ind]['game_id']}, 'player_id': {'S': player_id}}
                           for ind in range(len(game_list))]
        batch_predictions = get_item_batch_from_dynamo('SoccerLivePredictionStore', dynamo_key_list)
        for ind in range(len(game_list)):
            game_list[ind]['player_id'] = player_id
            game_list[ind]['player_name'] = batch_predictions[game_list[ind]['game_id']].pop('player_name')
            game_list[ind]['predictions'] = batch_predictions[game_list[ind]['game_id']]

    return game_list, team_names


def format_game_predictions(game_predictions, prediction_type, target_list, n_runs=1000):
    if prediction_type == 'league':
        feature_dict = {
            'n_runs': n_runs,
            'n_exact_home': len(game_predictions[0]['predictions']['score.exact_home']),
            'n_exact_away': len(game_predictions[0]['predictions']['score.exact_away']),
            'game_id': [],
            'pred_1X2_home': [],
            'pred_1X2_draw': [],
            'pred_1X2_away': [],
            'pred_exact_home_away': [],
            'home_team_id': [],
            'away_team_id': [],
            'current_score_home': [],
            'current_score_away': [],
        }
        results_dict = {
            'game_id': [],
            'score_home': [],
            'score_away': [],
            'home_team_id': [],
            'away_team_id': [],
        }
        for game_dict in game_predictions:
            if game_dict['predictions']['match_status'] in ['Played', 'Awarded']:
                results_dict['game_id'].append(game_dict['game_id'])
                results_dict['score_home'].append(int(game_dict['predictions']['current.score'][0]))
                results_dict['score_away'].append(int(game_dict['predictions']['current.score'][1]))
                results_dict['home_team_id'].append(game_dict['home_team_id'])
                results_dict['away_team_id'].append(game_dict['away_team_id'])
            elif game_dict['predictions']['match_status'] in ['Fixture', 'Playing', 'Postponed']:
                feature_dict['game_id'].append(game_dict['game_id'])
                feature_dict['home_team_id'].append(game_dict['home_team_id'])
                feature_dict['away_team_id'].append(game_dict['away_team_id'])
                feature_dict['pred_1X2_home'].append(game_dict['predictions']['score.1X2'][0])
                feature_dict['pred_1X2_draw'].append(game_dict['predictions']['score.1X2'][1])
                feature_dict['pred_1X2_away'].append(game_dict['predictions']['score.1X2'][2])
                feature_dict['pred_exact_home_away'].append(game_dict['predictions']['score.exact_home_away'])
                feature_dict['current_score_home'].append(int(game_dict['predictions']['current.score'][0]))
                feature_dict['current_score_away'].append(int(game_dict['predictions']['current.score'][1]))
            else:
                raise ValueError('Unexpected match status found for game "{0}": "{1}"'
                                 .format(game_dict['game_id'], game_dict['predictions']['match_status']))

    elif prediction_type == 'team':
        feature_dict = {
            'n_runs': n_runs,
            'game_id': [],
            'home_team_id': [],
            'away_team_id': [],
        }
        results_dict = {
            'game_id': [],
            'home_team_id': [],
            'away_team_id': []
        }
        for target in target_list:
            feature_dict[target + '_home'] = []
            feature_dict[target + '_away'] = []
            results_dict[target + '_home'] = []
            results_dict[target + '_away'] = []
        for game_dict in game_predictions:
            if game_dict['predictions']['match_status'] in ['Played', 'Awarded']:
                results_dict['game_id'].append(game_dict['game_id'])
                results_dict['home_team_id'].append(game_dict['home_team_id'])
                results_dict['away_team_id'].append(game_dict['away_team_id'])
                for target in target_list:
                    results_dict[target + '_home'].append(int(game_dict['predictions']['current.' + target][0]))
                    results_dict[target + '_away'].append(int(game_dict['predictions']['current.' + target][1]))
            elif game_dict['predictions']['match_status'] in ['Fixture', 'Playing', 'Postponed']:
                feature_dict['game_id'].append(game_dict['game_id'])
                feature_dict['home_team_id'].append(game_dict['home_team_id'])
                feature_dict['away_team_id'].append(game_dict['away_team_id'])
                for target in target_list:
                    feature_dict[target + '_home'].append(game_dict['predictions'][target + '.exact_home'])
                    feature_dict[target + '_away'].append(game_dict['predictions'][target + '.exact_away'])
            else:
                raise ValueError('Unexpected match status found for game "{0}": "{1}"'
                                 .format(game_dict['game_id'], game_dict['predictions']['match_status']))

    elif prediction_type == 'player':
        feature_dict = {
            'n_runs': n_runs,
            'player_id': game_predictions[0]['player_id'],
            'player_name': game_predictions[0]['player_name'],
            'game_id': [],
        }
        results_dict = {
            'game_id': [],
        }
        for target in target_list:
            feature_dict[target] = {
                'exact': [],
                'scalar': [],
            }
            results_dict[target] = []

        for game_dict in game_predictions:
            for target in target_list:
                if game_dict['predictions']['match_status'] in ['Played', 'Awarded']:
                    if target == target_list[0]:
                        results_dict['game_id'].append(game_dict['game_id'])
                    results_dict[target].append(game_dict['predictions']['current.{0:s}'.format(target)])
                elif game_dict['predictions']['match_status'] in ['Fixture', 'Playing', 'Postponed']:
                    if target == target_list[0]:
                        feature_dict['game_id'].append(game_dict['game_id'])
                    feature_dict[target]['exact'].append(game_dict['predictions']['{0:s}.exact'.format(target)])
                    feature_dict[target]['scalar'].append(game_dict['predictions']['{0:s}.scalar'.format(target)])
                else:
                    raise ValueError('Unexpected match status found for game "{0}": "{1}"'
                                     .format(game_dict['game_id'], game_dict['predictions']['match_status']))

    return feature_dict, results_dict


def get_season_predictions(feature_dict):
    predictions = _get_predictions(MODEL_ENDPOINTS['aggregator'], feature_dict)
    return predictions


def _get_predictions(model_url, input_features):
    model_response = aws4_requests.post(
        url=model_url,
        payload=json.dumps(input_features)
    )
    model_response_json = json.loads(model_response.text)
    return model_response_json


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


def aggregate_match_stats_by_team(results, target_list):
    results_df = pd.DataFrame(results)
    home_cols = {'home_team_id': 'team_id'}
    away_cols = {'away_team_id': 'team_id'}
    for target in target_list:
        home_cols[target + '_home'] = target
        home_cols[target + '_away'] = target + '_opp'
        away_cols[target + '_home'] = target + '_opp'
        away_cols[target + '_away'] = target
    home_df = results_df[list(home_cols.keys())].rename(columns=home_cols)
    away_df = results_df[list(away_cols.keys())].rename(columns=away_cols)
    results_df = pd.concat([home_df, away_df], axis=0)
    results_df['games'] = 1
    agg_df = results_df.groupby('team_id').sum()
    return agg_df


def simulate_season_standing_loop_based(team_results, payload, n_runs=1000):
    match_ids = payload['game_id']
    samples = {}
    for ind_match, match_id in enumerate(match_ids):
        outcomes = np.random.multinomial(
            n=1,
            pvals=[
                payload['pred_1X2_home'][ind_match],
                payload['pred_1X2_draw'][ind_match],
                payload['pred_1X2_away'][ind_match],
            ],
            size=(payload['n_runs'])
        )
        # Add home team results
        try:
            samples[str(payload['home_team_id'][ind_match])].append(
                np.sum(outcomes * np.array([3, 1, 0]).reshape(1, 3), axis=1).tolist())
        except KeyError:
            samples[str(payload['home_team_id'][ind_match])] = \
                [np.sum(outcomes * np.array([3, 1, 0]).reshape(1, 3), axis=1).tolist()]
        # Add away team results
        try:
            samples[str(payload['away_team_id'][ind_match])].append(
                np.sum(outcomes * np.array([0, 1, 3]).reshape(1, 3), axis=1).tolist())
        except KeyError:
            samples[str(payload['away_team_id'][ind_match])] = \
                [np.sum(outcomes * np.array([0, 1, 3]).reshape(1, 3), axis=1).tolist()]

    # Aggregate samples
    all_team_ids = np.unique(team_results.index.to_list() + list(samples.keys()))
    simulation_results = {}
    for team_id in all_team_ids:
        simulation_results[team_id] = {
            'rank': {i: {'n': 0} for i in range(1, all_team_ids.size + 1)},
            'points': {}
        }
    for run in range(n_runs):
        run_dict = {}
        for team_id, game_list in samples.items():
            run_dict[team_id] = {
                'points': 0,
                'G': len(game_list),
                'W': 0,
                'D': 0,
                'L': 0,
            }
            for game_results in game_list:
                points = game_results[run]
                run_dict[team_id]['points'] += points
                run_dict[team_id]['W'] += int(points) == 3
                run_dict[team_id]['D'] += int(points) == 1
                run_dict[team_id]['L'] += int(points) == 0
        run_df = pd.DataFrame(run_dict).T

        season_df = team_results.reindex(all_team_ids).fillna(0) + run_df.reindex(all_team_ids).fillna(0)
        season_df = season_df.sort_values(by='points', ascending=False).astype(int)
        season_df['rank'] = np.arange(1, season_df.shape[0] + 1)

        for team_id, run_results in season_df.iterrows():
            simulation_results[team_id]['rank'][run_results['rank']]['n'] += 1
            try:
                simulation_results[team_id]['points'][run_results['points']]['n'] += 1
            except KeyError:
                simulation_results[team_id]['points'][run_results['points']] = {'n': 1}
    # Add probabilities
    for team_id in all_team_ids:
        for i in range(1, all_team_ids.size + 1):
            simulation_results[team_id]['rank'][i]['p'] = simulation_results[team_id]['rank'][i]['n'] / n_runs
        for i, _ in simulation_results[team_id]['points'].items():
            simulation_results[team_id]['points'][i]['p'] = simulation_results[team_id]['points'][i]['n'] / n_runs
    return simulation_results


def simulate_season_standing(payload, team_results, team_names, sim_mode='1X2'):
    n_runs = payload['n_runs']
    match_ids = payload['game_id']
    run_list = []
    for ind_match, match_id in enumerate(match_ids):
        if sim_mode == '1X2':
            outcomes = np.random.multinomial(
                n=1,
                pvals=[
                    payload['pred_1X2_home'][ind_match],
                    payload['pred_1X2_draw'][ind_match],
                    payload['pred_1X2_away'][ind_match],
                ],
                size=(payload['n_runs'])
            )
            # New format
            home_df = pd.DataFrame(
                data=np.hstack((np.arange(1, n_runs + 1).reshape(-1, 1), outcomes)),
                columns=['run', 'W', 'D', 'L']
            )
            home_df['team_id'] = payload['home_team_id'][ind_match]
            away_df = pd.DataFrame(
                data=np.hstack((np.arange(1, n_runs + 1).reshape(-1, 1), np.fliplr(outcomes))),
                columns=['run', 'W', 'D', 'L']
            )
            away_df['team_id'] = payload['away_team_id'][ind_match]
            run_df = pd.concat([home_df, away_df], axis=0)
            run_df['G'] = 1
            run_df['points'] = (run_df[['W', 'D', 'L']] * np.array([3, 1, 0])).sum(axis=1)
            run_list.append(run_df)
        elif sim_mode == 'score':
            exact_score_probs = np.array(payload['pred_exact_home_away'][ind_match]) / \
                                np.sum(payload['pred_exact_home_away'][ind_match])
            scores = np.random.multinomial(
                n=1,
                pvals=exact_score_probs,
                size=(payload['n_runs'])
            )
            ind_scores = np.argmax(scores, axis=1)
            aux_home = np.tile(np.arange(0, payload['n_exact_home']), (payload['n_exact_away'], 1)).flatten(order='F')
            aux_away = np.tile(np.arange(0, payload['n_exact_away']), (payload['n_exact_home']))
            home_scores = aux_home[ind_scores] + payload['current_score_home'][ind_match]
            away_scores = aux_away[ind_scores] + payload['current_score_away'][ind_match]
            away_outcomes = np.zeros((payload['n_runs'], 3)).astype(int)
            away_outcomes[np.arange(payload['n_runs']), np.sign(home_scores - away_scores) + 1] = 1
            home_outcomes = np.fliplr(away_outcomes)

            home_df = pd.DataFrame(
                data=np.hstack((np.arange(1, n_runs + 1).reshape(-1, 1), home_outcomes, home_scores.reshape(-1, 1), away_scores.reshape(-1, 1))),
                columns=['run', 'W', 'D', 'L', 'GF', 'GA']
            )
            home_df['team_id'] = payload['home_team_id'][ind_match]
            away_df = pd.DataFrame(
                data=np.hstack((np.arange(1, n_runs + 1).reshape(-1, 1), away_outcomes, away_scores.reshape(-1, 1), home_scores.reshape(-1, 1))),
                columns=['run', 'W', 'D', 'L', 'GF', 'GA']
            )
            away_df['team_id'] = payload['away_team_id'][ind_match]
            run_df = pd.concat([home_df, away_df], axis=0)
            run_df['G'] = 1
            run_df['points'] = (run_df[['W', 'D', 'L']] * np.array([3, 1, 0])).sum(axis=1)
            run_list.append(run_df)
    all_runs_df = pd.concat(run_list, axis=0)

    # Aggregate simulated game results by run
    agg_runs_df = all_runs_df.groupby(['run', 'team_id']).sum().reset_index(drop=False)

    # Add potentially missing teams (teams present in observed results but without any remaining games)
    missing_team_ids = np.setdiff1d(team_results.index, agg_runs_df.team_id)
    aux_ind = pd.MultiIndex.from_product([np.arange(1, n_runs + 1), missing_team_ids], names=['run', 'team_id'])
    missing_df = pd.DataFrame(data=0, index=aux_ind, columns=['W', 'D', 'L', 'GF', 'GA', 'G', 'points']).reset_index(drop=False)
    agg_runs_df = pd.concat([agg_runs_df, missing_df], axis=0)

    # Add observed results from past games
    sum_cols = ['G', 'W', 'D', 'L', 'GF', 'GA', 'points']
    prev_df = team_results.reindex(agg_runs_df.team_id.values, columns=sum_cols).fillna(0).astype(int)
    agg_runs_df[sum_cols] = agg_runs_df[sum_cols].values + prev_df.values
    agg_runs_df['GD'] = agg_runs_df['GF'] - agg_runs_df['GA']
    agg_runs_df['random'] = np.random.random(size=agg_runs_df.shape[0])  # Randomize rank when points are tied
    agg_runs_df.sort_values(by=['run', 'points', 'GD', 'GF', 'random'], ascending=False, inplace=True)

    # Calculate rank ("proper" method, tagging teams in order from 1 to N in each season run - more costly)
    # agg_runs_df['rank'] = 1
    # reset_points = np.append(True, np.diff(agg_runs_df['run']) != 0)
    # temp_cumsum = agg_runs_df['rank'].cumsum().values
    # first_event_index = np.tile(np.nan, agg_runs_df.shape[0])
    # first_event_index[reset_points] = np.flatnonzero(reset_points)
    # first_event_index = pd.Series(first_event_index).fillna(method='ffill').values.astype(int)
    # rank = temp_cumsum - temp_cumsum[first_event_index] + 1

    # Calculate rank ("cheap" method, repeating a [1, 2, 3, ..., N] array as many times as runs)
    all_team_ids = np.unique(payload['home_team_id'] + payload['away_team_id'] + team_results.index.tolist())
    agg_runs_df['rank'] = np.arange(1, all_team_ids.size + 1).tolist() * n_runs

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
    aux = team_results[['points', 'GF', 'GA']].copy()
    aux['GD'] = aux['GF'] - aux['GA']
    ordered_team_ids = aux.sort_values(by=['points', 'GD', 'GF'], ascending=False).index.values
    final_list = [{'id': team_id,
                   'name': avg_points_df.loc[team_id, 'team_name'],
                   'current': team_results.loc[team_id].to_dict(),
                   'predicted': {'average': avg_points_df.loc[team_id, 'points'], 'rank': {}, 'points': {}}}
                  for team_id in ordered_team_ids]
    for (team_id, rank), sim_results in rank_dist.to_dict(orient='index').items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]['predicted']['rank'][rank] = sim_results
    for (team_id, rank), sim_results in points_dist.to_dict(orient='index').items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]['predicted']['points'][rank] = sim_results

    return final_list


def simulate_season_standing_with_tiebreakers(match_preds, match_results, team_names, comp_id):
    n_runs = match_preds['n_runs']
    match_ids = match_preds['game_id']
    run_list = []
    for ind_match, match_id in enumerate(match_ids):
        exact_score_probs = np.array(match_preds['pred_exact_home_away'][ind_match]) / \
                            np.sum(match_preds['pred_exact_home_away'][ind_match])
        scores = np.random.multinomial(
            n=1,
            pvals=exact_score_probs,
            size=(match_preds['n_runs'])
        )
        ind_scores = np.argmax(scores, axis=1)
        aux_home = np.tile(np.arange(0, match_preds['n_exact_home']), (match_preds['n_exact_away'], 1)).flatten(order='F')
        aux_away = np.tile(np.arange(0, match_preds['n_exact_away']), (match_preds['n_exact_home']))
        home_scores = aux_home[ind_scores] + match_preds['current_score_home'][ind_match]
        away_scores = aux_away[ind_scores] + match_preds['current_score_away'][ind_match]
        away_outcomes = np.zeros((match_preds['n_runs'], 3)).astype(int)
        away_outcomes[np.arange(match_preds['n_runs']), np.sign(home_scores - away_scores) + 1] = 1
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
    all_runs_df['GF_away'] = all_runs_df['GF'].where(all_runs_df['is_home'] == 0, 0)

    # Aggregate simulated game results by run
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
    for (team_id, rank), sim_results in rank_dist.to_dict(orient='index').items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]['predicted']['rank'][rank] = sim_results
    for (team_id, rank), sim_results in points_dist.to_dict(orient='index').items():
        idx_team = np.flatnonzero(ordered_team_ids == team_id)[0]
        final_list[idx_team]['predicted']['points'][rank] = sim_results
    return final_list


def simulate_euro_2020(match_preds, match_results, team_names, comp_id):
    n_runs = match_preds['n_runs']
    match_ids = match_preds['game_id']
    run_list = []
    for ind_match, match_id in enumerate(match_ids):
        exact_score_probs = np.array(match_preds['pred_exact_home_away'][ind_match]) / \
                            np.sum(match_preds['pred_exact_home_away'][ind_match])
        scores = np.random.multinomial(
            n=1,
            pvals=exact_score_probs,
            size=(match_preds['n_runs'])
        )
        ind_scores = np.argmax(scores, axis=1)
        aux_home = np.tile(np.arange(0, match_preds['n_exact_home']), (match_preds['n_exact_away'], 1)).flatten(order='F')
        aux_away = np.tile(np.arange(0, match_preds['n_exact_away']), (match_preds['n_exact_home']))
        home_scores = aux_home[ind_scores] + match_preds['current_score_home'][ind_match]
        away_scores = aux_away[ind_scores] + match_preds['current_score_away'][ind_match]
        away_outcomes = np.zeros((match_preds['n_runs'], 3)).astype(int)
        away_outcomes[np.arange(match_preds['n_runs']), np.sign(home_scores - away_scores) + 1] = 1
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
    all_runs_df['GF_away'] = all_runs_df['GF'].where(all_runs_df['is_home'] == 0, 0)

    # Aggregate simulated game results by run
    agg_runs_df = all_runs_df.groupby(['run', 'team_id']).sum().reset_index(drop=False)

    # Ranking criteria
    ranking_criteria = {
        '8tddm56zbasf57jkkay4kbf11': ['points', 'H2H_points', 'H2H_GD', 'H2H_GF', 'GD', 'GF'],  # UEFA European Championships,
    }[comp_id]
    # Preliminary criteria (anything that does not involve head-to-head subsetting)
    preliminary_criteria = []
    for i in ranking_criteria:
        if i.find('H2H_') == 0:
            break
        else:
            preliminary_criteria.append(i)

    # Add observed results from past games
    agg_match_results, match_team_results = aggregate_match_results_by_team(match_results, return_detail=True)
    sum_cols = ['G', 'W', 'D', 'L', 'GF', 'GA', 'points']
    prev_df = agg_match_results.reindex(agg_runs_df.team_id.values, columns=sum_cols).fillna(0).astype(int)
    agg_runs_df[sum_cols] = agg_runs_df[sum_cols].values + prev_df.values
    agg_runs_df['GD'] = agg_runs_df['GF'] - agg_runs_df['GA']

    if comp_id == '8tddm56zbasf57jkkay4kbf11':
        agg_runs_df['group_id'] = agg_runs_df['team_id'].map(EURO_2020_GROUPS)
    agg_runs_df = agg_runs_df.sort_values(by=['group_id', 'run'] + preliminary_criteria, ascending=False).reset_index(drop=True)

    # Identify ties (2 or more teams with identical values in the (preliminary) sorting features)
    aux = agg_runs_df.drop_duplicates(subset=['group_id', 'run'] + preliminary_criteria).index.values
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
    agg_runs_df = agg_runs_df.sort_values(by=['group_id', 'run'] + ranking_criteria, ascending=False).reset_index(drop=True)

    # Calculate rank ("cheap" method, repeating a [1, 2, 3, ..., N] array as many times as runs)
    n_groups = np.unique(list(EURO_2020_GROUPS.values())).size
    n_teams = len(EURO_2020_GROUPS)
    agg_runs_df['rank'] = np.arange(1, int(n_teams / n_groups) + 1).tolist() * n_runs * n_groups

    # Calculate best third qualifiers
    agg_runs_df = calculate_best_thirds(agg_runs_df, n_groups, n_runs)

    # Add Round of 16 matchup information
    agg_runs_df = add_round_of_16_matchups(agg_runs_df)

    # Distribution of group stage rankings
    rank_dist = agg_runs_df.groupby(['team_id', 'rank'])[['run']].count().rename(columns={'run': 'n'})
    rank_dist['p'] = rank_dist['n'] / n_runs
    # Distribution of group stage qualifying
    agg_runs_df['qualify'] = np.where((agg_runs_df['rank'].values <= 2) | (agg_runs_df['rank_3rd'].fillna(7).values <= 4), 'yes', 'no')
    rank_qual = agg_runs_df.groupby(['team_id', 'qualify'])[['run']].count().rename(columns={'run': 'n'})
    rank_qual['p'] = rank_qual['n'] / n_runs
    # Distribution of group stage league points
    points_dist = agg_runs_df.groupby(['team_id', 'points'])[['run']].count().rename(columns={'run': 'n'})
    points_dist['p'] = points_dist['n'] / n_runs

    # Merge results into a single dictionary
    avg_points_df = agg_runs_df.groupby('team_id')[['points']].mean().sort_values(by='points', ascending=False)
    avg_points_df['team_name'] = team_names.loc[avg_points_df.index].values
    ordered_team_ids = list(EURO_2020_GROUPS.keys())

    final_list = [{'id': team_id,
                   'group': EURO_2020_GROUPS[team_id],
                   'name': avg_points_df.loc[team_id, 'team_name'],
                   'current': agg_match_results.loc[team_id].to_dict() if team_id in agg_match_results.index else [],
                   'predicted': {'average': avg_points_df.loc[team_id, 'points'], 'rank': {}, 'points': {}, 'qualify': {}}}
                  for team_id in ordered_team_ids]
    for (team_id, rank), sim_results in rank_dist.to_dict(orient='index').items():
        idx_team = ordered_team_ids.index(team_id)
        final_list[idx_team]['predicted']['rank'][rank] = sim_results
    for (team_id, qualify_str), sim_results in rank_qual.to_dict(orient='index').items():
        idx_team = ordered_team_ids.index(team_id)
        final_list[idx_team]['predicted']['qualify'][qualify_str] = sim_results
    for (team_id, rank), sim_results in points_dist.to_dict(orient='index').items():
        idx_team = ordered_team_ids.index(team_id)
        final_list[idx_team]['predicted']['points'][rank] = sim_results

    return final_list


def calculate_best_thirds(df, n_groups, n_runs):
    mask_thirds = df['rank'].values == 3
    third_df = df.loc[mask_thirds, ['group_id', 'run', 'team_id', 'points', 'GD', 'GF']].copy()
    third_df = third_df.sort_values(by=['run', 'points', 'GD', 'GF'], ascending=False).reset_index(drop=True)
    third_df['rank_3rd'] = np.arange(1, n_groups + 1).tolist() * n_runs
    df = df.merge(right=third_df[['run', 'team_id', 'rank_3rd']], how='left', on=['run', 'team_id'])
    return df


def add_round_of_16_matchups(df):
    # Assign matchups between first and second placed teams first
    matchup_df = pd.DataFrame([
        {'group_id': 'A', 'rank': 1, 'matchup_group_id': 'C', 'matchup_rank': 2},  # 1A v 2C
        {'group_id': 'C', 'rank': 2, 'matchup_group_id': 'A', 'matchup_rank': 1},
        {'group_id': 'A', 'rank': 2, 'matchup_group_id': 'B', 'matchup_rank': 2},  # 2A v 2B
        {'group_id': 'B', 'rank': 2, 'matchup_group_id': 'A', 'matchup_rank': 2},
        {'group_id': 'D', 'rank': 2, 'matchup_group_id': 'E', 'matchup_rank': 2},  # 2D v 2E
        {'group_id': 'E', 'rank': 2, 'matchup_group_id': 'D', 'matchup_rank': 2},
        {'group_id': 'D', 'rank': 1, 'matchup_group_id': 'F', 'matchup_rank': 2},  # 1D v 2F
        {'group_id': 'F', 'rank': 2, 'matchup_group_id': 'D', 'matchup_rank': 1},
        # {'group_id': 'B', 'rank': 1, 'matchup_group_id': 'U', 'matchup_rank': -1},  # 1B v 3A/D/E/F
        # {'group_id': 'C', 'rank': 1, 'matchup_group_id': 'U', 'matchup_rank': -1},  # 1C v 3D/E/F
        # {'group_id': 'E', 'rank': 1, 'matchup_group_id': 'U', 'matchup_rank': -1},  # 1E v 3A/B/C/D
        # {'group_id': 'F', 'rank': 1, 'matchup_group_id': 'U', 'matchup_rank': -1},  # 1F v 3A/B/C
    ])
    df = df.merge(right=matchup_df, how='left', on=['group_id', 'rank'])

    # Now add matchups that involve a third placed team
    # These assignments are based on the 2016 ones (https://www.eurosport.com/football/euro-2016/2016/third-place-at-euro-2016-how-it-works-who-plays-whom-and-who-will-qualify-for-the-last-16_sto5651583/story.shtml)
    # Need to find the 2021 version
    aux = pd.DataFrame([
        {'pattern': 'ABCD', '1F': '3C', '1B': '3D', '1E': '3A', '1C': '3B'},
        {'pattern': 'ABCE', '1F': '3C', '1B': '3A', '1E': '3B', '1C': '3E'},
        {'pattern': 'ABCF', '1F': '3C', '1B': '3A', '1E': '3B', '1C': '3F'},
        {'pattern': 'ABDE', '1F': '3D', '1B': '3A', '1E': '3B', '1C': '3E'},
        {'pattern': 'ABDF', '1F': '3D', '1B': '3A', '1E': '3B', '1C': '3F'},
        {'pattern': 'ABEF', '1F': '3E', '1B': '3A', '1E': '3B', '1C': '3F'},
        {'pattern': 'ACDE', '1F': '3C', '1B': '3D', '1E': '3A', '1C': '3E'},
        {'pattern': 'ACDF', '1F': '3C', '1B': '3D', '1E': '3A', '1C': '3F'},
        {'pattern': 'ACEF', '1F': '3C', '1B': '3A', '1E': '3F', '1C': '3E'},
        {'pattern': 'ADEF', '1F': '3D', '1B': '3A', '1E': '3F', '1C': '3E'},
        {'pattern': 'BCDE', '1F': '3C', '1B': '3D', '1E': '3B', '1C': '3E'},
        {'pattern': 'BCDF', '1F': '3C', '1B': '3D', '1E': '3B', '1C': '3F'},
        {'pattern': 'BCEF', '1F': '3E', '1B': '3C', '1E': '3B', '1C': '3F'},
        {'pattern': 'BDEF', '1F': '3E', '1B': '3D', '1E': '3B', '1C': '3F'},
        {'pattern': 'CDEF', '1F': '3C', '1B': '3D', '1E': '3F', '1C': '3E'},
        ])
    matchup_3rd_teams_df = pd.DataFrame(columns=['pattern', 'rank', 'group_id', 'matchup_3rd_rank', 'matchup_3rd_group_id'])
    for _, i in aux.iterrows():
        for first_str in ['1F', '1B', '1E', '1C']:
            matchup_3rd_teams_df.loc[matchup_3rd_teams_df.shape[0]] = {
                'pattern': i['pattern'],
                'rank': int(first_str[0]),
                'group_id': first_str[1],
                'matchup_3rd_rank': int(i[first_str][0]),
                'matchup_3rd_group_id': i[first_str][1],
            }
            matchup_3rd_teams_df.loc[matchup_3rd_teams_df.shape[0]] = {
                'pattern': i['pattern'],
                'rank': int(i[first_str][0]),
                'group_id': i[first_str][1],
                'matchup_3rd_rank': int(first_str[0]),
                'matchup_3rd_group_id': first_str[1],
            }

    mask_best_thirds = df.rank_3rd.fillna(10).values <= 4
    best_third_groups = df.loc[mask_best_thirds, ['run', 'group_id']].sort_values(by='group_id').groupby('run').sum()
    df['pattern'] = best_third_groups.loc[df.run.values].values
    df = df.merge(right=matchup_3rd_teams_df, how='left', on=['pattern', 'group_id', 'rank'])

    mask_3rd_matchup = ~df.matchup_3rd_group_id.isna().values
    df.loc[mask_3rd_matchup, ['matchup_group_id', 'matchup_rank']] = df.loc[mask_3rd_matchup, ['matchup_3rd_group_id', 'matchup_3rd_rank']].values
    df.drop(columns=['pattern', 'matchup_3rd_group_id', 'matchup_3rd_rank'], inplace=True)

    return df


def simulate_team_season(payload, team_results, target_list, team_names):
    n_runs = payload['n_runs']
    match_ids = payload['game_id']
    output_dict = {id: {'name': name} for id, name in team_names.iteritems()}

    for target in target_list:
        run_list = []
        for ind_match, match_id in enumerate(match_ids):
            # Home team samples
            home_probs = np.array(payload[target + '_home'][ind_match]) / np.sum(payload[target + '_home'][ind_match])
            home_match_runs = np.random.multinomial(
                n=1,
                pvals=home_probs,
                size=(payload['n_runs'])
            )
            home_match_target_values = (home_match_runs * np.array(np.arange(0, home_match_runs.shape[1]))).sum(axis=1)
            # Away team samples
            away_probs = np.array(payload[target + '_away'][ind_match]) / np.sum(payload[target + '_away'][ind_match])
            away_match_runs = np.random.multinomial(
                n=1,
                pvals=away_probs,
                size=(payload['n_runs'])
            )
            away_match_target_values = (away_match_runs * np.array(np.arange(0, away_match_runs.shape[1]))).sum(axis=1)
            # Organize samples by team ID
            home_df = pd.DataFrame({
                'team_id': payload['home_team_id'][ind_match],
                'run': np.arange(1, n_runs + 1),
                target: home_match_target_values,
                target + '_opp': away_match_target_values,
            })
            away_df = pd.DataFrame({
                'team_id': payload['away_team_id'][ind_match],
                'run': np.arange(1, n_runs + 1),
                target: away_match_target_values,
                target + '_opp': home_match_target_values,
            })
            run_df = pd.concat([home_df, away_df], axis=0)
            run_df['games'] = 1
            run_list.append(run_df)

        # Aggregate simulated game results by run
        all_runs_df = pd.concat(run_list, axis=0)
        agg_runs_df = all_runs_df.groupby(['run', 'team_id']).sum().reset_index(drop=False)

        # Add potentially missing teams (teams present in observed results but without any remaining games)
        missing_team_ids = np.setdiff1d(team_results.index, agg_runs_df.team_id)
        if missing_team_ids.size > 0:
            aux_ind = pd.MultiIndex.from_product([np.arange(1, n_runs + 1), missing_team_ids], names=['run', 'team_id'])
            missing_df = pd.DataFrame(data=0, index=aux_ind, columns=['games', target, target + '_opp']).reset_index(drop=False)
            agg_runs_df = pd.concat([agg_runs_df, missing_df], axis=0)

        # Add observed results from past games
        sum_cols = ['games', target, target + '_opp']
        prev_df = team_results.reindex(agg_runs_df.team_id.values, columns=sum_cols).fillna(0).astype(int)
        agg_runs_df[sum_cols] = agg_runs_df[sum_cols].values + prev_df.values

        agg_runs_df.set_index('team_id', inplace=True)
        for team_id in team_names.index.values:
            output_dict[team_id]['games'] = team_results.loc[team_id, 'games']
            for aux_str in ['', '_opp']:
                output_dict[team_id][target + aux_str] = {}
                output_dict[team_id][target + aux_str]['current'] = team_results.loc[team_id, target + aux_str]
                output_dict[team_id][target + aux_str]['predicted'] = {
                    'scalar': agg_runs_df.loc[team_id, target + aux_str].mean(),
                    'dist': {},
                }
                target_value, target_n = np.unique(agg_runs_df.loc[team_id, target + aux_str].values, return_counts=True)
                for (v, n) in zip(target_value, target_n):
                    output_dict[team_id][target + aux_str]['predicted']['dist'][int(v)] = {'n': int(n), 'p': n / n_runs}
    return output_dict


def simulate_player_season(player_results, payload, target_list):
    n_runs = payload['n_runs']
    match_ids = payload['game_id']
    output_dict = {
        'id': payload['player_id'],
        'name': payload['player_name']
    }
    for target in target_list:
        run_list = []
        for ind_match, match_id in enumerate(match_ids):
            target_probs = np.array(payload[target]['exact'][ind_match]) / np.sum(payload[target]['exact'][ind_match])
            match_runs = np.random.multinomial(
                n=1,
                pvals=target_probs,
                size=(payload['n_runs'])
            )
            match_target_values = (match_runs * np.array(np.arange(0, match_runs.shape[1]))).sum(axis=1)
            run_list.append(match_target_values)

        current_target_value = pd.DataFrame(player_results)[target].sum()
        end_of_season_values = np.array(run_list).sum(axis=0) + current_target_value

        # Format output object
        target_value, target_n = np.unique(end_of_season_values, return_counts=True)
        output_dict[target] = {
            'current': current_target_value,
            'predicted': {
                'average': end_of_season_values.mean(),
                'dist': {}
            }
        }
        for (v, n) in zip(target_value, target_n):
            output_dict[target]['predicted']['dist'][int(v)] = {'n': int(n), 'p': n / n_runs}

    return output_dict


def get_tournament_schedule(cal_id):
    url = opta_feed_url_builder(query='MA0', identifier=cal_id)
    with open_requests_session() as s:
        r = s.get(url)
        r.raise_for_status()
        data = json.loads(r.content.decode())
    return build_schedule_data_frame(data).sort_values(by='date', ascending=True)


def build_schedule_data_frame(data):
    match_dicts = []
    for match_day in data['matchDate']:
        for match in match_day['match']:
            match_date = parse_datetime(match['date'], match['time'])
            try:
                match_dicts.append({
                    'game_id': match['id'],
                    'date': match_date,
                    'home_team_id': match['homeContestantId'],
                    'away_team_id': match['awayContestantId'],
                    'home_team_name': match['homeContestantName'],
                    'away_team_name': match['awayContestantName'],
                })
            except KeyError:
                continue
    return pd.DataFrame(match_dicts)


def parse_datetime(date_str, time_str):
    if time_str == '':
        parsed_datetime = datetime.datetime.strptime(date_str[:-1], '%Y-%m-%d')
    else:
        parsed_datetime = datetime.datetime.strptime(date_str[:-1] + ' ' + time_str[:-1], '%Y-%m-%d %H:%M:%S')
    return parsed_datetime


def opta_feed_url_builder(query, identifier, supplier=None, entity_type=None):
    if query == 'MFL':  # Match Fixture List
        if entity_type == 'tournament_calendar':
            url = 'http://{0:s}/mfl/{1:s}/fixture/?_fmt=json&_pgSz=1&tmcl={2:s}'.format(
                IP_OR_DNS,
                OUTLET_AUTH_KEY,
                identifier
            )
        else:
            supplier_id = {
                'opta': 'esyxxkmsrcrd719ohjmhuaksy',
                'rb': '9o9erudg04emtwrx3ae67xnvv',
                'uuid': None
            }[supplier]
            if supplier == 'uuid':
                url = 'http://{0:s}/mfl/{1:s}/fixture/{2:s}?_fmt=json'.format(
                    IP_OR_DNS,
                    OUTLET_AUTH_KEY,
                    identifier
                )
            else:
                url = 'http://{0:s}/mfl/{1:s}/fixture?_fmt=json&supl={2:s}&entTp=fixt&entId={3}'.format(
                    IP_OR_DNS,
                    OUTLET_AUTH_KEY,
                    supplier_id,
                    identifier
                )
    elif query == 'MA0':  # Tournament Schedule
        url = 'http://{0:s}/soccerdata/tournamentschedule/{1:s}/{2:s}?_fmt=json'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            identifier
        )
    elif query == 'MA2':  # Match Stats
        url = 'http://{0:s}/soccerdata/matchstats/{1:s}/{2:s}?_fmt=json&detailed=yes'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            identifier,
        )
    elif query == 'MAP':  # Mappings
        valid_suppliers = ["opta", "stats", "perform", "uuid"]
        assert supplier in valid_suppliers, \
            'Valid values for "supplier" parameter for MAP requests are: ' + str(valid_suppliers)
        valid_entity_types = ["fixture", "contestant", "competition", "person"]
        assert entity_type in valid_entity_types, \
            'Valid values for "entity_type" parameter for MAP requests are: ' + str(valid_entity_types)
        if isinstance(identifier, list):
            id_str = ''
            for id in identifier:
                id_str += str(id) + ','
            id_str = id_str[:-1]
        else:
            id_str = str(identifier)
        url = 'http://{0:s}/soccerdata/mappings/{1:s}?_fmt=json&idType=urn:perform:{4:s}:{3:s}&idList={2:s}'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            id_str,
            entity_type,
            supplier
        )
    elif query == 'PE2':  # Player Career
        url = 'http://{0:s}/soccerdata/playercareer/{1:s}?_fmt=json&_rt=c&prsn={2:s}'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            identifier,
        )
    elif query == 'TM3':  # Squads
        url = 'http://{0:s}/soccerdata/squads/{1:s}?_fmt=json&_rt=c&ctst={2:s}&tmcl={3:s}&detailed=yes'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            identifier['team_id'],
            identifier['tcal_id'],
        )
    elif query == 'OT2':  # Tournament Calendars
        url = 'http://{0:s}/soccerdata/tournamentcalendar/{1:s}?_fmt=json&_rt=c&comp={2:s}'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            identifier,
        )
    elif query == 'MA1':  # Fixtures and Results
        url = 'http://{0:s}/soccerdata/match/{1:s}?_fmt=json&_rt=c&tmcl={2:s}{3:s}&live=yes&_ordSrt=asc&_pgSz=400&_pgNm=1'.format(
            IP_OR_DNS,
            OUTLET_AUTH_KEY,
            identifier['tcal_id'],
            '&ctst=' + identifier['team_id'] if identifier['team_id'] is not None else '',
        )
    return url


def open_requests_session():
    s = requests.Session()
    s.headers.update({'referer': '.statsperform.global/'})
    s.headers.update({'host': 'api.performfeeds.com/'})
    return s


def get_game_ids(tcal_id, team_ids):
    schedule = get_tournament_schedule(tcal_id)
    if team_ids == 'all':
        mask_team = np.ones(schedule.shape[0]).astype(bool)
    else:
        assert isinstance(team_ids, list), 'Parameter "team_id" must be "all" or a list of team UUIDs'
        mask_team = (schedule.home_team_id.isin(team_ids)) | (schedule.away_team_id.isin(team_ids))

    # Get a map of team ID -> name
    team_names = pd.concat([
        schedule.set_index('home_team_id')['home_team_name'],
        schedule.set_index('away_team_id')['away_team_name']
    ]).drop_duplicates()

    return schedule.loc[mask_team, ['game_id', 'home_team_id', 'away_team_id']].to_dict(orient='records'), team_names


def get_item_from_dynamo(table_name, key):
    response = dynamo.get_item(TableName=table_name, Key=key)

    def parse_data_dict(d):
        if list(d.keys())[0] == 'N':
            return float(d['N'])
        elif list(d.keys())[0] == 'S':
            return d['S']
        elif list(d.keys())[0] == 'B':
            return d['B']
        elif list(d.keys())[0] == 'L':
            return [parse_data_dict(i) for i in v['L']]
        else:
            raise ValueError('Unexpected data type found: {0}'.format(list(d.keys())[0]))

    if 'Item' in response:
        response_data = {}
        for k, v in response['Item'].items():
            if k in key.keys():
                continue
            response_data[k] = parse_data_dict(v)
        return response_data
    else:
        return None


def get_item_batch_from_dynamo(table_name, key_list):
    n_keys = len(key_list)
    n_batches = int(np.ceil(n_keys/100))
    items = []
    for batch_number in range(n_batches):
        ind_start = 100 * batch_number
        ind_end = min(100 * (batch_number + 1), n_keys)
        batch_key_list = key_list[ind_start:ind_end]
        batch_response = dynamo.batch_get_item(RequestItems={table_name: {'Keys': batch_key_list}})
        assert len(batch_response['UnprocessedKeys']) == 0, 'Dynamo batch query returned unprocessed keys. Fix this.'
        if len(batch_response['Responses'][table_name]) == len(batch_key_list):
            items += batch_response['Responses'][table_name]
        else:
            returned_game_ids = [i['game_id']['S'] for i in batch_response['Responses'][table_name]]
            key_game_ids = [i['game_id']['S'] for i in batch_key_list]
            missing_ids = np.setdiff1d(key_game_ids, returned_game_ids).tolist()
            raise KeyError('Missing game predictions in DynamoDB (table "{0}") for these games: {1}'
                           .format(table_name, missing_ids))

    def parse_data_dict(d):
        if list(d.keys())[0] == 'N':
            return float(d['N'])
        elif list(d.keys())[0] == 'S':
            return d['S']
        elif list(d.keys())[0] == 'B':
            return d['B']
        elif list(d.keys())[0] == 'L':
            return [parse_data_dict(i) for i in v['L']]
        else:
            raise ValueError('Unexpected data type found: {0}'.format(list(d.keys())[0]))

    response_dict = {}
    for item in items:
        response_data = {}
        for k, v in item.items():
            if k in key_list[0].keys():
                continue
            response_data[k] = parse_data_dict(v)
        response_dict[item['game_id']['S']] = response_data

    return response_dict


if __name__ == '__main__':
    event_dict = {
        "tcal_ids": ["4b80uzt9gxak7d1vaa5jp17qi"],
        "team_ids": "all",  # ["9dntj5dioj5ex52yrgwzxrq9l"],
        "targets": ["score"],
        "prediction_types": ["team"],
        "n_runs": 1000
    }
    lambda_handler(event_dict, None)
