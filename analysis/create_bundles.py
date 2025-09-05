import json
import os
import shutil

import pandas as pd

REPLAY_FOLDER = './replays'
TICK_LOG_FOLDER = './tick_logs'

CUT_ABLE_LIST = ['tomato']
DELIVER_ABLE_ITEMS = ['tomato_salad']
ASSEMBLE_ABLE_ITEMS = ['tomato_cut', 'plate']

MIN_ACTIONS = 5
FIRST_ACTION_MAX_TICK = 12 * 30  # 30 seconds at 12 ticks per second
LAST_ACTION_MIN_TICK = 3 * 12 * 60 - FIRST_ACTION_MAX_TICK  # 3 minutes at 12 ticks per second, minus FIRST_ACTION_MAX_TICK


def bundle(
        replay_json,
        tick_log_csv,
        name
):
    # get the map name
    map_name = replay_json['config']['init_args']['map_nr']
    # create a folder with the name if it does not exist

    # get player ids
    agent_ids = replay_json['agents'].keys()

    # make a meta file
    meta = {
        'agents': replay_json['agents'],
        'map_name': map_name,
    }

    # get intends
    intents_all = replay_json['intents']

    player_datas = []

    player_conditions = {}

    for player_id in agent_ids:
        player_conditions.update(get_player_stats(player_id, replay_json))

    first_player_condition = None
    last_player_condition = None
    for p_c in player_conditions:
        if int(player_conditions[p_c]['player_nr']) == 1:
            first_player_condition = player_conditions[p_c]
        else:
            last_player_condition = player_conditions[p_c]
    if first_player_condition is None or last_player_condition is None:
        print('Could not determine player conditions, skipping bundle')
        print(player_conditions)
        raise
    first_player_str = f"[p{first_player_condition['start_pos']}_cs({first_player_condition['cutting_speed']})_ws({first_player_condition['walking_speed']})]"
    last_player_str = f"[p{last_player_condition['start_pos']}_cs({last_player_condition['cutting_speed']})_ws({last_player_condition['walking_speed']})]"

    additional_info = first_player_condition['additional_condition_info']
    condition_str = f'{first_player_str}_{last_player_str}_{additional_info}'

    path = f'./bundles/{map_name}_{condition_str}/{name}'
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


    for player_id in agent_ids:
        intents = [i for i in intents_all if i['agent_id'] == player_id]
        player_data = PlayerData(intents, tick_log_csv, player_id, map_name, name, player_conditions[player_id], condition_str)
        player_datas.append(player_data)
        if player_data.total_actions < MIN_ACTIONS:
            print(f"Skipping bundle for {name}, player {player_id} has only {player_data.total_actions} actions")
            shutil.rmtree(path)
            return
        if player_data.first_action_tick is None or player_data.first_action_tick > FIRST_ACTION_MAX_TICK:
            print(
                f"Skipping bundle for {name}, player {player_id} has first action at tick {player_data.first_action_tick}")
            shutil.rmtree(path)
            return
        if player_data.last_action_tick is None or player_data.last_action_tick < LAST_ACTION_MIN_TICK:
            print(
                f"Skipping bundle for {name}, player {player_id} has last action at tick {player_data.last_action_tick}")
            shutil.rmtree(path)
            return

    for data in player_datas:
        data.csv_actions.to_csv(f'{path}/{data.player_id}_actions.csv', index=False)
        data.csv_positions.to_csv(f'{path}/{data.player_id}_positions.csv', index=False)
        meta['final_score'] = data.final_game_score
        meta['agents'][f'{data.player_id}']['player_score'] = data.final_player_score
        meta['agents'][f'{data.player_id}']['total_actions'] = data.total_actions
        meta['agents'][f'{data.player_id}']['first_action_tick'] = data.first_action_tick
        meta['agents'][f'{data.player_id}']['last_action_tick'] = data.last_action_tick

    with open(f'{path}/meta.json', 'w') as f:
        json.dump(meta, f)


def get_player_stats(player_id, replay_json):
    agent_info = replay_json['agents'][f'{player_id}']['initial_state']
    agent_url_params = agent_info['url_params']
    agent_url_params = agent_info["url_params"]
    if 'player_nr' in agent_info:
        player_nr = int(agent_info['player_nr'])

        player_starting_pos_x = int(agent_url_params[f'slot_x_p{player_nr}'][0])
        player_starting_pos_y = int(agent_url_params[f'slot_y_p{player_nr}'][0])
        player_cutting_speed = float(agent_url_params[f'cutting_speed_p{player_nr}'][0])
        player_walking_speed = float(agent_url_params[f'walking_speed_p{player_nr}'][0])
    else:

        player_nr = int(agent_url_params["player"][0][-1])
        player_starting_pos_x = int(agent_url_params[f'slot_x'][0])
        player_starting_pos_y = int(agent_url_params[f'slot_y'][0])
        player_cutting_speed = float(agent_url_params[f'cutting_speed'][0])
        player_walking_speed = float(agent_url_params[f'walking_speed'][0])
    additional_condition_info = 'default'
    if 'additional_condition_info' in agent_url_params:
        additional_condition_info = agent_url_params['additional_condition_info'][0]
    if 'start_time' in replay_json:
        if replay_json['start_time'].startswith('2025-09-04'):
            additional_condition_info = 'ability_hints'

    return {
        player_id: {
            'player_nr': player_nr,
            'start_pos': (player_starting_pos_x, player_starting_pos_y),
            'cutting_speed': player_cutting_speed,
            'walking_speed': player_walking_speed,
            'additional_condition_info': additional_condition_info,
        }
    }


class PlayerData:
    def __init__(self,
                 intents,
                 tick_log_csv,
                 player_id,
                 map_name,
                 game_id,
                 player_condition, condition):

        self.intents = intents

        self.tick_log_csv = tick_log_csv
        self.player_id = player_id

        # get all instances of items exchanged by the player
        self.item_exchange_instances = []
        self.init_item_exchange_instances_raw()
        self.add_item_exchange_targets()

        # get all instances of items put on the cutting board
        self.cutting_board_item_column_names = [
            column for column in tick_log_csv.columns
            if column.startswith('CuttingBoard') and
               column.endswith('item')]
        self.put_on_cutting_board_instances = []
        self.init_cutting_board_items()

        # get all instances of score changes for player and overall score
        self.player_score_column_name = f'{self.player_id}_score'
        self.overall_score_column_name = [
            column for column in tick_log_csv.columns
            if column.endswith('_score') and not 'player' in column
        ][0]
        self.player_score_change_instances = []
        self.overall_score_change_instances = []
        self.init_score_change_instances()
        # get score at the end of the game
        self.final_game_score = _convert_score(
            tick_log_csv[self.overall_score_column_name].iloc[-1])
        self.final_player_score = _convert_score(
            tick_log_csv[self.player_score_column_name].iloc[-1]
        )

        # check instances of items exchanged by the player to find instances
        # where player put an item on the cutting board (in other words, start cutting)
        self.process_player_item_exchange_to_start_cutting()

        # check instances of items exchanged to be delivery actions
        self.process_player_item_exchange_to_deliver()

        # check instances of salads getting assembled
        self.process_player_item_exchange_to_assemble_salad()

        # make target_type and position
        self.process_player_item_exchange_instance_target()

        # make action_long
        self.process_player_item_exchange_to_action_long()

        self.csv_actions = pd.DataFrame(self.item_exchange_instances)
        self.csv_actions.to_dict(orient="list")
        self.csv_actions['player_id'] = self.player_id
        self.csv_actions['map_name'] = map_name
        self.csv_actions['game_id'] = game_id

        self.positions = {
            'x': list(self.tick_log_csv[f'{self.player_id}_x']),
            'y': list(self.tick_log_csv[f'{self.player_id}_y']),
            'tick': list(self.tick_log_csv['tick']),
        }

        # add the cumulative distance walked
        self.positions['distance_walked'] = [0.0]
        for i in range(1, len(self.tick_log_csv)):
            prev_x = self.positions['x'][i - 1]
            prev_y = self.positions['y'][i - 1]
            current_x = self.positions['x'][i]
            current_y = self.positions['y'][i]
            if pd.isna(prev_x) or pd.isna(prev_y) or pd.isna(current_x) or pd.isna(current_y):
                self.positions['distance_walked'].append(self.positions['distance_walked'][-1])
                continue
            else:
                dist = ((float(current_x) - float(prev_x)) ** 2 + (float(current_y) - float(prev_y)) ** 2) ** 0.5
                self.positions['distance_walked'].append(self.positions['distance_walked'][-1] + dist)

        # add the cumulative distance to the actions dataframe (at the tick of the action)
        # for each row in the action dataframe, find the corresponding tick in the position dataframe
        tick_to_dist = dict(zip(self.positions["tick"], self.positions["distance_walked"]))

        if len(self.csv_actions):

            # add distance at the exact action tick (NaN if the tick isn't present)
            self.csv_actions["distance_walked"] = [
                tick_to_dist.get(str(t) if pd.notna(t) else str(t), float("nan"))
                for t in self.csv_actions["tick"]
            ]
            self.csv_actions["distance_walked_since_last_action"] = (
                self.csv_actions["distance_walked"].diff().fillna(0.0)
            )

            tick_to_overall_score = {
                d["tick"]: d["score_change"] for d in self.overall_score_change_instances
            }

            tick_to_player_score = {
                d["tick"]: d["score_change"] for d in self.player_score_change_instances
            }

            # add overall score changes to get cumulative overall score
            cur_overall_score = {
                'tick': [],
                'overall_score': []
            }
            cur_score = 0
            for i in range(len(self.tick_log_csv)):
                tick = int(self.tick_log_csv['tick'].iloc[i])
                if tick in tick_to_overall_score:
                    cur_score += tick_to_overall_score[tick]
                cur_overall_score['tick'].append(tick)
                cur_overall_score['overall_score'].append(cur_score)

            cur_overall_score = dict(zip(cur_overall_score['tick'], cur_overall_score['overall_score']))
            self.csv_actions['overall_score'] = [
                cur_overall_score.get(int(t) if pd.notna(t) else t, 0.0)
                for t in self.csv_actions['tick']
            ]

            # same pattern if you want per-player:
            self.csv_actions['player_score_change'] = [
                tick_to_player_score.get(int(t) if pd.notna(t) else t, 0.0)
                for t in self.csv_actions['tick']
            ]
            self.csv_actions['player_score'] = self.csv_actions['player_score_change'].cumsum()
            self.csv_actions['walking_speed'] = player_condition['walking_speed']
            self.csv_actions['cutting_speed'] = player_condition['cutting_speed']
            self.csv_actions['start_pos'] = str(player_condition['start_pos'])
            self.csv_actions['condition'] = condition

        self.csv_positions = pd.DataFrame(self.positions)
        self.csv_positions['walking_speed'] = player_condition['walking_speed']
        self.csv_positions['cutting_speed'] = player_condition['cutting_speed']
        self.csv_positions['start_pos'] = str(player_condition['start_pos'])

        # total number of actions
        self.total_actions = len(self.csv_actions)

        if self.total_actions > 0:
            self.first_action_tick = int(self.csv_actions['tick'].min())
            self.last_action_tick = int(self.csv_actions['tick'].max())
        else:
            self.first_action_tick = None
            self.last_action_tick = None

    def init_score_change_instances(self):
        player_scores = list(self.tick_log_csv[self.player_score_column_name])
        overall_scores = list(self.tick_log_csv[self.overall_score_column_name])

        for i in range(1, len(player_scores)):
            prev_player_score = _convert_score(player_scores[i - 1])
            current_player_score = _convert_score(player_scores[i])
            if current_player_score != prev_player_score:
                self.player_score_change_instances.append({
                    'tick': i,
                    'score_change': current_player_score - prev_player_score
                })
            prev_overall_score = _convert_score(overall_scores[i - 1])
            current_overall_score = _convert_score(overall_scores[i])
            if current_overall_score != prev_overall_score:
                self.overall_score_change_instances.append({
                    'tick': i,
                    'score_change': current_overall_score - prev_overall_score
                })

    def init_item_exchange_instances_raw(self):
        player_item_column_name = f'{self.player_id}_item'
        held_items = list(self.tick_log_csv[player_item_column_name])

        for i in range(1, len(held_items)):
            prev_item = str(held_items[i - 1])
            current_item = str(held_items[i])
            self.item_exchange_instances += get_player_item_exchange_instance(prev_item, current_item, i)

    def add_item_exchange_targets(self):
        for instance in self.item_exchange_instances:
            tick = instance['tick']
            last_intent = self.get_last_intent(tick)
            if last_intent is None:
                continue
            else:
                instance['target'] = last_intent['action']['target']

    def init_cutting_board_items(self):
        for column in self.cutting_board_item_column_names:
            cutting_board_items = list(self.tick_log_csv[column])
            for i in range(1, len(cutting_board_items)):
                prev_item = str(cutting_board_items[i - 1])
                current_item = str(cutting_board_items[i])
                self.put_on_cutting_board_instances += get_cutting_board_item_exchange_instance(
                    prev_item, current_item, i, column.replace('_item', ''))  # remove '_item' suffix to get board name

    def process_player_item_exchange_instance_target(self):
        for instance in self.item_exchange_instances:
            instance['target_type'] = instance['target'].split('_')[0]
            instance['target_position'] = instance['target'].split('_')[1]

    def process_player_item_exchange_to_action_long(self):
        for instance in self.item_exchange_instances:
            action = instance['action']
            if action == 'deliver':
                instance['action_long'] = f"deliver {instance['item']}"
            if action == 'start cutting':
                instance['action_long'] = f"start cutting {instance['item']}"
            if action == 'assemble salad':
                instance['action_long'] = f"assemble salad"
            if action == 'put down':
                instance[
                    'action_long'] = f"{instance['action']} {instance['item']} on {instance['target_type'].lower()}"
            elif action == 'pick up':
                instance[
                    'action_long'] = f"{instance['action']} {instance['item']} from {instance['target_type'].lower()}"

    def process_player_item_exchange_to_assemble_salad(self):
        for instance in self.item_exchange_instances:
            item = instance['item']
            target = instance['target']
            tick = instance['tick']
            if not _is_salad_ingredient(item):
                continue
            # get item on target
            item_on_target_before = self.get_item_on(target, tick - 1)
            item_on_target = self.get_item_on(target, tick)
            if not _is_salad_ingredient(item_on_target_before):
                continue
            if item == item_on_target_before:
                continue
            if not _is_deliverable(item_on_target):
                continue
            instance['action'] = 'assemble salad'

    def get_item_on(self, name, tick):
        column_name = f'{name}_item'
        if column_name not in self.tick_log_csv.columns:
            return None
        # get the row with tick == tick
        row = self.tick_log_csv.loc[tick, column_name]
        if pd.isna(row):
            return None
        return str(row)

    def process_player_item_exchange_to_deliver(self):
        for instance in self.item_exchange_instances:
            if not _is_deliverable(instance['item']):
                continue
            if not _is_deliver_target(instance['target']):
                continue
            if not instance['action'] == 'put down':
                continue
            # check if the score increased in this next tick
            score_changes_on_this_tick = [el for el in self.player_score_change_instances if
                                          el['tick'] == instance['tick']]
            if not score_changes_on_this_tick:
                continue
            instance['action'] = 'deliver'

    def process_player_item_exchange_to_start_cutting(self):
        for instance in self.item_exchange_instances:
            self.process_player_item_exchange_instance_start_cutting(
                instance)

    def process_player_item_exchange_instance_start_cutting(self, instance):
        item = instance['item']
        action = instance['action']
        tick = instance['tick']
        if action != 'put down':
            return
        if item not in CUT_ABLE_LIST:
            return
        # get something put on the cutting board in the same tick
        put_on_cutting_board = [el for el in self.put_on_cutting_board_instances if el['tick'] == tick]
        if not put_on_cutting_board:
            return
        for el in put_on_cutting_board:
            if el['item'] == item:
                # check if player clicked on the as the last intent
                last_intent = self.get_last_intent(tick)
                if last_intent['action']['target'] == el['board']:
                    instance['action'] = 'start cutting'

    def get_last_intent(self, tick):
        intents_before = [intent for intent in self.intents if intent['tick'] < tick]
        if not intents_before:
            return None
        return intents_before[-1]


def get_player_item_exchange_instance(item_prev, item_current, tick):
    res = []
    if item_prev == item_current:
        return res
    if str(item_prev) != 'nan':
        res.append({
            'tick': tick,
            'item': item_prev,
            'action': 'put down'
        })
    if str(item_current) != 'nan':
        res.append({
            'tick': tick,
            'item': item_current,
            'action': 'pick up'
        })
    return res


def get_cutting_board_item_exchange_instance(item_prev, item_current, tick, cutting_board_name):
    if item_prev == item_current or item_current.endswith('cut'):
        return []
    return [{
        'tick': tick,
        'item': item_current,
        'board': cutting_board_name,
    }]


def main():
    for filename in os.listdir(REPLAY_FOLDER):
        if filename.endswith('.json'):  # Process only JSON files
            filepath_json = os.path.join(REPLAY_FOLDER, filename)
            filepath_csv = os.path.join(TICK_LOG_FOLDER, filename.replace('.json', '.csv'))
            filename = filename.replace('.json', '')
            json_data = json.load(open(filepath_json))
            if not os.path.exists(filepath_csv):
                print(f"Tick log for {filename} not found, skipping...")
                continue
            csv_data = pd.read_csv(
                filepath_csv,
                sep=None,  # auto-detect delimiter
                engine='python',  # more tolerant parser
                on_bad_lines='warn',  # log bad rows; don't crash
                encoding='utf-8',  # try 'latin-1' if this fails
                dtype=str  # avoid dtype inference surprises
            )
            # check if dir already exists
            # if os.path.exists(f'./bundles/{filename}'):
            #     print(f"Bundle for {filename} already exists, skipping...")
            #     continue

            bundle(json_data, csv_data, filename)


def _is_deliverable(item):
    return str(item).endswith('_salad')


def _is_deliver_target(target):
    return str(target).startswith('Delivery_')


def _is_salad_ingredient(item):
    return str(item) in ASSEMBLE_ABLE_ITEMS


def _convert_score(score):
    if str(score) == 'nan':
        return 0
    try:
        return float(score)
    except:
        return 0


if __name__ == '__main__':
    main()
