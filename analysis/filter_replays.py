import os
import json

from tqdm import tqdm

REPLAY_FOLDER_RAW = './replays_raw/replays'
REPLAY_FOLDER = './replays'

EXPECTED_NR_OF_AGENTS = 2

def main():
    errors = {
        'unexpected_agent_count': [],
        'mismatched_game_id': [],
        'no_intents': [],
        'only_one_player_intent': [],
        'invalid_json': []
    }
    for filename in tqdm(os.listdir(REPLAY_FOLDER_RAW)):
        if filename.endswith('.json'):  # Process only JSON files
            filepath_in = os.path.join(REPLAY_FOLDER_RAW, filename)
            filepath_out = os.path.join(REPLAY_FOLDER, filename)
            try:
                json_data = json.load(open(filepath_in))
            except json.JSONDecodeError:
                errors['invalid_json'].append(filename)
                continue

            agents = json_data["agents"]
            if not len(agents) == EXPECTED_NR_OF_AGENTS:
                errors['unexpected_agent_count'].append(filename)
                continue

            _agents = [el['initial_state'] for el in agents.values()]
            agent_pids = [el['url_params'].get('PROLIFIC_PID', [False])[0] for el in _agents]

            # agent_pids_non_prolific = [el.startswith('non_prolific') for el in agent_pids]
            if not all(agent_pids):
                errors['unexpected_agent_count'].append(filename)
                continue

            game_ids = [el.get("initial_state").get('url_params').get('gameId') for el in agents.values()]
            game_ids = [el[0] for el in game_ids if isinstance(el, list) and len(el) > 0]
            if not len(set(game_ids)) <= 1:
                errors['mismatched_game_id'].append(filename)
                continue

            intents = json_data["intents"]
            if not isinstance(intents, list) or len(intents) <= 0:
                errors['no_intents'].append(filename)
                continue

            player_intents = [intent['agent_id'] for intent in intents if intent.get('agent_id') in agents]
            if len(set(player_intents)) < 2:
                errors['only_one_player_intent'].append(filename)
                continue

            with open(filepath_out, 'w') as dat:
                json.dump(json_data, dat, indent=4)


    with open('filter_report.json', 'w') as rep:
        json.dump(errors, rep, indent=4)


if __name__ == '__main__':
    main()