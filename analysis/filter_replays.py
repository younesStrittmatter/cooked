import os
import json

from tqdm import tqdm

REPLAY_FOLDER_RAW = './replays_raw/replays'
REPLAY_FOLDER = './replays'

EXPECTED_NR_OF_AGENTS = 2
DEFAULT_MAP_PREFIX = ""
DEFAULT_ALLOWED_CONDITIONS = {"mixed", "optimal", "superstar", "asymetric_slow_walker", "mixed_extreme"}


def _float_param(url_params, key):
    try:
        return float(url_params[key][0])
    except Exception:
        return None


def _same(a, b):
    return a is not None and abs(a - b) < 1e-9


def _player_conditions(json_data):
    conditions = {}
    additional_info = "default"
    for agent in json_data.get("agents", {}).values():
        state = agent.get("initial_state", {})
        url_params = state.get("url_params", {})
        if "additional_condition_info" in url_params:
            additional_info = url_params["additional_condition_info"][0]

        player_nr = state.get("player_nr")
        try:
            if player_nr is not None:
                player_nr = int(player_nr)
                cutting_speed = _float_param(url_params, f"cutting_speed_p{player_nr}")
                walking_speed = _float_param(url_params, f"walking_speed_p{player_nr}")
            else:
                player_nr = int(url_params["player"][0][-1])
                cutting_speed = _float_param(url_params, "cutting_speed")
                walking_speed = _float_param(url_params, "walking_speed")
        except Exception:
            continue

        conditions[player_nr] = {
            "cutting_speed": cutting_speed,
            "walking_speed": walking_speed,
        }
    return conditions, additional_info


def _condition_name(player_conditions):
    p1 = player_conditions.get(1, {})
    p2 = player_conditions.get(2, {})
    p1cs, p1ws = p1.get("cutting_speed"), p1.get("walking_speed")
    p2cs, p2ws = p2.get("cutting_speed"), p2.get("walking_speed")

    if _same(p1cs, 1.0) and _same(p1ws, 1.0) and _same(p2cs, 1.0) and _same(p2ws, 1.0):
        return "superstar"
    if _same(p1cs, 1.0) and _same(p1ws, 0.4) and _same(p2cs, 0.2) and _same(p2ws, 1.0):
        return "mixed"
    # Extreme mixed: P1 full cutter / very slow walker (0.1), P2 full walker / very slow cutter (0.1).
    if _same(p1cs, 1.0) and _same(p1ws, 0.1) and _same(p2cs, 0.1) and _same(p2ws, 1.0):
        return "mixed_extreme"
    # P1 cuts at full speed but walks slowly (0.2); P2 is an unimpaired superstar.
    if _same(p1cs, 1.0) and _same(p1ws, 0.2) and _same(p2cs, 1.0) and _same(p2ws, 1.0):
        return "asymetric_slow_walker"
    if _same(p1ws, 0.7) and _same(p2cs, 0.4):
        return "optimal"
    if _same(p1cs, 0.4) and _same(p2ws, 0.7):
        return "optimal"
    return "other"


def should_include_condition(json_data):
    map_name = str(json_data.get("config", {}).get("init_args", {}).get("map_nr", ""))
    map_prefix = os.getenv("MAP_PREFIX", DEFAULT_MAP_PREFIX)
    include_encouraged = os.getenv("INCLUDE_ENCOURAGED", "1") == "1"
    collision_only = os.getenv("COLLISION_ONLY", "1") == "1"
    allowed_conditions = {
        item.strip()
        for item in os.getenv("ALLOWED_CONDITIONS", ",".join(sorted(DEFAULT_ALLOWED_CONDITIONS))).split(",")
        if item.strip()
    }
    player_conditions, additional_info = _player_conditions(json_data)

    if map_prefix and not map_name.startswith(map_prefix):
        return False
    if not include_encouraged and map_name.startswith("encouraged"):
        return False
    if collision_only and "collision" not in additional_info:
        return False
    if allowed_conditions and _condition_name(player_conditions) not in allowed_conditions:
        return False
    return True


def clear_filtered_replays():
    os.makedirs(REPLAY_FOLDER, exist_ok=True)
    for filename in os.listdir(REPLAY_FOLDER):
        if filename.endswith(".json"):
            os.remove(os.path.join(REPLAY_FOLDER, filename))

def main():
    errors = {
        'unexpected_agent_count': [],
        'mismatched_game_id': [],
        'no_intents': [],
        'only_one_player_intent': [],
        'invalid_json': [],
        'excluded_condition': []
    }
    clear_filtered_replays()
    for filename in tqdm(os.listdir(REPLAY_FOLDER_RAW)):
        if filename.endswith('.json'):  # Process only JSON files
            filepath_in = os.path.join(REPLAY_FOLDER_RAW, filename)
            filepath_out = os.path.join(REPLAY_FOLDER, filename)
            try:
                json_data = json.load(open(filepath_in))
            except json.JSONDecodeError:
                errors['invalid_json'].append(filename)
                continue
            if not should_include_condition(json_data):
                errors['excluded_condition'].append(filename)
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