from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT = Path("./bundles")


def load_actions_pair(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Stable order for reproducibility
    files = sorted(files, key=lambda p: p.name)
    df1 = pd.read_csv(files[0])
    df2 = pd.read_csv(files[1])
    # label sources so you can trace origin
    df1["source"] = files[0].stem
    df2["source"] = files[1].stem
    return df1, df2


def process_together(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    PROCESS BOTH TOGETHER (customize this).
    Default logic:
      - If both have a 'tick' column, do a merge on 'tick' with suffixes.
      - Else, concatenate with a 'source' column preserved.
    """

    merged = pd.concat([df1, df2], ignore_index=True)
    merged['tick'] = pd.to_numeric(merged['tick'], errors='coerce')

    # Within-tick ordering
    PUT_DOWN_ACTIONS = {'put down', 'start cutting', 'assemble salad', 'deliver'}
    PICK_UP_ACTIONS = {'pick up'}

    def _priority(a: str) -> int:
        if a in PUT_DOWN_ACTIONS: return 0
        if a in PICK_UP_ACTIONS:  return 1
        return 0

    merged['_action_priority'] = merged['action'].map(_priority).fillna(0).astype(int)
    merged = (merged
              .sort_values(['tick', 'target', '_action_priority'], kind='stable')
              .reset_index(drop=True))

    # Pre-allocate per-row logs to guarantee same length as merged
    n = len(merged)
    log_last_touched = [None] * n
    log_touched_list = [None] * n
    log_tomato_history = [None] * n
    log_plate_history = [None] * n
    log_collaboration = [None] * n

    def record(i, *, last_touched, touched_list, tomato_history, plate_history, collaboration):
        # store copies to avoid mutation later
        log_last_touched[i] = last_touched
        log_touched_list[i] = None if touched_list is None else list(touched_list)
        log_tomato_history[i] = None if tomato_history is None else list(tomato_history)
        log_plate_history[i] = None if plate_history is None else list(plate_history)
        log_collaboration[i] = collaboration

    # Inventories
    player_ids = merged['player_id'].dropna().unique()
    player_inventory = {pid: None for pid in player_ids}  # map player -> item dict
    world_inventory = {}  # map tile -> item dict

    last_put_down_tick = 0
    # Iterate rows
    for i, row in enumerate(merged.itertuples(index=False)):
        action = row.action
        player = row.player_id
        target = row.target
        item_name = getattr(row, 'item', None)
        target_type = row.target_type
        tick = row.tick

        # Default record for rows we don't handle explicitly
        # (will be overwritten below if relevant)
        record(i, last_touched=None, touched_list=None,
               tomato_history=None, plate_history=None,
               collaboration=None)

        # PICK UP
        if action == 'pick up':
            if target_type == 'Dispenser':
                # spawn a fresh item in hand
                new_item = {
                    'touched_list': [],
                    'last_touched': None,
                    'tomato_history': None,
                    'plate_history': None,
                    'name': item_name,
                }
                player_inventory[player] = new_item
                record(i,
                       last_touched=None,
                       touched_list=[],
                       tomato_history=None,
                       plate_history=None,
                       collaboration=None)
            else:
                # pick from world tile
                itm = world_inventory.get(target)
                if itm is None:
                    # still
                    raise Exception(f'Picking up from {target} which has no item (tick {tick}, player {player})')
                # record pre-pick state
                record(i,
                       last_touched=itm.get('last_touched'),
                       touched_list=itm.get('touched_list', []),
                       tomato_history=itm.get('tomato_history'),
                       plate_history=itm.get('plate_history'),
                       collaboration=(itm.get('last_touched') != player))
                # transfer to player & update touch
                itm['last_touched'] = itm.get('last_touched', None)
                itm['touched_list'] = itm.get('touched_list', [])
                # itm.setdefault('touched_list', []).append(player)
                player_inventory[player] = itm
                # remove from world only if not an "exchange" action, meaning the player is not putting down and picking up the same item in the same tick
                if last_put_down_tick != tick and itm.get('last_touched') == player:
                    world_inventory[target] = None

        # PUT DOWN / START CUTTING / ASSEMBLE SALAD
        elif action in PUT_DOWN_ACTIONS:
            if action == 'put down' or (action == 'start cutting' and item_name == 'tomato'):
                last_put_down_tick = tick
            held = player_inventory.get(player)
            if held is None and action != 'assemble salad':
                raise Exception(f'Player {player} has nothing to put down at tick {tick}')

            if action in {'put down', 'start cutting'}:
                # place on a surface; allow both spellings
                if target_type not in {'Counter', 'Cutting Board', 'CuttingBoard'}:
                    # You can relax this if you have more surfaces
                    pass
                # record pre-place state

                if held:
                    held['last_touched'] = player
                    held.setdefault('touched_list', []).append(player)
                    held['name'] = item_name or held.get('name')
                    world_inventory[target] = held
                    player_inventory[player] = None
                record(i,
                       last_touched=player,
                       touched_list=held.get('touched_list', []) if held else None,
                       tomato_history=held.get('tomato_history') if held else None,
                       plate_history=held.get('plate_history') if held else None,
                       collaboration=None)

            elif action == 'assemble salad':
                # combine player-held + tile item *on target*
                held = player_inventory.get(player)
                counter = world_inventory.get(target)
                if held is None or counter is None:
                    raise Exception(
                        f'Assemble needs held + counter items at tick {tick} (held={held is not None}, counter={counter is not None})')

                # before combining, record the player's held item state


                # choose which contributes tomato vs plate history
                # using item_name from the event as your signal:
                if item_name == 'tomato_cut':
                    counter['tomato_history'] = list(held.get('touched_list', [])) + [player]
                    counter['plate_history'] = list(counter.get('touched_list', []))
                else:
                    counter['tomato_history'] = list(counter.get('touched_list', []))
                    counter['plate_history'] = list(held.get('touched_list', [])) + [player]

                record(i,
                       last_touched=counter.get('last_touched'),
                       touched_list=[player],
                       tomato_history=counter.get('tomato_history'),
                       plate_history=counter.get('plate_history'),
                       collaboration=(counter.get('last_touched') != player))

                # merge touch lists and finalize
                counter['touched_list'] = [player]
                counter['last_touched'] = player
                counter['name'] = 'tomato_salad'  # or keep previous; your call

                # player now free; tile keeps combined item
                player_inventory[player] = None
                world_inventory[target] = counter
            elif action == 'deliver':
                record(
                    i,
                    last_touched=player,
                    touched_list=held.get('touched_list', []) + [player] if held else None,
                    tomato_history=held.get('tomato_history') if held else None,
                    plate_history=held.get('plate_history') if held else None,
                    collaboration=None,
                )
        # else: ignore other actions but keep row alignment via pre-filled record()

    # Build the added columns (same length!)
    df_added = pd.DataFrame({
        'last_touched': log_last_touched,
        'touched_list': log_touched_list,
        'tomato_history': log_tomato_history,
        'plate_history': log_plate_history,
        'is_exchange_collaboration': log_collaboration,
    })

    out = pd.concat([merged.drop(columns=['_action_priority']),
                     df_added], axis=1)
    return out


def main():
    # Find all *_actions.csv and group by their parent directory
    files = list(ROOT.rglob("*_actions.csv"))
    by_parent: dict[Path, list[Path]] = {}
    for f in files:
        by_parent.setdefault(f.parent, []).append(f)

    # Iterate folders with exactly 2 action files
    for folder, file_list in tqdm(sorted(by_parent.items(), key=lambda kv: str(kv[0]))):
        if len(file_list) != 2:
            # Skip or warn; you can print if you want:
            # print(f"Skipping {folder} (found {len(file_list)} *_actions.csv files)")
            continue

        df1, df2 = load_actions_pair(file_list)

        # print(f"Processing folder: {folder}")
        # --- CUSTOM PROCESSING (together) ---
        processed = process_together(df1, df2)
        # ------------------------------------

        out_path = folder / "actions_processed.csv"
        processed.to_csv(out_path, index=False)
        # print(f"Processed: {folder} -> {out_path.name}")


if __name__ == "__main__":
    main()
