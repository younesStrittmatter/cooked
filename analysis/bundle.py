import os
import pandas as pd
from tqdm import tqdm


def main():
    df_all = pd.DataFrame()
    df_all_with_history = pd.DataFrame()
    for folder in tqdm(os.listdir('./bundles')):
        if folder == '.DS_Store' or folder == 'maps':
            continue
        for game_id in tqdm(os.listdir(os.path.join('./bundles', folder))):
            if game_id == '.DS_Store':
                continue
            # get the two csv files that end with _positions.csv
            action_files = [f for f in os.listdir(os.path.join('./bundles', folder, game_id)) if f.endswith('_actions.csv')]
            processed_files = [f for f in os.listdir(os.path.join('./bundles', folder, game_id)) if f.endswith('_processed.csv')]
            for pf in action_files:
                df = pd.read_csv(os.path.join('./bundles', folder, game_id, pf))
                # add player
                df_all = pd.concat([df_all, df])
            for pf in processed_files:
                df = pd.read_csv(os.path.join('./bundles', folder, game_id, pf))
                # add player
                df_all_with_history = pd.concat([df_all_with_history, df])

    df_all.to_csv('./combined_actions.csv', index=False)
    df_all_with_history.to_csv('./combined_actions_with_history.csv', index=False)




if __name__ == '__main__':
    main()