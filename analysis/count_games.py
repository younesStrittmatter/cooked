import os
import pandas as pd

def main():
    nr_games = 0
    for folder in os.listdir('./bundles'):
        if folder == '.DS_Store' or folder == 'maps':
            continue
        nr_games += len(os.listdir(os.path.join('./bundles', folder)))
    print(f'Found {nr_games} games')




if __name__ == '__main__':
    main()