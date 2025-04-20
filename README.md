## Cooked Engine

...

## Spoiled-Broth Game

folder: `spoiled_broth`
- `spoiled_broth\main.py`: Starts the game: Play it in the browser on localhost:5000
- It uses a "RandomClickerAgent" as stand-in for a RL agent. You can find the clicker here: `engine/extensions/topDownGridWorld/ai_controller/random_tile_clicker.py` for reference.
- Some interesting parameters for you in the `spoiled_broth\main.py` file:
  - You can add agents in with the `agent_map` dictionary.
  - If you set `n_players` to `0` the game will run without any players.
  - If you set `max_speed` to `True` the game will run at "max speed" using a fixed delta time (set with tick_rate). This is useful for training agents.
  - The `ai_tick_rate` determines how often the AI will make a decision.



Python-Version: 3.13

## Google Cloud

You'll need to setup a Google Cloud project here: https://console.cloud.google.com/

Login:

```shell
gcloud auth login
```

This will open a browser window and ask you to login with your Google account. After logging in, you will be asked to
give permission to the gcloud command line tool to access your Google account.
If you run into problems with the permissions, sometimes you need to use the application default credentials instead of
the user credentials. This is done with the following command:

```shell
gcloud auth application-default login
```

### Set Project

You need to set the project you want to deploy to. You can do this by running the following command. The project ID can be found in the Google Cloud Console. Make sure to use the full project ID (often has the format `project-name-123456`).
```shell
gcloud config set project <project-id>
```

### Deploy

```shell
gcloud app deploy
```

