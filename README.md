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

### Training
In the folder `spoiled_broth/rl` you can find the training scripts.
- `spoiled_broth/rl/train.py`: This is the main training script. It uses the `spoiled_broth/rl/agent.py` file to create the agent and train it.
- `spoiled_broth/rl/game_env.py`: This is the "main" game environment used for training. It contains all the logic used for calculating rewards, stepping and so on.


### To change maps:
(1) In `games/spoiled_broth/static/index.html`: canvas has to be `tilesize * width x tilesize * height`
(2) In `games/spoiled_broth/game.py` change width and height
(3) Add map to `game/spoiled_broth/maps`

MAKE SURE GAME_TIME is set correctly in wsgi in the game folder

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
dev-scripts/
```

### Giving a collaborator access (collect data + deploy new versions)

Everything lives in one Google Cloud project: **`cooked-455218`** (region `us-central1`).
A collaborator needs to be able to (1) download/manage replay data in
`gs://replay_files/replays/` and (2) build + deploy new versions (Cloud Build,
Artifact Registry, Cloud Run, and the asset/state buckets).

The simplest way to cover all of that — including deleting/purging replays and the
one-time IAM bindings the deploy script sets — is to add them as a project **Owner**.

**1) Owner grants them access to everything (run this yourself):**

```shell
COLLAB="colleague@gmail.com"          # their Google account
gcloud projects add-iam-policy-binding cooked-455218 \
  --member="user:${COLLAB}" \
  --role="roles/owner"
```

`roles/owner` already includes Storage admin (so `analysis/PURGE_PRUNED_PIDS.py`
can delete from GCS), Service Usage (so the `-u cooked-455218` billing flag in
`analysis/download.sh` works), and all Cloud Build / Artifact Registry / Cloud Run /
`iam.serviceAccountUser` permissions needed by `dev-scripts/_deploy.sh`.

**2) The collaborator logs in on their machine:**

```shell
gcloud auth login
gcloud auth application-default login
gcloud config set project cooked-455218
```

After that they can run `analysis/download.sh` to collect data and
`dev-scripts/deploy.sh <game>` to ship new versions.

> Prefer least-privilege instead of Owner? Grant these project roles instead:
> `roles/storage.admin`, `roles/serviceusage.serviceUsageConsumer`,
> `roles/run.admin`, `roles/cloudbuild.builds.editor`,
> `roles/artifactregistry.writer`, and `roles/iam.serviceAccountUser`.
> Note the deploy script's `gcloud projects add-iam-policy-binding` steps need
> `roles/resourcemanager.projectIamAdmin`; since they're idempotent one-time setup,
> run one deploy yourself first so the collaborator won't need that role.

