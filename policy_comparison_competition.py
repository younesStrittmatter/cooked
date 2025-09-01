import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from spoiled_broth.game import SpoiledBroth
from spoiled_broth.world.tiles import Counter, CuttingBoard
from pathlib import Path

EPOCH = 10000  # Epoch to load
DATETIME = "2023-08-31_10-00-00"  # Date of training run
MAP_NR = "simple_kitchen_competition"  # Map used in training
ATTITUDE_1 = "competitive"  # or "cooperative"
ATTITUDE_2 = "cooperative"  # or "competitive"
INTENT_VERSION = "v3.1"

def generate_key_observations(agent_id="ai_rl_1", agent_food_type="tomato"):
    """
    Generates all key decision point observations for different item states:
    1. Having raw food (own/other) - should look for cutting board
    2. Having cut food (own/other) - should look for counter with plate
    3. Having salad (own/other) - should look for delivery
    """
    game = SpoiledBroth(map_nr="simple_kitchen_competition")
    game.add_agent("ai_rl_1")
    game.add_agent("ai_rl_2")
    
    agent_food_types = {
        "ai_rl_1": agent_food_type,
        "ai_rl_2": "pumpkin"  # Different food type for other agent
    }
    
    observations = []
    agent = game.gameObjects[agent_id]
    
    # Place agent in a valid position
    walkable_positions = []
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            if (game.grid.tiles[x][y] and game.grid.tiles[x][y].is_walkable):
                walkable_positions.append((x, y))
    
    if not walkable_positions:
        raise ValueError("No walkable positions found in the map")
        
    pos = walkable_positions[0]
    agent.x = pos[0] * game.grid.tile_size + game.grid.tile_size // 2
    agent.y = pos[1] * game.grid.tile_size + game.grid.tile_size // 2
    
    # Test scenarios with different items
    scenarios = [
        # Raw food scenarios
        {"item": agent_food_types[agent_id], "label": "own_food"},
        {"item": agent_food_types["ai_rl_2"], "label": "other_food"},
        
        # Cut food scenarios
        {"item": f"{agent_food_types[agent_id]}_cut", "label": "own_food_cut"},
        {"item": f"{agent_food_types['ai_rl_2']}_cut", "label": "other_food_cut"},
        
        # Salad scenarios
        {"item": f"{agent_food_types[agent_id]}_salad", "label": "own_salad"},
        {"item": f"{agent_food_types['ai_rl_2']}_salad", "label": "other_salad"},
    ]
    
    for scenario in scenarios:
        agent.item = scenario["item"]
        obs, _ = game.game_to_obs_matrix_competition(game, agent_id, agent_food_types)
        observations.append({
            "obs": obs,
            "item": scenario["label"],
            "holding": scenario["item"]
        })

    return observations

def compare_policies(policy1, policy2, observations):
    """
    Compare two policies on the key decision points
    Returns percentage of agreement and detailed differences
    """
    agreements = 0
    differences = []
    
    for obs_dict in observations:
        obs = obs_dict["obs"]
        action1 = policy1(obs)
        action2 = policy2(obs)
        
        if action1 == action2:
            agreements += 1
        else:
            differences.append({
                "item": obs_dict["item"],
                "position": obs_dict["position"],
                "action1": action1,
                "action2": action2
            })
    
    agreement_rate = agreements / len(observations)
    return agreement_rate, differences

def analyze_policy(policy, observations):
    """
    Analyze a single policy's behavior on key decision points
    """
    results = {
        "own_food": [],
        "other_food": [],
        "own_salad": [],
        "other_salad": []
    }
    
    for obs_dict in observations:
        obs = obs_dict["obs"]
        action = policy(obs)
        results[obs_dict["item"]].append({
            "position": obs_dict["position"],
            "action": action
        })
    
    return results

def load_policy(map_nr, attitude, date, epoch, game_version="COMPETITION", intent_version=None):
    """
    Load a policy from a specific training run and epoch following the training script structure
    Args:
        map_nr: The map number used in training
        attitude: 'cooperative' or 'competitive'
        date: Training date in format YYYY-MM-DD_HH-MM-SS
        epoch: Epoch number to load
        game_version: 'CLASSIC' or 'COMPETITION'
        intent_version: Optional intent version if used in training
    """
    # Construct path following the training script structure
    base_path = Path("data/samuel_lozano/cooked")
    
    if game_version == "CLASSIC":
        base_path = base_path / "classic"
    elif game_version == "COMPETITION":
        base_path = base_path / "competition"
    
    if intent_version:
        base_path = base_path / intent_version
    
    model_path = (base_path / f"map_{map_nr}" / attitude.lower() / 
                 f"Training_{date}" / f"checkpoint_{epoch}")
    
    model = PPO.load(str(model_path))
    return lambda obs: model.predict(obs, deterministic=True)[0]

def visualize_policy_comparison(policy_results, title):
    """
    Create a bar plot comparing own vs other actions for different item states
    """
    categories = ['Raw Food', 'Cut Food', 'Salad']
    own_actions = []
    other_actions = []
    
    # Map of states to what we consider a "correct" action
    correct_action_map = {
        'own_food': lambda a: a in [3, 4],  # Interact with cutting board
        'other_food': lambda a: a in [3, 4],  # Interact with cutting board
        'own_food_cut': lambda a: a in [3, 4],  # Interact with counter/plate
        'other_food_cut': lambda a: a in [3, 4],  # Interact with counter/plate
        'own_salad': lambda a: a in [3, 4],  # Interact with delivery
        'other_salad': lambda a: a in [3, 4],  # Interact with delivery
    }
    
    # Count correct actions for each category
    category_map = {
        'Raw Food': ['own_food', 'other_food'],
        'Cut Food': ['own_food_cut', 'other_food_cut'],
        'Salad': ['own_salad', 'other_salad']
    }
    
    for cat in categories:
        own_key = category_map[cat][0]
        other_key = category_map[cat][1]
        
        # Count correct actions according to our mapping
        own_correct = sum(1 for action in policy_results[own_key] 
                         if correct_action_map[own_key](action['action']))
        other_correct = sum(1 for action in policy_results[other_key] 
                          if correct_action_map[other_key](action['action']))
        
        # Get total actions for this category
        own_total = len(policy_results[own_key]) or 1  # avoid division by zero
        other_total = len(policy_results[other_key]) or 1
        
        # Convert to percentages
        own_actions.append((own_correct / own_total) * 100)
        other_actions.append((other_correct / other_total) * 100)
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, own_actions, width, label='Own Items')
    ax.bar(x + width/2, other_actions, width, label='Other\'s Items')
    
    ax.set_ylabel('Percentage of Correct Actions')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add percentage labels on top of each bar
    for i, v in enumerate(own_actions):
        ax.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
    for i, v in enumerate(other_actions):
        ax.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
    
    plt.savefig(f'policy_comparison_{title.replace(" ", "_")}.png')
    plt.close()

# Configuration
training_configs = [
    {
        "map_nr": MAP_NR,
        "attitude": ATTITUDE_1.lower(),
        "date": DATETIME,  # Replace with actual training date
        "epoch": EPOCH,  # Replace with desired epoch
        "game_version": "COMPETITION",
        "intent_version": INTENT_VERSION,  # Set if you used an intent version
        "label": ATTITUDE_1  # Label for the plot
    },
    {
        "map_nr": MAP_NR,
        "attitude": ATTITUDE_2.lower(),
        "date": DATETIME,  # Replace with actual training date
        "epoch": EPOCH,  # Replace with desired epoch
        "game_version": "COMPETITION",
        "intent_version": INTENT_VERSION,  # Set if you used an intent version
        "label": ATTITUDE_2  # Label for the plot
    }
]

observations = generate_key_observations()

# Analyze each policy
for config in training_configs:
    policy = load_policy(
        map_nr=config["map_nr"],
        attitude=config["attitude"],
        date=config["date"],
        epoch=config["epoch"],
        game_version=config["game_version"],
        intent_version=config["intent_version"]
    )
    results = analyze_policy(policy, observations)
    
    visualize_policy_comparison(
        results, 
        f"{config['label']} Policy (epoch {config['epoch']})"
    )
    
    print(f"\nAnalysis for {config['attitude']} policy:")
    for item_type, actions in results.items():
        interact_count = sum(1 for a in actions if a['action'] in [3, 4])
        total_count = len(actions)
        if total_count > 0:
            print(f"{item_type}: {interact_count}/{total_count} interactions " +
                  f"({interact_count/total_count*100:.1f}%)")

# Compare policies if we have exactly 2
if len(training_configs) == 2:
    policy1 = load_policy(
        map_nr=training_configs[0]["map_nr"],
        attitude=training_configs[0]["attitude"],
        date=training_configs[0]["date"],
        epoch=training_configs[0]["epoch"],
        game_version=training_configs[0]["game_version"],
        intent_version=training_configs[0]["intent_version"]
    )
    policy2 = load_policy(
        map_nr=training_configs[1]["map_nr"],
        attitude=training_configs[1]["attitude"],
        date=training_configs[1]["date"],
        epoch=training_configs[1]["epoch"],
        game_version=training_configs[1]["game_version"],
        intent_version=training_configs[1]["intent_version"]
    )
    
    agreement_rate, differences = compare_policies(policy1, policy2, observations)
    print(f"\nPolicies agree on {agreement_rate*100:.1f}% of key decisions")
    print("\nKey differences:")
    for diff in differences:
        print(f"When holding {diff['item']} at {diff['position']}:")
        print(f"  {training_configs[0]['attitude']} chose: {diff['action1']}")
        print(f"  {training_configs[1]['attitude']} chose: {diff['action2']}\n")