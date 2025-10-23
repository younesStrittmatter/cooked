# Observation Space and Action Space Implementation

This document explains how the observation space and action space are implemented for the RL agents in the `spoiled_broth` environment. The relevant code can be found in:
- `spoiled_broth/rl/observation_space.py`
- `spoiled_broth/rl/action_space.py`

## Observation Space

The observation space encodes the current state of the environment as a vector for each agent. It is designed to provide all necessary information for decision-making, while being independent of the map layout.

### Key Features
- **Distances to Key Tiles:** For each important tile type (e.g., dispensers, cutting boards, delivery, counters), the observation includes:
  - Distance from the agent to the closest tile of that type
  - Distance from the midpoint between both agents to the closest tile of that type
- **Items on Counters:** For each item type (e.g., tomato, plate, tomato_cut, tomato_salad, etc.), the observation includes:
  - Distance from the agent to the closest counter with that item
  - Distance from the midpoint to the closest counter with that item
- **Distance to Other Agent:** The path distance between the agent and the other agent
- **Agent Inventories:** One-hot vectors representing the items held by the agent and the other agent
- **Normalization:** All distance values can be normalized by the maximum possible distance for the map, making the observation space consistent across different layouts.

### Modes
- **Classic Mode:** Supports basic items (tomato, plate, etc.) and tile types.
- **Competition Mode:** Adds support for additional items (e.g., pumpkin) and tile types.

### Main Functions
- `game_to_obs_vector_classic` and `game_to_obs_vector_competition`: Build the observation vector for each mode.
- `normalize_obs_vector`: Normalizes distance values.
- `game_to_obs_vector`: Wrapper to select the correct mode and normalization.

## Action Space

The action space defines the set of high-level, human-like actions available to the RL agent. These actions are independent of the map layout and are mapped to specific tile interactions at runtime.

### Possible Actions and Their Meanings

#### Classic Mode Actions

| Action Name                                 | Meaning                                                                 |
|---------------------------------------------|-------------------------------------------------------------------------|
| pick_up_tomato_from_dispenser               | Pick up a tomato from the tomato dispenser                              |
| pick_up_plate_from_dispenser                | Pick up a plate from the plate dispenser                                |
| use_cutting_board                           | Use a cutting board (e.g., to chop an ingredient)                       |
| use_delivery                                | Deliver a completed dish                                                |
| put_down_item_on_free_counter_closest       | Put down the held item on the closest free counter                      |
| put_down_item_on_free_counter_midpoint      | Put down the held item on the counter closest to the agents' midpoint   |
| pick_up_tomato_from_counter_closest         | Pick up a tomato from the closest counter                               |
| pick_up_tomato_from_counter_midpoint        | Pick up a tomato from the counter closest to the agents' midpoint       |
| pick_up_plate_from_counter_closest          | Pick up a plate from the closest counter                                |
| pick_up_plate_from_counter_midpoint         | Pick up a plate from the counter closest to the agents' midpoint        |
| pick_up_tomato_cut_from_counter_closest     | Pick up a chopped tomato from the closest counter                       |
| pick_up_tomato_cut_from_counter_midpoint    | Pick up a chopped tomato from the counter closest to the agents' midpoint |
| pick_up_tomato_salad_from_counter_closest   | Pick up a tomato salad from the closest counter                         |
| pick_up_tomato_salad_from_counter_midpoint  | Pick up a tomato salad from the counter closest to the agents' midpoint |

#### Competition Mode Actions

All classic actions, plus:

| Action Name                                         | Meaning                                                                 |
|-----------------------------------------------------|-------------------------------------------------------------------------|
| pick_up_pumpkin_from_dispenser                      | Pick up a pumpkin from the pumpkin dispenser                            |
| pick_up_pumpkin_from_counter_closest                | Pick up a pumpkin from the closest counter                              |
| pick_up_pumpkin_from_counter_midpoint               | Pick up a pumpkin from the counter closest to the agents' midpoint      |
| pick_up_pumpkin_cut_from_counter_closest            | Pick up a chopped pumpkin from the closest counter                      |
| pick_up_pumpkin_cut_from_counter_midpoint           | Pick up a chopped pumpkin from the counter closest to the agents' midpoint |
| pick_up_pumpkin_salad_from_counter_closest          | Pick up a pumpkin salad from the closest counter                        |
| pick_up_pumpkin_salad_from_counter_midpoint         | Pick up a pumpkin salad from the counter closest to the agents' midpoint |

**Note:** Each action with a `_closest` or `_midpoint` suffix targets either the nearest relevant tile or the one closest to the midpoint between both agents, respectively.

### Key Features
- **High-Level Actions:** Actions such as picking up items from dispensers or counters, using cutting boards or delivery, and putting down items on free counters.
- **Variants:** Many actions have both 'closest' and 'midpoint' variants, allowing the agent to target either the nearest relevant tile or the one closest to the midpoint between agents.
- **Map Independence:** The action names do not depend on specific tile locations; instead, helper functions map them to the correct tile index based on the current state and distance map.

### Modes
- **Classic Mode:** Includes actions for basic items and interactions.
- **Competition Mode:** Expands the action set to include additional items and interactions.

### Main Functions
- `get_rl_action_space`: Returns the list of available actions for the selected mode.
- `convert_action_to_tile`: Maps a high-level action to a specific tile index to click, using the agent's position, the other agent's position, and the distance map.
- `find_closest_tile` and `find_midpoint_tile`: Helpers to select the best tile candidate for an action.

## Summary
- The observation space provides a normalized, map-independent vector encoding all relevant distances and inventories.
- The action space offers a set of high-level actions, mapped to specific tile interactions at runtime, supporting both classic and competition modes.

For more details, see the code and docstrings in `spoiled_broth/rl/observation_space.py` and `spoiled_broth/rl/action_space.py`.
