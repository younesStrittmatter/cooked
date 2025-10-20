# ---- Reward analysis module ---- #
def get_rewards(self, agent_events, agent_penalties, REWARDS):
    """
    Calculate pure and modified rewards based on the game mode.
    """
    if self.game_mode == "classic":
        return get_rewards_classic(self, agent_events, agent_penalties, REWARDS)
    elif self.game_mode == "competition":
        return get_rewards_competition(self, agent_events, agent_penalties, REWARDS)
    else:
        raise ValueError(f"Unknown game mode: {self.game_mode}")


# ---- Classic mode without ownership awareness ---- #
def get_rewards_classic(self, agent_events, agent_penalties, REWARDS):
    """
    Calculate pure and modified rewards for classic mode.
    """
    # Get reward from delivering items
    rewards_cfg = REWARDS
    event_rewards = {agent_id: 0.0 for agent_id in self.agents}
    deliver_rewards = {agent_id: 0.0 for agent_id in self.agents}
    for agent_id in self.agents:
        # Event rewards: only positive events
        event_rewards[agent_id] = (
            agent_events[agent_id]["cut"] * rewards_cfg["item_cut"]
            + agent_events[agent_id]["salad"] * rewards_cfg["salad_created"]
        )
        deliver_rewards[agent_id] = agent_events[agent_id]["delivered"] * rewards_cfg["delivered"]

    shared_deliver_reward = sum(deliver_rewards.values())

    for agent_id in self.agents:
        reward = shared_deliver_reward + event_rewards[agent_id]
        self.cumulated_pure_rewards[agent_id] += reward

        alpha, beta = self.reward_weights.get(agent_id, (1.0, 0.0))
        other_agents = [a for a in self.agents if a != agent_id]
        avg_other_reward = (
            sum(deliver_rewards[a] + event_rewards[a] for a in other_agents) / len(other_agents)
            if other_agents else 0.0
        )  # in case there is only one agent

        # Modified rewards: include penalties
        self.rewards[agent_id] = alpha * (reward - agent_penalties[agent_id]) + beta * avg_other_reward
        self.cumulated_modified_rewards[agent_id] += self.rewards[agent_id]

    return self.cumulated_pure_rewards, self.cumulated_modified_rewards

# ---- Competition mode with ownership awareness ---- #
def get_rewards_competition(self, agent_events, agent_penalties, REWARDS):
    # Get reward from delivering items
    rewards_cfg = REWARDS
    pure_rewards = {agent_id: 0.0 for agent_id in self.agents}
    for agent_id in self.agents:
        # Pure rewards: only positive events, no IDLE or useless penalties
        own_reward = (
            agent_events[agent_id]["delivered_own"] * rewards_cfg["delivered"]
            + agent_events[agent_id]["salad_own"] * rewards_cfg["salad_created"]
            + agent_events[agent_id]["cut_own"] * rewards_cfg["item_cut"]
        )

        other_reward = (
            agent_events[agent_id]["delivered_other"] * rewards_cfg["delivered"]
            + agent_events[agent_id]["salad_other"] * rewards_cfg["salad_created"]
            + agent_events[agent_id]["cut_other"] * rewards_cfg["item_cut"]
        )

        other_penalty = 0
        for other_agent_id in self.agents:
            if other_agent_id == agent_id:
                continue
            other_penalty += (
                agent_events[other_agent_id]["delivered_other"] * rewards_cfg["delivered"]
                + agent_events[other_agent_id]["salad_other"] * rewards_cfg["salad_created"]
                + agent_events[other_agent_id]["cut_other"] * rewards_cfg["item_cut"]
            )

        pure_rewards[agent_id] = (self.payoff_matrix[0] * own_reward + self.payoff_matrix[1] * other_reward + self.payoff_matrix[2] * other_penalty)
        self.cumulated_pure_rewards[agent_id] += pure_rewards[agent_id]

    for agent_id in self.agents:
        alpha, beta = self.reward_weights.get(agent_id, (1.0, 0.0))
        other_agents = [a for a in self.agents if a != agent_id]
        if other_agents:
            avg_other_reward = sum(pure_rewards[a] for a in other_agents) / len(other_agents)
        else:
            avg_other_reward = 0.0  # in case there is only one agent
        # Modified rewards: include penalties
        self.rewards[agent_id] = alpha * (pure_rewards[agent_id] - agent_penalties[agent_id]) + beta * avg_other_reward
        self.cumulated_modified_rewards[agent_id] += self.rewards[agent_id]

    return self.cumulated_pure_rewards, self.cumulated_modified_rewards