import numpy as np
import gym
import random
import torch
import torch.nn as nn


def Eval(eval_env, agent, eval_num,render):
    reward_history = []
    max_action = float(eval_env.action_space.high[0])
    for eval_iter in range(eval_num):
        state = eval_env.reset()
        reward_sum = 0
        for eval_step in range(eval_env._max_episode_steps):
            if (eval_iter == eval_num-1)&(render):
                eval_env.render()
            action = agent.eval_action(state)
            next_state, reward, terminal, _ = eval_env.step(action*max_action)
            reward_sum += reward
            state = next_state
            if terminal:
                break

        reward_history.append(reward_sum)
    return min(reward_history), sum(reward_history)/len(reward_history), max(reward_history)


def soft_update(network, target_network, TAU):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def copy_weight(network, target_network):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    random.seed(random_seed)

    return random_seed


def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env


