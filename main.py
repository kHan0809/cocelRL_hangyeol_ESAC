import argparse
import torch
import gym
import sys
import numpy as np
from Algorithm.SAC import SAC_fix_alpha
from Common.Utils import set_seed, gym_env, Eval

sys.path.append('C:/Users/owner/.mujoco/mujoco200/bin')

def hyperparameters():
    parser = argparse.ArgumentParser(description='Efficient Soft Actor Critic (SAC) v2 example')
    #environment
    parser.add_argument('--env-name', default='HalfCheetah-v2', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--policy-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=5000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=5, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=10000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='offline, online')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--train-alpha-flag', default=False, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--tau', default=0.05, type=float)
    parser.add_argument('--critic-update', default=2, type=int)
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')
    #image
    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=False, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=False, type=bool, help='when logged, write log')

    parser.add_argument('--model', default=False, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')
    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=10000, type=int, help='buffer saving frequency')

    args = parser.parse_args()

    return args

def main(args,iter):
    if args.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.log == True:
        f = open("./log" + str(iter) + "SAC_Multi" + ".txt", 'w')
        f.close()

    print("Device: ", device)
    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Random Seed:", random_seed)

    env, eval_env = gym_env(args.env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    episode_length = env._max_episode_steps

    reward_buff = []
    algorithm = SAC_fix_alpha(state_dim,action_dim,device,args)

    state = env.reset()
    epi_reward = 0
    epi_timesteps = 0
    epi_num = 0

    for global_step in range(int(args.max_step)):
        if global_step < args.policy_start:
            action = env.action_space.sample() / max_action
        else:
            # algorithm.train(args.training_step)  # 임시로
            action = algorithm.get_action(state)
            action = action + 0.1 * np.random.normal(0, 1.0, [action_dim])
            action = np.clip(action, -1., 1.)
        next_state, reward, terminal, _ = env.step(action*max_action)
        epi_timesteps += 1

        done = float(terminal) if epi_timesteps < episode_length else 0

        algorithm.buffer.store(state, action, reward, next_state, done)

        state = next_state
        epi_reward += reward

        if global_step >= args.training_start:
            algorithm.train(args.training_step)

        if terminal:
            state = env.reset()
            epi_reward = 0
            epi_timesteps = 0
            epi_num += 1

        if ((global_step + 1) % args.eval_step==0)&(args.eval == True):
            min_rwd, avg_rwd, max_rwd = Eval(eval_env, algorithm, args.eval_episode,args.render)
            print(
                f"[#EPISODE {epi_num} | #GLOBALSTEP {global_step + 1}] MIN : {min_rwd:.2f}, AVE : {avg_rwd:.2f}, MAX : {max_rwd:.2f}")

            # Logging (e.g., csv file / npy file/ txt file)
            global_step, min_rwd, avg_rwd, max_rwd = str(global_step + 1), str(min_rwd), str(avg_rwd), str(max_rwd)

            f = open("./log" + str(iter) + "SAC_Multi" + ".txt", 'a')
            f.write(" ".join([global_step, min_rwd, avg_rwd, max_rwd]))
            f.write("\n")
            f.close()

if __name__ == '__main__':
    for iter in range(1,6):
        args = hyperparameters()
        main(args,iter)