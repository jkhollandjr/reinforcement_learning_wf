import torch
import torch.nn as nn
import torch.optim as optim
from dqn_environment import TorTraffic
from collections import namedtuple
from collections import deque
import argparse
import numpy as np

GAMMA = .99
BATCH_SIZE = 64
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 600000
EPSILON_START = 1.0
EPSILON_FINAL = .02

HIDDEN_SIZE = 128

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class DQN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),
                nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        if(np.random.randint(0,100) == 10):
            with open("dqn_log.txt", "a") as f:
                action_list = list(actions)
                f.write(' '.join(str(action_list)) + "\n")
                    

        return np.array(states, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states, dtype=np.float32)

class Agent:
    def __init__(self, exp_buffer):
        self.env = TorTraffic()
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = np.random.randint(0,self.env.action_size)
        else:
            state_a = np.array([self.state], copy=False).astype(np.float32)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            #if(np.random.randint(0,100)==0):
            #    print(q_vals_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        #step through environment
        new_state, is_done, website, trace_num, self.current_trace, reward = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":
    obs_size = 20
    n_actions = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    net = DQN(obs_size, HIDDEN_SIZE, n_actions).to(device)
    tgt_net = DQN(obs_size, HIDDEN_SIZE, n_actions).to(device)

    print(device)
    print(net)

    exp_buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(exp_buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    step_idx = 0
    best_mean_reward = None


    while True:
        step_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - step_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            #going to need to only end episode at the right times
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("{}: done {} games, mean reward {}, eps {}, attack acc {}".format(step_idx, len(total_rewards), mean_reward, epsilon, str(sum(agent.env.accuracy_list[-100:]))))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), "dqn_wf_best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated {} -> {}, model saved".format(best_mean_reward, mean_reward))

                best_mean_reward = mean_reward
        
        if len(exp_buffer) < REPLAY_START_SIZE:
            continue

        if step_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        if step_idx % 20000 == 0:
            agent.env.retrain_attack()

        if(step_idx % 5 == 0):
            optimizer.zero_grad()
            batch = exp_buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()






