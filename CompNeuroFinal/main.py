from spikeWaveEprop import *
from weightvisual import draw_weights
from env import Agent, map_size_x, map_size_y, environment, get_loss
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--init_weights',         type=int, default=1,     help="flag to initialize weights")
parser.add_argument('--train',         type=int, default=1,     help="flag to train model")
parser.add_argument('--timesteps',         type=int, default=250,     help="number of training steps")
parser.add_argument('--trials',         type=int, default=25,     help="number of repeated tests")
args = parser.parse_args()

def spikewave_trial(replay, replay_amt, mem_size, mem_type):
    # cost map (10 x 10) matrix
    #   The outer border has a higher cost to prevent paths from going out of bounds
    #   An inner square is given a moderate cost to simulate an obstacle
    cost_map = np.ones((map_size_y, map_size_x))
    n1 = cost_map.shape[0]
    n2 = cost_map.shape[1]
    # Weights can either be initialized to some value or loaded from a file
    if args.init_weights:
        wgt = np.zeros([n1, n2, n1, n2])
        w_init = 2
        for i in range(n1):
            for j in range(n2):
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        # 8 directions for these mazes
                        if (i + m >= 0) and (i + m < n1) and (j + n >= 0) and (j + n < n2) and (m != 0 or n != 0):
                            wgt[i][j][i + m][j + n] = w_init
    else:
        wgt = np.load("wgt.npy")
    # Run the spike wave navigation for N trials. Use a Levy Flight distribution to choose a waypoint.
    # The waypoint must be at least 1 grid position away from the current position.
    wp_end = np.array([1, 1])
    wp_start = np.copy(wp_end)
    agent = Agent(wp_end, mem_size, map_size_y, map_size_x)
    losses = []
    t = 0
    while t < args.timesteps:

        lf = levy_flight(2, 3)  # levy_flight will return a new waypoint
        # waypoint coordinates must be whole numbers
        wp_end[0] += round(lf[0])
        # check that the waypoint is within the map boundaries
        if wp_end[0] < 0:
            wp_end[0] = 0
        elif wp_end[0] >= n1:
            wp_end[0] = n1-1
        wp_end[1] += round(lf[1])
        if wp_end[1] < 0:
            wp_end[1] = 0
        elif wp_end[1] >= n1:
            wp_end[1] = n1-1

        # if the new waypoint is not over 1 grid position away from the current position,
        #     skip this waypoint and find another.
        if get_distance(wp_start, wp_end) > 1:
            t += 1
            et, p = spike_wave(wgt, cost_map, wp_start[0], wp_start[1], wp_end[0], wp_end[1])

            cost_map, p_len = agent.drive(p, cost_map, et)
            p = p[len(p) - p_len:]

            agent.pathmem.push(p)
            agent.etmem.push((et, 1))

            wgt = update_weights(cost_map, et, p, wgt)

            wp_end = agent.current_pos
            wp_start = np.copy(wp_end)

            if (t + 1) % 10 == 0:
                #draw_weights(n1, n2, 1, 1, 0, 15, f"{mem_type}\{t+1}_{mem_type}", wgt)
                agent.moveTo(np.array([1, 1]))
                wp_start = np.array([1, 1])
                if replay and t > replay_amt:
                    wgt = agent.replay(replay_amt, cost_map, wgt, mem_type)
                    #draw_weights(n1, n2, 1, 1, 0, 15, f"{mem_type}\{t+1}_{mem_type}_afterreplay", wgt)

                print(f"{t+1}/{args.timesteps} completed {mem_type}")

            losses.append(get_loss(n1, n2, wgt, environment))
    
    return losses

losses_nomem = []
losses_mem = []
losses_exp = []
losses_mem_recent = []

for i in range(args.trials):
    losses_nomem.append(spikewave_trial(False, 0, 0, "nomemory"))
    losses_mem.append(spikewave_trial(True, 10, 100, "uniform"))
    losses_exp.append(spikewave_trial(True, 10, 100, "exp"))
    losses_mem_recent.append(spikewave_trial(True, 10, 100, "recent"))

losses_nomem = np.array(losses_nomem)
losses_mem = np.array(losses_mem)
losses_exp = np.array(losses_exp)
losses_mem_recent = np.array(losses_mem_recent)

std_nomem = np.std(losses_nomem, axis=0)
std_mem = np.std(losses_mem, axis=0)
std_exp = np.std(losses_exp, axis=0)
std_mem_recent = np.std(losses_nomem, axis=0)

losses_nomem = np.mean(losses_nomem, axis=0)
losses_mem = np.mean(losses_mem, axis=0)
losses_exp = np.mean(losses_exp, axis=0)
losses_mem_recent = np.mean(losses_mem_recent, axis=0)

plt.figure()
plt.plot(list(range(args.timesteps)), losses_nomem, label="No Replay")
plt.fill_between(list(range(args.timesteps)), losses_nomem + std_nomem, losses_nomem - std_nomem, alpha=0.2, linewidth=4, linestyle='dashdot', antialiased=True)

plt.plot(list(range(args.timesteps)), losses_mem, label="Uniform Replay")
plt.fill_between(list(range(args.timesteps)), losses_mem + std_mem, losses_mem - std_mem, alpha=0.2, linewidth=4, linestyle='dashdot', antialiased=True)

plt.plot(list(range(args.timesteps)), losses_exp, label="Experience Replay")
plt.fill_between(list(range(args.timesteps)), losses_exp + std_exp, losses_exp - std_exp, alpha=0.2, linewidth=4, linestyle='dashdot', antialiased=True)

plt.plot(list(range(args.timesteps)), losses_mem_recent, label="Recency Replay")
plt.fill_between(list(range(args.timesteps)), losses_mem_recent + std_mem_recent, losses_mem_recent - std_mem_recent,  alpha=0.2, linewidth=4, linestyle='dashdot', antialiased=True)

plt.title("Loss")
plt.xlabel("Timestep")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

#np.save("wgt.npy", wgt)