import gym
from gym.utils import play
import argparse
import pygame
import matplotlib
from threading import Timer
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None
from collections import deque

parser = argparse.ArgumentParser(description='Have a human play a specific gym.')
parser.add_argument('env_name', type=str, help='Environment name (e.g., Pong-v4)')
parser.add_argument('--plot-rewards', default=False, help='Plot rewards while playing (warning: can cause keyboard input glitches)', action='store_true')
parser.add_argument('--zoom-level', default=3.0, type=float, help='Set custom zoom level (default is 3.0)')
parser.add_argument('--timeout', default=10.0, type=float, help='Game timeout (default is 10.0 seconds)')

def main():
	# Define reward callback
	timestep = 0
	def callback(obs_t, obs_tp1, action, rew, done, info):
		nonlocal timestep
		timestep += 1
		data.append(rew)

	# Initialize data structures for plotting
	fig, ax = plt.subplots(1)
	ax.set_title("Reward over time")
	horizon_timesteps = int(30 * args.timeout)
	data = deque(maxlen = horizon_timesteps)
	if not args.plot_rewards:
		callback = None

	# Initialize game timer
	t = Timer(args.timeout, lambda: pygame.quit())
	t.start()

	# Run main game loop
	try:
		env = gym.make(args.env_name)
		play.play(env, fps=30, zoom=args.zoom_level, callback=callback)
	except pygame.error:
		pass

	# Plot rewards over time before quitting
	xmin, xmax = max(0, timestep - horizon_timesteps), timestep
	ax.scatter(range(xmin, xmax), list(data), c='blue')
	ax.set_xlim(xmin, xmax)
	plt.show()

if __name__ == '__main__':
	args = parser.parse_args()
	main()
