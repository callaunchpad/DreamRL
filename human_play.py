import gym
from gym.utils import play
import argparse

parser = argparse.ArgumentParser(description='Have a human play a specific gym.')
parser.add_argument('env_name', type=str, help='Environment name (e.g., Pong-v4)')
parser.add_argument('--plot-rewards', default=False, help='Plot rewards while playing (warning: can cause keyboard input glitches)', action='store_true')
parser.add_argument('--zoom-level', default=3, type=float, help='Set custom zoom level')

def main():
	def callback(obs_t, obs_tp1, action, rew, done, info):
	    return [rew,]

	plotter = play.PlayPlot(callback, 30 * 5, ["reward"])
	env = gym.make(args.env_name)
	if not args.plot_rewards:
		callback = None
	else:
		callback = plotter.callback
	play.play(env, zoom=args.zoom_level, callback=callback)

if __name__ == '__main__':
	args = parser.parse_args()
	main()
