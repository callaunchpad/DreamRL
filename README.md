# DreamRL
Building compact, generative models of environments for efficient reinforcement learning.

See our final presentation [here](https://docs.google.com/presentation/d/e/2PACX-1vT1Nu-tarmVfCaj7zU9X-aMgAO-I0phGKJLQ-molPYxjEv9RWq9sXzf0g76c9waMh4KwwGloflT7aja/pub?start=false&loop=false&delayms=3000).

# Usage

Using conda or venv is probably helpful here. Python version must be 3.7. 
The base requirements.txt seems to not work as of 2020-11-03, probably because some dependencies broke down. 
To remedy this, try using this conda [environment.yml](https://gist.github.com/andrewchenk/e2fe59d3876652835c2030f3db90869a). 

To install with the environment.yml: 
```
conda env create -f environment.yml
```
human_play is a demo for human players to see how well they can perform in openai gym environments. 
For human_play.py:

```
usage: human_play.py [-h] [--plot-rewards] [--zoom-level ZOOM_LEVEL]
                     [--timeout TIMEOUT]
                     env_name
```

See human_play.py for more details. 

Unfortunately, vis.py will not work straight away because the weight files (>1 MB) from training are missing from model_weights. If someone from the DreamRL team can help out with this, that would be great! 

## Contributors
Jonathan Lin (project lead), Joey Hejna, Chelsea Chen, Andrew Chen, Michael Huang, Sumer Kohli, Anish Nag, Jihan Yin
