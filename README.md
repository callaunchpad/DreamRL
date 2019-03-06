# DreamRL
Building compact, generative models of environments for efficient reinforcement learning.

## Setup
I strongly encourage building dependencies within a virtual environment. To use the Python virtual environments, follow [this guide on python environments](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/), or to use Conda environments, follow [this guide on conda environments](https://conda.io/docs/user-guide/tasks/manage-environments.html).

Once inside your environment, from the base directory of this repo, run

```
pip install -r requirements.txt
```
This should add any required dependencies to your virtual environment.

## Contributing Code and Workflow
**Please make sure all commits have meaningful comments!**

Do not use `git add .` or `git add *`to add files. This adds files that should potentially be untracked. 
Instead, add the individual files you've worked on or use
```
git commit -am "commit message"
```
This adds and commits all modified files that are tracked. If you need to track new files, then use
```
git add new_file_name_here
```

## Contributors
Jonathan Lin (project lead), Jihan Yin, Joey Hejna, Chelsea Chen, Andrew Chen, Michael Huang, Sumer Kohli, Anish Nag