# Contact-GraspNet ROS Integration

This repository contains a [Poetry][poetry-home] virtual environment to use *Contact-GraspNet*[1] with ROS 1 Noetic.

This is a fork of the original project with the `main` branch tracking the
upstream repository. Switch to the `main` branch to view the original README.md.

[1]

```bibtex
@article{sundermeyer2021contact,
  title={Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

- [paper](https://arxiv.org/abs/2103.14127)
- [project page](https://research.nvidia.com/publication/2021-03_Contact-GraspNet%3A--Efficient)
- [video](http://www.youtube.com/watch?v=qRLKYSLXElM)

[poetry-home]: https://python-poetry.org/

## Getting Started

### CUDA + cuDNN Runtime and Developer Tools

This repository requires CUDA and cuDNN developer tools to build.

This branch is tested with CUDA 11.8 and cuDNN 8 using the
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` Docker image.

### Install Python 3.9

The [Deadsnakes Ubuntu PPA][deadsnakes] provides alternative Python versions
that are not present in the official package repositories. These commands will
set up the PPA and install the required Python 3.9 packages:

```shell
apt update && apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.9 python3.9-distutils python3.9-venv
```

[deadsnakes]: <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>

### Build

This repository is set up to build using `catkin` and/or `cmake`. The build
process installs the Poetry dependency manager if it is not already present on
the system, installs Python dependencies to `.venv`, downloads model
checkpoints, and builds miscellaneous tensorflow operators.

If this package is within a `catkin` workspace, run `catkin build` within the
workspace. Otherwise, you can run `cmake` directly from the top-level directory
of this repository:

```shell
cmake -B build && cmake --build build
```

## Running ROS Nodes

There are multiple ways to run ROS nodes depending on context.

### Launch Files

If you are in a catkin workspace, source `devel/setup.$(basename $SHELL)` in the
top level of the workspace and then run `roslaunch contact_graspnet $LAUNCH_FILE_NAME`, where `$LAUNCH_FILE_NAME` is the name of one of the files in [`launch/`](/launch/).

If you are in the top level of this repository, you can also run
`roslaunch launch/$LAUNCH_FILE_NAME` directly without sourcing a setup file.

### Direct Invocation

Using the virtual environment directly is sometimes easier for prototyping and
testing. From the top level of this repository,

```shell
# Activate the virtual environment
poetry shell

# Run the node as a Python module, e.g.,
python3 -m rosnode.inference_node
```
