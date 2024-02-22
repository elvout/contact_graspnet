# Contact-GraspNet ROS Integration

This repository contains Python modules and package specifications to use
*Contact-GraspNet*[1] with ROS 2. Additional documentation about how this
repository is organized can be found in [`ARCHITECTURE.md`](/ARCHITECTURE.md).

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

## Getting Started

### Requirements

- ROS 2 Humble
- Python 3.10
  - Included with Ubuntu 22.04
  - For other supported Ubuntu LTS distributions, the [Deadsnakes Ubuntu
    PPA][deadsnakes] provides the required `python3.10 python3.10-distutils
    python3.10-venv` packages.
- The [Poetry][poetry-docs] dependency manager
- CUDA 11.8.x + cuDNN runtime environment and developer tools (nvcc, etc.).
  - Tested with CUDA 11.8 and cuDNN 8 using the
    `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` Docker image.

[deadsnakes]: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
[poetry-docs]: https://python-poetry.org/docs/

### Building

This repository uses Poetry to install Python dependencies to a virtual
environment. The virtual environment is located at the top level of this
repository after running the build process.

This repository is set up to build with `colcon` using the `ament_cmake`
backend. The build process initializes the virtual environment, downloads model
checkpoints, and builds miscellaneous tensorflow operators used by Contact
GraspNet.

This package must be located inside a ROS 2 workspace. To build, run `colcon
build --symlink-install` from the top level of the ROS 2 workspace.

### Running ROS Nodes

All ROS nodes are Python modules that must run within the virtual environment.
There are multiple ways to run ROS nodes depending on context.

#### Launch Files

Use the following commands at the top level of the workspace.

```shell
source install/setup.$(basename $SHELL)

# Replace LAUNCH_FILE_NAME with one of the files in launch/
ros2 launch contact_graspnet LAUNCH_FILE_NAME
```

#### Direct Invocation

Running nodes directly with Python may be useful for development and testing.
Use the following commands from the top level of this repository.

```shell
# Activate the virtual environment
poetry shell

# Run the node as a Python module, e.g.,
python3 -m rosnode.inference_node
```
