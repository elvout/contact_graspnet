cmake_minimum_required(VERSION 3.16)
project(contact_graspnet)

# Find CMake dependencies
find_package(ament_cmake REQUIRED)

# Build Python virtual environment
add_custom_target(
    poetry_virtualenv ALL
    COMMAND poetry install --no-root --no-ansi --no-interaction
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building ${CMAKE_PROJECT_NAME} virtualenv... may take some time"
)

# Compile custom tensorflow operators
set(
    TENSORFLOW_OPERATORS
    ${CMAKE_SOURCE_DIR}/pointnet2/tf_ops/grouping/tf_grouping_so.so
    ${CMAKE_SOURCE_DIR}/pointnet2/tf_ops/3d_interpolation/tf_interpolate_so.so
    ${CMAKE_SOURCE_DIR}/pointnet2/tf_ops/sampling/tf_sampling_so.so
)
add_custom_command(
    OUTPUT ${TENSORFLOW_OPERATORS}
    COMMAND bash -c ". .venv/bin/activate && bash compile_pointnet_tfops.sh"
    DEPENDS poetry_virtualenv compile_pointnet_tfops.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Compiling pointnet tensorflow operators"
    VERBATIM
)
add_custom_target(
    pointnet_tfops ALL
    DEPENDS ${TENSORFLOW_OPERATORS}
)

# Download and extract model checkpoints
set(
    CHECKPOINT_ARCHIVE
    ${CMAKE_SOURCE_DIR}/checkpoints_minimal_inference.tar.gz
)
set(
    CHECKPOINT_FILES
    ${CMAKE_SOURCE_DIR}/checkpoints/scene_test_2048_bs3_hor_sigma_001/model.ckpt-54054.index
    ${CMAKE_SOURCE_DIR}/checkpoints/scene_test_2048_bs3_hor_sigma_001/checkpoint
    ${CMAKE_SOURCE_DIR}/checkpoints/scene_test_2048_bs3_hor_sigma_001/model.ckpt-54054.data-00000-of-00001
)
file(
    DOWNLOAD
    https://drive.google.com/uc?export=download&id=1OUaDmLD2OkNc77ca1wjJ_C5lwP5vMdXZ
    ${CHECKPOINT_ARCHIVE}
    EXPECTED_HASH SHA256=e030e0653db41562a56fed2fc27469522db3f5a5f72dbcb0f13a64716bf41c43
)
add_custom_command(
    OUTPUT ${CHECKPOINT_FILES}
    COMMAND tar xf checkpoints_minimal_inference.tar.gz
    DEPENDS ${CHECKPOINT_ARCHIVE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Extracting contact_graspnet checkpoints"
    VERBATIM
)
add_custom_target(
    checkpoints ALL
    DEPENDS ${CHECKPOINT_FILES}
)

# Install launch files.
# https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-system.html
install(
    DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
)

# ROS 1 provided a mechanism in XML launch files to find the corresponding
# source directory of a package. ROS 2 removed this mechanism but instead allows
# users to reference the "install" path of a package within a workspace. Thus,
# any files and folders required to run nodes must be discoverable inside the
# workspace's install directory. We manually symlink several directories for two
# reasons:
# 1) "colcon build" would otherwise copy gigabytes worth of libraries
# 2) "colcon build --symlink-install" creates a symlink for each file in the
#    directory, rather than a single symlink for the directory itself.
#
# ROS 2 looks for executables in ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}.
file(
    # "file" commands are executed at build time. The install directory won't
    # exist yet during a fresh build, which will cause the next CREATE_LINK to
    # fail.
    MAKE_DIRECTORY
    ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}
)
file(
    CREATE_LINK
    ${CMAKE_SOURCE_DIR}/.venv
    ${CMAKE_INSTALL_PREFIX}/.venv
    SYMBOLIC
)
file(
    CREATE_LINK
    ${CMAKE_SOURCE_DIR}/checkpoints
    ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/checkpoints
    SYMBOLIC
)
file(
    CREATE_LINK
    ${CMAKE_SOURCE_DIR}/contact_graspnet
    ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/contact_graspnet
    SYMBOLIC
)
file(
    CREATE_LINK
    ${CMAKE_SOURCE_DIR}/pointnet2
    ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/pointnet2
    SYMBOLIC
)
file(
    CREATE_LINK
    ${CMAKE_SOURCE_DIR}/rosnode
    ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/rosnode
    SYMBOLIC
)

install(
    PROGRAMS
    scripts/python_entrypoint.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
