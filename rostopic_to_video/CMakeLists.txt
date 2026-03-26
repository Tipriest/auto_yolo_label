cmake_minimum_required(VERSION 3.0.2)
project(rostopic_to_video)

find_package(catkin REQUIRED COMPONENTS
	rospy
	sensor_msgs
	cv_bridge
	message_filters
	image_geometry
	tf
	tf2_ros
)

catkin_package()

include_directories(
	${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
	scripts/rostopic_to_video_node.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
	FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY config/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
	FILES_MATCHING PATTERN "*.yaml"
)
