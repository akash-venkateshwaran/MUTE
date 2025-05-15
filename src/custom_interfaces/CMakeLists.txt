cmake_minimum_required(VERSION 3.8)
project(custom_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/HelperPosition.msg"
  "msg/HeadingAngle.msg"
  "msg/State.msg"
  "msg/Path.msg"
  "msg/MultiState.msg"
  "msg/ShipInfo.msg"
  "msg/SPL.msg"
  "msg/Speed.msg"
)

set(srv_files
  "srv/BathyService.srv"
  "srv/BoundsService.srv"
  "srv/StateService.srv"
  "srv/PolygonService.srv"
)

# Add the custom message and service files
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
)

ament_package()
