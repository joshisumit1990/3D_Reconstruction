cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project( 3D_Reconstruction )

find_package( PCL 1.2 REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
add_executable( 3D_Reconstruction ../src/opencv_reproject_image_pcl.cpp )
target_link_libraries( 3D_Reconstruction ${OpenCV_LIBS} ${PCL_LIBRARIES})

