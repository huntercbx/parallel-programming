cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory("01 - Parallel For")
add_subdirectory("02 - Blocked Range 2D")
add_subdirectory("04 - Parallel Reduce")
add_subdirectory("05 - Parallel Do")
add_subdirectory("06 - Parallel Pipeline")
add_subdirectory("07 - Flow Graph")
add_subdirectory("08 - Unsharp Mask")
