# Seting variable once to use it everywhere
set(NAME UnsharpMask)
# Setting executable
add_executable(${NAME} ${NAME}.cpp)
# Setting includes
include_directories(
    ${TBB_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
)
# Linking
target_link_libraries(${NAME}
    ${TBB_LIBRARIES}
    ${OpenCV_LIBS}
)
# Copy executable to build directory
add_custom_command(TARGET ${NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${NAME} ${PROJECT_BINARY_DIR})