project(convolution-gtest)

add_executable(test_convolution test_conv3D.cpp)
target_link_libraries(test_convolution PRIVATE gtest_main gtest gmock)
target_include_directories(test_convolution PRIVATE ${CMAKE_SOURCE_DIR}/src)