project(convolution-gtest)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

add_executable(test_convolution test_conv3D.cpp)
target_link_libraries(test_convolution PRIVATE gtest_main gtest gmock)
target_include_directories(test_convolution PRIVATE ${CMAKE_SOURCE_DIR}/src)