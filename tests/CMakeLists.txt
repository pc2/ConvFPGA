project(convolution-gtest)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)

add_executable(test_convolution test_main.cpp)
target_link_libraries(test_convolution PUBLIC gtest_main gtest gmock)