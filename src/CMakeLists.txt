# External Dependencies
include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts
  GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)

add_executable(convolution
  ${PROJECT_SOURCE_DIR}/src/main.cpp
)
target_link_libraries(convolution PRIVATE cxxopts)

target_include_directories(convolution 
  PRIVATE 
    ${PROJECT_SOURCE_DIR}/include 
    ${CMAKE_BINARY_DIR}/include) 