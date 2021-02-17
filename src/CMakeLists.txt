# External Dependencies
project(convolution-hostcode)

add_executable(convolution
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/conv3D.cpp
)
target_link_libraries(convolution PRIVATE cxxopts)

target_include_directories(convolution 
  PRIVATE ${PROJECT_SOURCE_DIR})