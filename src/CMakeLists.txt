# External Dependencies
cmake_minimum_required(VERSION 3.10)
project(convolution-hostcode)

set(FFTW_PLAN "FFTW_ESTIMATE" CACHE STRING "FFTW Plan")
set_property(CACHE FFTW_PLAN PROPERTY STRINGS "FFTW_ESTIMATE" "FFTW_MEASURE" "FFTW_PATIENT" "FFTW_EXHAUSTIVE")

add_executable(convolution
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/fft_conv3D.cpp
  ${PROJECT_SOURCE_DIR}/helper.cpp
)

target_compile_options(convolution PRIVATE -Wall -Werror)
target_compile_features(convolution PRIVATE cxx_std_17)

target_include_directories(convolution 
  PRIVATE ${PROJECT_SOURCE_DIR} 
          ${IntelFPGAOpenCL_INCLUDE_DIRS}
          ${CMAKE_BINARY_DIR}/include 
          ${FFTW_INCLUDE_DIRS})

target_link_libraries(convolution 
  PRIVATE convfpga cxxopts MPI::MPI_CXX OpenMP::OpenMP_CXX  
    ${IntelFPGAOpenCL_LIBRARIES})

if(FFTW_FOUND)
  target_compile_definitions(convolution PRIVATE USE_FFTW)
  target_link_libraries(convolution PUBLIC fftw3 fftw3f ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB} ${FFTW_FLOAT_MPI_LIB})
else()
  message(WARNING, "FFTW library not found. Cannot perform verification")
endif()