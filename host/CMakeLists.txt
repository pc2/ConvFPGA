# Arjun Ramaswami
cmake_minimum_required(VERSION 3.16)
project(convfpga VERSION 1.0
            DESCRIPTION "APIs for FFT based convolution using FPGAs"
            LANGUAGES C)

add_library(convfpga STATIC 
              ${PROJECT_SOURCE_DIR}/src/fpga_init.c 
              ${PROJECT_SOURCE_DIR}/src/convfpga.c 
              ${PROJECT_SOURCE_DIR}/src/convfpga_svm.c 
              ${PROJECT_SOURCE_DIR}/src/svm.c
              ${PROJECT_SOURCE_DIR}/src/opencl_utils.c
              ${PROJECT_SOURCE_DIR}/src/misc.c)

#target_compile_features(convfpga PRIVATE cxx_std_17)
target_compile_options(convfpga
    PRIVATE -Wall -Werror)
    
if(USE_DEBUG)
  target_compile_definitions(convfpga PRIVATE DEBUG)
endif()

target_include_directories(convfpga
    PRIVATE src 
    PUBLIC ${IntelFPGAOpenCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
  
target_link_libraries(convfpga
    PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m)
