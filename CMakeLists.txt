cmake_minimum_required(VERSION 2.8.9)

project( FFT )

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

INCLUDE(OpenCL)

set(SRC_FILES main.cpp applyfft.cpp)

set(ENTRY_FILE main.cpp)

set(HDR_FILE applyfft.h)

ADD_EXECUTABLE(example.x ${SRC_FILES} ${ENTRY_FILE})

TARGET_LINK_LIBRARIES(example.x  oclSetup clFFT ${OPENCL_LIBRARIES} ${CMAKE_DL_LIBS})
