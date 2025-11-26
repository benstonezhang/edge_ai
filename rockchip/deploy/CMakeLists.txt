cmake_minimum_required(VERSION 3.15.1)

project(mtmd)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

if(NOT RKLLM_PATH)
    message(FATAL "path of rknn-llm (https://github.com/airockchip/rknn-llm.git) must be specified, the rkllm can ONLY Work with bundled opencv")
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options("-Wl,-Bsymbolic")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(TARGET_LIB_ARCH aarch64)
    else()
        set(TARGET_LIB_ARCH armhf)
    endif()
endif()

find_package(OpenMP REQUIRED)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${RKLLM_PATH}/examples/multimodal_model_demo/deploy/3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
    set(OpenCV_DIR ${RKLLM_PATH}/examples/multimodal_model_demo/deploy/3rdparty/opencv/opencv-linux-${TARGET_LIB_ARCH}/share/OpenCV)
endif()
find_package(OpenCV REQUIRED)

# rknn runtime
set(RKNN_LIB ${RKLLM_PATH}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/${CMAKE_SYSTEM_NAME}/librknn_api/${TARGET_LIB_ARCH}/librknnrt.so)
include_directories(${RKLLM_PATH}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/${CMAKE_SYSTEM_NAME}/librknn_api/include)

# rkllm runtime
set(RKLLM_LIB ${RKLLM_PATH}/rkllm-runtime/${CMAKE_SYSTEM_NAME}/librkllm_api/${TARGET_LIB_ARCH}/librkllmrt.so)
include_directories(${RKLLM_PATH}/rkllm-runtime/${CMAKE_SYSTEM_NAME}/librkllm_api/include)

add_executable(${PROJECT_NAME} main.cpp)
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${RKNN_LIB} ${RKLLM_LIB} log)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${RKNN_LIB} ${RKLLM_LIB})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS -Wl,--no-undefined)

# Install the executable file to the specified directory
set(CACHE{CMAKE_INSTALL_PREFIX} TYPE STRING HELP "default install folder" ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME}_${CMAKE_SYSTEM_NAME}_${TARGET_LIB_ARCH})
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
install(PROGRAMS ${RKNN_LIB} ${RKLLM_LIB} DESTINATION lib)
