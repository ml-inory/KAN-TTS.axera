cmake_minimum_required(VERSION 3.15)
project(kantts_sdk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include ${AX_RUNTIME_ROOT}/include)
link_directories(${AX_RUNTIME_ROOT}/lib)

add_library(kantts_tts STATIC
    src/ax_engine.cpp
    src/kantts.cpp
)
target_include_directories(kantts_tts PUBLIC include)
target_link_libraries(kantts_tts PRIVATE ax_engine ax_sys pthread)

add_executable(kantts_tts_example src/main.cpp)
target_link_libraries(kantts_tts_example PRIVATE kantts_tts ax_engine ax_sys)
