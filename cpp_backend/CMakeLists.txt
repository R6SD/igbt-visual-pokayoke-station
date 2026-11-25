cmake_minimum_required(VERSION 3.20)
project(igbt_pokayoke_cpp VERSION 4.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MINGW OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -O3 -Wa,-mbig-obj)
endif()

add_compile_definitions(
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
    UNICODE
    _UNICODE
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn ml)

find_path(SQLITE3_INCLUDE_DIR sqlite3.h REQUIRED)
find_library(SQLITE3_LIBRARY NAMES sqlite3 libsqlite3 REQUIRED)

set(IGBT_CORE_SOURCES
    src/fsm.cpp
    src/vision_engine.cpp
    src/training_manager.cpp
    src/database.cpp
    src/http_server.cpp
)

add_library(igbt_core STATIC ${IGBT_CORE_SOURCES})
target_include_directories(igbt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
    ${SQLITE3_INCLUDE_DIR}
)
target_link_libraries(igbt_core PUBLIC
    ${OpenCV_LIBS}
    ${SQLITE3_LIBRARY}
    ws2_32
    wsock32
    bcrypt
    crypt32
)
if (WIN32)
    target_link_libraries(igbt_core PUBLIC
        ole32
        dbghelp
    )
endif()

add_executable(igbt_station_server src/main.cpp)
target_link_libraries(igbt_station_server PRIVATE igbt_core)

enable_testing()
add_executable(test_cpp_station tests/test_cpp_station.cpp)
target_link_libraries(test_cpp_station PRIVATE igbt_core)
add_test(NAME test_cpp_station COMMAND test_cpp_station)
