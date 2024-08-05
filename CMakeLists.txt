cmake_minimum_required(VERSION 3.21)

project(xab)

add_library(xab-c_flags INTERFACE)
target_compile_features(xab-c_flags INTERFACE c_std_17)
target_compile_options(
  xab-c_flags
  INTERFACE
    # "$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
    # "$<BUILD_INTERFACE:-Wall;-Wl,--exclude-libs>"
    "$<BUILD_INTERFACE:$<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:-fPIC;-ggdb3;-O0;-fno-omit-frame-pointer;-fsanitize=address>>"
)

target_compile_definitions(
  xab-c_flags INTERFACE "$<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:IU_MEM_LK=1>")

add_library(xab-cxx_flags INTERFACE)
target_compile_features(xab-cxx_flags INTERFACE cxx_std_17)
target_compile_options(
  xab-cxx_flags
  INTERFACE
    "$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
    "$<BUILD_INTERFACE:$<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:-fPIC;-ggdb3;-O0;-fno-omit-frame-pointer;-fsanitize=address>>"
)
target_link_options(
  xab-cxx_flags
  INTERFACE
  "$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
  "$<BUILD_INTERFACE:$<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:-fPIC;-ggdb3;-O0;-fno-omit-frame-pointer;-fsanitize=address>>"
)

target_compile_definitions(
  xab-cxx_flags
  INTERFACE "$<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:IU_MEM_LK=1>")

add_executable(xab main.cpp)

add_subdirectory(c)

target_link_libraries(xab xab-c xab-cxx_flags xab-c_flags)

target_include_directories(xab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS xab xab-c DESTINATION bin)
install(FILES xab_defines.h DESTINATION include/${PROJECT_NAME})
