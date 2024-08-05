add_library(mem_lk SHARED IMPORTED)
add_library(Ubject SHARED IMPORTED)
add_library(laud-c SHARED IMPORTED)
add_library(backtrace STATIC IMPORTED)

set_property(TARGET mem_lk PROPERTY IMPORTED_LOCATION
                                    ${CMAKE_INSTALL_PREFIX}/bin/libmem_lk.so)
set_property(TARGET Ubject PROPERTY IMPORTED_LOCATION
                                    ${CMAKE_INSTALL_PREFIX}/bin/libUbject.so)
set_property(TARGET laud-c PROPERTY IMPORTED_LOCATION
                                    ${CMAKE_INSTALL_PREFIX}/bin/liblaud-c.so)
set_property(
  TARGET backtrace PROPERTY IMPORTED_LOCATION
                            ${CMAKE_INSTALL_PREFIX}/bin/libbacktrace.a)

add_library(xab-c SHARED)
include(sources.cmake)
add_dependencies(xab-c laud-c)

target_link_libraries(
  xab-c
  PUBLIC laud-c mem_lk Ubject
  PRIVATE xab-c_flags xab-cxx_flags)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  # target_compile_definitions(xab-c -fsanitize)
  add_custom_command(
    TARGET xab-c
    POST_BUILD
    COMMAND echo "Config. Type: $<CONFIG>")
endif()

target_include_directories(xab-c PRIVATE ${CMAKE_INSTALL_PREFIX}/include/mem_lk)
target_include_directories(xab-c PRIVATE ${CMAKE_INSTALL_PREFIX}/include/Ubject)
target_include_directories(xab-c PRIVATE ${CMAKE_INSTALL_PREFIX}/include/laud)
target_include_directories(xab-c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

include(headers.cmake)
