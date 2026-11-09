cmake_minimum_required(VERSION 3.16)
project(pixelpeeper_native VERSION 1.0.0 LANGUAGES C CXX)

option(PP_BUILD_SHARED "Build libpixelpeeper as a shared library" ON)
option(PP_BUILD_CLI "Build the regression-test command-line driver" ON)
option(PP_BUILD_TESTS "Build the C ABI smoke test" ON)
option(PP_BUILD_JNI "Build the optional Android JNI bridge" OFF)
option(PP_BUILD_NAPI "Build the optional HarmonyOS Node-API bridge" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(PP_BUILD_SHARED)
    add_library(pixelpeeper SHARED src/pixelpeeper.cpp)
    target_compile_definitions(pixelpeeper PRIVATE PP_BUILD_SHARED)
    set_target_properties(pixelpeeper PROPERTIES
        OUTPUT_NAME pixelpeeper
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
else()
    add_library(pixelpeeper STATIC src/pixelpeeper.cpp)
    set_target_properties(pixelpeeper PROPERTIES OUTPUT_NAME pixelpeeper)
endif()

target_include_directories(pixelpeeper
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    INTERFACE $<INSTALL_INTERFACE:include>)

if(MSVC)
    target_compile_options(pixelpeeper PRIVATE /W4 /permissive-)
else()
    target_compile_options(pixelpeeper PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PP_BUILD_CLI)
    add_executable(pp_cli tools/pp_cli.cpp)
    target_link_libraries(pp_cli PRIVATE pixelpeeper)
endif()

if(PP_BUILD_TESTS)
    enable_testing()
    add_executable(pp_smoke tests/smoke.c)
    target_link_libraries(pp_smoke PRIVATE pixelpeeper)
    add_test(NAME pp_smoke COMMAND pp_smoke)
endif()

if(PP_BUILD_JNI)
    add_library(pixelpeeper_jni SHARED platform/android/pixelpeeper_jni.cpp)
    target_link_libraries(pixelpeeper_jni PRIVATE pixelpeeper)
    set_target_properties(pixelpeeper_jni PROPERTIES OUTPUT_NAME pixelpeeper_jni)
endif()

if(PP_BUILD_NAPI)
    # HarmonyOS imports the module by library name, so this target must be
    # libpixelpeeper.so. Build it in a separate build directory with
    # -DPP_BUILD_SHARED=OFF -DPP_BUILD_CLI=OFF -DPP_BUILD_TESTS=OFF.
    add_library(pixelpeeper_napi SHARED
        src/pixelpeeper.cpp
        platform/harmony/pixelpeeper_napi.cpp)
    target_include_directories(pixelpeeper_napi PRIVATE include)
    target_compile_definitions(pixelpeeper_napi PRIVATE PP_BUILD_SHARED)
    set_target_properties(pixelpeeper_napi PROPERTIES OUTPUT_NAME pixelpeeper)
endif()

include(GNUInstallDirs)
install(TARGETS pixelpeeper
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/pixelpeeper.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
