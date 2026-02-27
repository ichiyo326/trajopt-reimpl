cmake_minimum_required(VERSION 3.16)
project(trajopt_reimpl VERSION 0.1.0 LANGUAGES CXX)

# ============================================================
# Compiler settings
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address -Wall -Wextra")

# ============================================================
# Dependencies
# ============================================================

# Eigen (header-only linear algebra)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Optional: OpenMP for parallel collision checking
find_package(OpenMP QUIET)

# ============================================================
# Library: trajopt (header-only)
# ============================================================
add_library(trajopt INTERFACE)
target_include_directories(trajopt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(trajopt INTERFACE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(trajopt INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - parallel collision checking enabled")
endif()

# ============================================================
# Demo executables
# ============================================================

# 2D planar arm demo
add_executable(arm_2d_demo demo/arm_2d/arm_2d_demo.cpp)
target_link_libraries(arm_2d_demo PRIVATE trajopt)
target_compile_options(arm_2d_demo PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -Wall>
)

# ============================================================
# Tests
# ============================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # SE(3) math tests
    add_executable(test_se3 tests/test_se3.cpp)
    target_link_libraries(test_se3 PRIVATE trajopt)
    add_test(NAME se3_tests COMMAND test_se3)

    # GJK distance tests
    add_executable(test_gjk tests/test_gjk.cpp)
    target_link_libraries(test_gjk PRIVATE trajopt)
    add_test(NAME gjk_tests COMMAND test_gjk)

    # SQP solver tests
    add_executable(test_scp tests/test_scp.cpp)
    target_link_libraries(test_scp PRIVATE trajopt)
    add_test(NAME scp_tests COMMAND test_scp)
endif()

# ============================================================
# Installation
# ============================================================
install(TARGETS trajopt EXPORT trajoptTargets)
install(DIRECTORY include/ DESTINATION include)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Demos: arm_2d_demo")
if(BUILD_TESTS)
    message(STATUS "Tests: test_se3, test_gjk, test_scp")
endif()
