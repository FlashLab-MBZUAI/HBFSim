cmake_minimum_required(VERSION 3.20)
project(Obelisk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(OBELISK_BUILD_TESTS "Build unit tests" ON)
option(OBELISK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(OBELISK_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

include(FetchContent)

FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(tomlplusplus spdlog)

if(OBELISK_BUILD_TESTS)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.6.0
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(Catch2)
endif()

add_library(obelisk_core STATIC
    src/base/simulation.cc
    src/base/stats.cc
    src/base/packet_impl.cc
    src/frontend/trace_reader.cc
    src/frontend/requester.cc
    src/host/host_bus.cc
    src/controller/hbf_controller.cc
    src/controller/scheduler_fcfs.cc
    src/controller/scheduler_frfcfs.cc
    src/controller/scheduler_hbf_aware.cc
    src/controller/scheduler_factory.cc
    src/logic_die/logic_die.cc
    src/logic_die/channel_mux.cc
    src/logic_die/subarray_dispatcher.cc
    src/logic_die/tsv_model.cc
    src/media/nand/subarray.cc
    src/media/nand/nand_die.cc
    src/media/nand/nand_stack.cc
    src/media/hbm/hbm_interface.cc
    src/memory_system/address_mapper.cc
    src/memory_system/memory_system.cc
    src/config/config_loader.cc
)

target_include_directories(obelisk_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(obelisk_core PUBLIC
    tomlplusplus::tomlplusplus
    spdlog::spdlog
)

add_executable(obelisk src/main.cc)
target_link_libraries(obelisk PRIVATE obelisk_core)

if(OBELISK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
