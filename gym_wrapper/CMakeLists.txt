CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

PROJECT(vangers_gym_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main option for building shared engine library
OPTION(BUILD_SHARED_ENGINE "Build Vangers engine as shared library for gym wrapper" ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# Find SDL2 (required by game engine)
find_package(SDL2 REQUIRED)
find_package(SDL2_net)
if(NOT SDL2_NET_FOUND)
    message(WARNING "SDL2_net not found. Some networking features will be disabled.")
    set(SDL2_NET_LIBRARIES "")
    set(SDL2_NET_INCLUDE_DIRS "")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/xsound
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/xgraph
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/xtool
    ${SDL2_INCLUDE_DIR}
)

if(JSONCPP_FOUND)
    include_directories(${JSONCPP_INCLUDE_DIRS})
endif()

if(SDL2_NET_FOUND)
    include_directories(${SDL2_NET_INCLUDE_DIRS})
endif()

# Only build if BUILD_SHARED_ENGINE is enabled
if(BUILD_SHARED_ENGINE)

    # Engine library headers
    set(ENGINE_LIB_HEADERS
        vangers_engine_lib.h
    )

    # Engine library sources (minimal implementation)
    set(ENGINE_LIB_SOURCES
        vangers_engine_lib.cpp
    )

    # Engine core sources (full engine compiled as separate static lib)
    file(GLOB_RECURSE ENGINE_CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/*/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/xgraph/*.cpp
    )

    # Build engine core as static library
    add_library(vangers_core STATIC
        ${ENGINE_CORE_SOURCES}
    )

    target_include_directories(vangers_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/xgraph
    )

    # Create shared engine library for in-process execution (wrapper)
    add_library(vangers_engine SHARED
        ${ENGINE_LIB_SOURCES}
        ${ENGINE_LIB_HEADERS}
    )

    # Link engine core into the shared wrapper library. Use PUBLIC so that
    # consumers of the shared library (example binaries) also link the
    # transitive dependencies resolved by the engine core and third-party
    # libraries.
    #
    # The engine core depends on the helper/utility static library `xtool`
    # (which contains iniparser and related objects). Ensure the shared
    # vangers_engine target links xtool as well so symbols such as the
    # iniparser functions are provided to the produced shared library.
    target_link_libraries(vangers_engine PUBLIC vangers_core xtool)

    # Compiler flags for shared library
    target_compile_options(vangers_engine PRIVATE
        -Wall -Wextra -fPIC
        -DVANGERS_ENGINE_SHARED
        -DHEADLESS_MODE
        -DDETERMINISTIC_MODE
        -D_USE_MATH_DEFINES
    )

    if(JSONCPP_FOUND)
        target_compile_definitions(vangers_engine PRIVATE -DHAVE_JSONCPP)
        target_compile_options(vangers_engine PRIVATE ${JSONCPP_CFLAGS_OTHER})
    endif()

    # Link libraries for shared engine
    target_link_libraries(vangers_engine PRIVATE
        ${CLUNK_LIBRARY}
        ${FFMPEG_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${OGG_LIBRARY}
        ${VORBIS_LIBRARY}
        ${VORBISFILE_LIBRARY}
        ${SDL2_LIBRARY}
        pthread
        dl
        m
        stdc++
    )

    if(JSONCPP_FOUND)
        target_link_libraries(vangers_engine PRIVATE ${JSONCPP_LIBRARIES})
    endif()

    if(SDL2_NET_FOUND)
        target_link_libraries(vangers_engine PRIVATE ${SDL2_NET_LIBRARIES})
    endif()

    # Set shared library properties
    set_target_properties(vangers_engine PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
        PUBLIC_HEADER "vangers_engine_lib.h"
    )

    # Python package files
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
        ${CMAKE_CURRENT_BINARY_DIR}/setup.py
        COPYONLY
    )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/vangers_env.py
        ${CMAKE_CURRENT_BINARY_DIR}/vangers_env.py
        COPYONLY
    )

    # Copy the built shared library into the python package directory so it can be
    # bundled with the Python package (wheel/sdist). This places the produced
    # shared object next to the Python sources (e.g. gym_wrapper/libvangers_engine.so)
    # which makes it straightforward to include as package_data.
    #
    # The copy is performed as a POST_BUILD step on the library target so the
    # artifact is available in the source package tree after building.
    add_custom_command(TARGET vangers_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vangers_engine> ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:vangers_engine>
        COMMENT "Copying vangers_engine shared library into Python package directory"
    )

    # Examples
    option(BUILD_EXAMPLES "Build example programs" ON)

    if(BUILD_EXAMPLES)
        # C++ test client for shared library
        add_executable(engine_test_client
            examples/engine_test_client.cpp
        )

        target_link_libraries(engine_test_client
            vangers_engine
        )

        # Copy Python examples
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/python_example.py
            ${CMAKE_CURRENT_BINARY_DIR}/examples/python_example.py
            COPYONLY
        )

        # Create examples directory
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/examples)

        install(TARGETS engine_test_client
            RUNTIME DESTINATION bin
        )
    endif()

    # Install targets
    install(TARGETS vangers_engine
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/vangers_engine
    )

    # pkg-config file
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/vangers_engine.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/vangers_engine.pc
        @ONLY
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vangers_engine.pc
        DESTINATION lib/pkgconfig
    )

    # Python package installation target
    add_custom_target(install-python
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
            python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py install --user
        DEPENDS vangers_engine
        COMMENT "Installing Python package"
    )

    message(STATUS "Vangers Shared Engine Library: ENABLED")

else()
    message(STATUS "Vangers Shared Engine Library: DISABLED")
    message(STATUS "  Enable with: cmake -DBUILD_SHARED_ENGINE=ON")
endif()

# Debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(BUILD_SHARED_ENGINE)
        target_compile_definitions(vangers_engine PRIVATE
            -DGYM_DEBUG
            -DENGINE_DEBUG
        )
    endif()
endif()

# Optimization flags for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(BUILD_SHARED_ENGINE)
        target_compile_options(vangers_engine PRIVATE
            -O3 -DNDEBUG
        )
    endif()
endif()

# Summary
message(STATUS "Vangers Gym Wrapper Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared Engine Library: ${BUILD_SHARED_ENGINE}")
if(BUILD_SHARED_ENGINE)
    message(STATUS "  Examples: ${BUILD_EXAMPLES}")
endif()
