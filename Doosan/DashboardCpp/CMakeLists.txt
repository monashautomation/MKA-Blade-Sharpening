cmake_minimum_required(VERSION 3.16)
project(dashboard_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Third-party single-header libs (fetched at configure time) ----
include(FetchContent)

FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)
FetchContent_MakeAvailable(json)

# ---- libmodbus (system) ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(MODBUS REQUIRED libmodbus)

# ---- DRFL (vendored under Doosan/API-DRFL, linked when we wire up drl_start) ----
# Uncomment when adopting per-file architecture:
# set(DRFL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../API-DRFL)
# add_library(drfl SHARED IMPORTED)
# set_target_properties(drfl PROPERTIES
#     IMPORTED_LOCATION ${DRFL_ROOT}/library/Linux/libDRFL.so
#     INTERFACE_INCLUDE_DIRECTORIES ${DRFL_ROOT}/include)

add_executable(dashboard_cpp main.cpp)
target_include_directories(dashboard_cpp PRIVATE ${MODBUS_INCLUDE_DIRS})
target_link_libraries(dashboard_cpp PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${MODBUS_LIBRARIES}
    pthread
    # drfl                       # enable with DRFL block above
)

# index.html sits next to the binary so cpp-httplib's mount_point finds it.
add_custom_command(TARGET dashboard_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../Dashboard/index.html
        $<TARGET_FILE_DIR:dashboard_cpp>/index.html)
