# ====================================================================
# 项目基本设置
# ====================================================================
cmake_minimum_required(VERSION 3.16)
project(cabanaDSMC LANGUAGES CXX C)

# C++标准设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# 本地库目录
set(LOCAL_LIB_DIR "/home/xinxiangsama/local" CACHE PATH "Local library installation directory")

# ====================================================================
# 依赖包配置
# ====================================================================
message(STATUS "Searching for required libraries...")

# MPI依赖
find_package(MPI REQUIRED)
message(STATUS "Found MPI.")
find_package(OpenMP REQUIRED)
message(STATUS "Found OpenMP:")
message(STATUS "  OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "  OpenMP CXX libraries: ${OpenMP_CXX_LIBRARIES}")
message(STATUS "  OpenMP CXX include dirs: ${OpenMP_CXX_INCLUDE_DIRS}")

find_package(Catch2 REQUIRED)
message(STATUS "Found Catch2")
# Kokkos依赖
set(Kokkos_ROOT "${LOCAL_LIB_DIR}/kokkos" CACHE PATH "Root directory of Kokkos installation")
find_package(Kokkos REQUIRED)
message(STATUS "Found Kokkos.")

# cabana依赖
set(Cabana_ROOT "${LOCAL_LIB_DIR}/cabana" CACHE PATH "Root directory of cabana installation")
find_package(Cabana REQUIRED)
message(STATUS "Found Cabana.")

# # ArborX依赖
# set(ArborX_ROOT "${LOCAL_LIB_DIR}/arborx" CACHE PATH "Root directory of ArborX installation")
# find_package(ArborX REQUIRED)
# message(STATUS "Found ArborX.")

# yaml cpp依赖
set(yaml-cpp_ROOT "${LOCAL_LIB_DIR}/yaml_cpp" CACHE PATH "Root directory of yaml cpp installation")
find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp.")

# ====================================================================
# 第三方库配置
# ====================================================================
# 包含目录
set(INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/src"
    # "${LOCAL_LIB_DIR}/vtk-9.5/include/vtk-9.5"
    "${LOCAL_LIB_DIR}/ParMETIS/include"
    "${LOCAL_LIB_DIR}/METIS/include"
)

# 链接目录
set(LINK_DIRS
    "${LOCAL_LIB_DIR}/vtk/lib"
)

# 链接库列表
set(LINK_LIBS
    # "vtkCommonCore-9.5"
    # "vtksys-9.5"
    # "vtkCommonDataModel-9.5"
    # "vtkIOXML-9.5"
    # "vtkIOParallelXML-9.5"
    # "vtkRenderingOpenGL2-9.5"
    # "vtkRenderingCore-9.5"
)

# ====================================================================
# 可执行文件目标
# ====================================================================
# 创建可执行文件
add_executable(cabanaDSMC)

# 收集源文件
file(GLOB_RECURSE APP_SOURCES "src/*.cpp")
target_sources(cabanaDSMC PRIVATE ${APP_SOURCES})

# 设置包含目录
target_include_directories(cabanaDSMC PUBLIC ${INCLUDE_DIRS})

# 设置链接目录
target_link_directories(cabanaDSMC PUBLIC ${LINK_DIRS})

# 链接库 - Kokkos优先
target_link_libraries(cabanaDSMC PUBLIC Kokkos::kokkos)

# 链接库 
target_link_libraries(cabanaDSMC PUBLIC Cabana::Grid)

# 链接其他库和MPI
target_link_libraries(cabanaDSMC PUBLIC ${LINK_LIBS} MPI::MPI_CXX Catch2::Catch2 yaml-cpp::yaml-cpp)

# ====================================================================
# 编译和链接选项
# ====================================================================
# 通用编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall ")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif()

# 链接选项 - 解决PIE问题
target_link_options(cabanaDSMC PRIVATE -no-pie)

# 显示配置摘要
message(STATUS "Project configured successfully")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Local library directory: ${LOCAL_LIB_DIR}")


# ===================================================================
# 单元测试支持
# ===================================================================
# 1. 开启测试功能（必须写在最前）
enable_testing()
include(CTest)          # 提供 add_test / ctest 命令

# 2. 收集测试源文件
file(GLOB_RECURSE TEST_SOURCES
     "${CMAKE_SOURCE_DIR}/test/*.cpp"
     "${CMAKE_SOURCE_DIR}/test/*.cc"
    )

# 3. 创建测试可执行文件
add_executable(cabanaDSMC_tests ${TEST_SOURCES})

# 4. 让测试用主项目的所有 include / link 设置
target_include_directories(cabanaDSMC_tests PRIVATE ${INCLUDE_DIRS})
target_link_directories(cabanaDSMC_tests PRIVATE ${LINK_DIRS})

# 5. 链接库：复用主目标的库，再加上 Catch2 自带的 Catch2WithMain
target_link_libraries(cabanaDSMC_tests
                      PRIVATE
                      Kokkos::kokkos
                      Cabana::Grid
                      ${LINK_LIBS}
                      MPI::MPI_CXX
                      Catch2::Catch2)

# 6. 统一编译选项
target_compile_options(cabanaDSMC_tests PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(cabanaDSMC_tests PRIVATE ${OpenMP_CXX_FLAGS} -no-pie)

# 7. 自动发现 Catch2 测试并注册到 CTest
include(Catch)          # CMake 3.16 自带 FindCatch2 后提供
catch_discover_tests(cabanaDSMC_tests)

# 8. （可选）把测试目标命名为 check，习惯上 make check 即可跑测试
add_custom_target(check
                  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                  DEPENDS cabanaDSMC_tests)