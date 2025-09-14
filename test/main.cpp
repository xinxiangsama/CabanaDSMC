// 1. 告诉 Catch2：不要给我生成 main
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp> // Catch2 v3 头

// 2. 你的其他头（如果需要）
// #include "simulation/run.hpp"

// 3. 系统头
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
    // 3.1 初始化 MPI
    MPI_Init(&argc, &argv);

    // 3.2 初始化 Kokkos（ScopeGuard 会在出作用域时自动 finalise）
    Kokkos::ScopeGuard kokkos(argc, argv);

    // 3.3 把命令行交给 Catch2，跑所有 TEST_CASE
    int catch_result = Catch::Session().run(argc, argv);

    // 3.4 清理 MPI
    MPI_Finalize();

    // 3.5 把测试失败数返回给操作系统
    return catch_result;
}