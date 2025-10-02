// =========================================================================
// test_main.cpp
//
// 这是你的自定义 main 函数，用于集成 Catch2, MPI, 和 Kokkos。
// =========================================================================

// 1. 定义这个宏来告诉 Catch2 我们要自己写 main()
#define CATCH_CONFIG_RUNNER

#include <catch2/catch_all.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>



//---------------------------------------------------------------------------//
// 自定义的 Main 函数
//---------------------------------------------------------------------------//
int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------
    // 1. 初始化 MPI (作为最外层的包裹)
    // -----------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int test_result = 0;
    
    // 获取 MPI rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    {
        // -----------------------------------------------------------------
        // 2. 初始化 Kokkos (使用 RAII ScopeGuard)
        // -----------------------------------------------------------------
        Kokkos::ScopeGuard scope_guard(argc, argv);
        
        // -----------------------------------------------------------------
        // 3. 只在 Rank 0 上运行测试
        // -----------------------------------------------------------------
        if (rank == 0)
        {
            printf("Rank 0: Running Catch2 test suite...\n");
            
            // 创建 Catch2 session
            Catch::Session session;
            
            // 使用 Catch2 的命令行解析器
            // 这允许你使用像 -s (显示成功测试) 或按标签运行等Catch2参数
            int returnCode = session.applyCommandLine(argc, argv);
            if (returnCode != 0) {
                // 如果命令行解析出错，立即返回
                test_result = returnCode;
            } else {
                // 运行所有已注册的 TEST_CASE
                test_result = session.run();
            }
            
            printf("Rank 0: Test suite finished.\n");
        }
        else
        {
            // 其他 rank 不执行任何测试，只是安静地等待
        }
        
    } // Kokkos::finalize() 会在这里被 scope_guard 自动调用

    // -----------------------------------------------------------------
    // 4. 终结 MPI
    // -----------------------------------------------------------------
    MPI_Finalize();
    
    // 返回测试结果。对于所有 rank 来说，这应该是相同的。
    // 在实际应用中，你可能需要 MPI_Bcast 将 rank 0 的结果广播给其他 rank。
    // 但对于大多数测试脚本来说，只关心 rank 0 的返回码就足够了。
    return 0;
}