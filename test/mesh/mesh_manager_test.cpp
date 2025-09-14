#include "../../src/mesh/mesh.hpp" // 包含你的MeshManager
#include <catch2/catch_all.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp> // 需要Kokkos
#include <iostream>
using namespace CabanaDSMC::Mesh;
using namespace Catch;
// 定义测试使用的设备类型
using MemorySpace = Kokkos::HostSpace; // 在单元测试中，通常在CPU上测试更方便

TEST_CASE("MeshManager Initialization and Properties", "[mesh_manager]")
{
    // --- 1. 准备测试参数 ---
    std::array<double, 3> low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> high_corner = {1.0, 2.0, 3.0}; // 使用不对称的尺寸
    std::array<int, 3> num_cells = {10, 20, 30};
    std::array<bool, 3> periodic = {false, true, false};
    unsigned int halo_width = 2;
    
    // 获取MPI信息
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    // --- 2. 创建被测试的对象 ---
    // 使用 std::make_unique 来管理生命周期
    auto mesh_manager = std::make_unique<MeshManager<MemorySpace>>(
        low_corner,
        high_corner,
        num_cells,
        MPI_COMM_WORLD,
        periodic,
        halo_width
    );
    
    // --- 3. 开始验证 ---
    
    SECTION("GlobalMesh properties are correct")
    {
        auto global_mesh = mesh_manager->getGlobalMesh();
        REQUIRE(global_mesh != nullptr); // 确保指针有效

        // 验证物理边界
        REQUIRE(global_mesh->lowCorner(0) == Approx(low_corner[0]));
        REQUIRE(global_mesh->lowCorner(1) == Approx(low_corner[1]));
        REQUIRE(global_mesh->lowCorner(2) == Approx(low_corner[2]));
        
        REQUIRE(global_mesh->highCorner(0) == Approx(high_corner[0]));
        REQUIRE(global_mesh->highCorner(1) == Approx(high_corner[1]));
        REQUIRE(global_mesh->highCorner(2) == Approx(high_corner[2]));
        
        // 验证单元总数
        REQUIRE(global_mesh->globalNumCell(0) == num_cells[0]);
        REQUIRE(global_mesh->globalNumCell(1) == num_cells[1]);
        REQUIRE(global_mesh->globalNumCell(2) == num_cells[2]);
        
        // 验证计算出的单元尺寸
        REQUIRE(global_mesh->cellSize(0) == Approx( (high_corner[0] - low_corner[0]) / num_cells[0] ));
    }

    SECTION("GlobalGrid properties are correct")
    {
        auto global_grid = mesh_manager->getGlobalGrid();
        REQUIRE(global_grid != nullptr);

        // 验证周期性
        REQUIRE(global_grid->isPeriodic(0) == periodic[0]);
        REQUIRE(global_grid->isPeriodic(1) == periodic[1]);
        REQUIRE(global_grid->isPeriodic(2) == periodic[2]);

        // 验证总进程数
        REQUIRE(global_grid->totalNumBlock() == comm_size);

        // 验证每个进程的 owned 单元数之和等于总单元数
        // 1. 获取本进程拥有的单元数 (三维)
        long long local_owned_count = 
            (long long)global_grid->ownedNumCell(0) *
            (long long)global_grid->ownedNumCell(1) *
            (long long)global_grid->ownedNumCell(2);
            
        // 2. 使用 MPI_Allreduce 将所有进程的本地计数值求和
        long long total_owned_count = 0;
        MPI_Allreduce(&local_owned_count, &total_owned_count, 1, 
                    MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

        // 3. 计算期望的全局总单元数
        long long expected_global_count = 
            (long long)num_cells[0] * num_cells[1] * num_cells[2];
        
        // 4. 验证两者是否相等
        REQUIRE(total_owned_count == expected_global_count);
        std::cout <<"my rank is "<<comm_rank<<" local cell num is "<< local_owned_count << std::endl;
        if(comm_rank == 0){
            std::cout <<"global cell num is "<< expected_global_count << std::endl;
        }
    }
}