#include "../../src/cell/cell.hpp" // 包含你的CellData
#include "mesh/mesh.hpp" // 需要MeshManager来创建LocalGrid
#include <catch2/catch_all.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>

using namespace CabanaDSMC;
using namespace Catch;


// 定义测试使用的设备类型
using MemorySpace = Kokkos::HostSpace; // 在CPU上进行测试
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

TEST_CASE("CellData Initialization", "[cell_data]")
{
    // --- 1. 准备一个有效的LocalGrid作为输入 ---
    //    我们复用MeshManager来为我们完成这个复杂的任务
    std::array<double, 3> low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> high_corner = {1.0, 1.0, 1.0};
    std::array<int, 3> num_cells = {10, 10, 10};
    std::array<bool, 3> periodic = {true, true, true};
    unsigned int halo_width = 2;
    
    auto mesh_manager = std::make_shared<Mesh::MeshManager<MemorySpace>>(
        low_corner, high_corner, num_cells, MPI_COMM_WORLD, periodic, halo_width
    );
    auto local_grid = mesh_manager->getLocalGrid();
    REQUIRE(local_grid != nullptr);

    // --- 2. 创建被测试的CellData对象 ---
    auto cell_data = std::make_unique<Cell::CellData<double, MemorySpace, Cabana::Grid::UniformMesh<double, 3>>>(local_grid);
    
    // --- 3. 开始验证 ---
    
    SECTION("All Array members are successfully created")
    {
        REQUIRE(cell_data->Velocity != nullptr);
        REQUIRE(cell_data->Temperature != nullptr);
        REQUIRE(cell_data->Pressure != nullptr);
        REQUIRE(cell_data->Density != nullptr);
        REQUIRE(cell_data->Max_collision_rate != nullptr);
    }
    
    SECTION("Array layouts have correct degrees of freedom (DoFs)")
    {
        // 验证矢量场
        REQUIRE(cell_data->Velocity->layout()->dofsPerEntity() == 3);
        
        // 验证标量场
        REQUIRE(cell_data->Temperature->layout()->dofsPerEntity() == 1);
        REQUIRE(cell_data->Pressure->layout()->dofsPerEntity() == 1);
        REQUIRE(cell_data->Density->layout()->dofsPerEntity() == 1);
        REQUIRE(cell_data->Max_collision_rate->layout()->dofsPerEntity() == 1);
    }
    
    using namespace Cabana::Grid;
    SECTION("Array views have correct dimensions")
    {
        // 获取本地网格（包括幽灵层）的索引空间
        auto ghosted_space = local_grid->indexSpace(Ghost(), Cabana::Grid::Cell(), Local());
        
        // --- 验证Velocity (矢量场) ---
        auto vel_view = cell_data->Velocity->view();
        // 维度应该是4D: (i, j, k, dof)
        REQUIRE(vel_view.rank == 4); 
        // 验证每个维度的长度
        REQUIRE(vel_view.extent(0) == ghosted_space.extent(0));
        REQUIRE(vel_view.extent(1) == ghosted_space.extent(1));
        REQUIRE(vel_view.extent(2) == ghosted_space.extent(2));
        REQUIRE(vel_view.extent(3) == 3); // 3个自由度

        // --- 验证Density (标量场) ---
        auto den_view = cell_data->Density->view();
        REQUIRE(den_view.rank == 4);
        REQUIRE(den_view.extent(0) == ghosted_space.extent(0));
        REQUIRE(den_view.extent(1) == ghosted_space.extent(1));
        REQUIRE(den_view.extent(2) == ghosted_space.extent(2));
        REQUIRE(den_view.extent(3) == 1); // 1个自由度
    }
    
    // SECTION("Array values can be accessed and modified")
    {
        // 这是一个更深入的测试，确保我们可以实际使用这些Array
        
        // 1. 获取一个Array并将其所有值设为5.0
        auto density_array = cell_data->Density;
        Cabana::Grid::ArrayOp::assign(*density_array, 5.0, Ghost());
        
        // 2. 将数据拷贝回主机进行验证
        auto density_view = density_array->view();
        auto host_density_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density_view);
        
        // 3. 检查某个点的值
        //    (注意：这里的索引是本地索引)
        auto owned_space = local_grid->indexSpace(Own(), Cabana::Grid::Cell(), Local());
        int test_i = owned_space.min(0);
        int test_j = owned_space.min(1);
        int test_k = owned_space.min(2);
        
        REQUIRE(host_density_view(test_i, test_j, test_k, 0) == Approx(5.0));
    }
}