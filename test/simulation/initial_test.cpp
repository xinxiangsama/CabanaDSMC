#include "../../src/simulation/initialize.hpp" // 包含被测试的模块
#include "mesh/mesh.hpp"
#include "../../src/cell/cell.hpp"
#include <catch2/catch_all.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
using namespace Catch;
using namespace CabanaDSMC;

// 定义测试使用的设备类型 (在CPU上测试)
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

TEST_CASE("FieldInitializer Test", "[initialization]")
{
    std::array<double, 3> low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> high_corner = {1.0, 1.0, 1.0};
    std::array<int, 3> num_cells = {10, 10, 10};
    std::array<bool, 3> periodic = {true, true, true};
    unsigned int halo_width = 0;
    
    auto mesh_manager = std::make_shared<Mesh::MeshManager<MemorySpace>>(
        low_corner, high_corner, num_cells, MPI_COMM_WORLD, periodic, halo_width
    );
    auto local_grid = mesh_manager->getLocalGrid();
    REQUIRE(local_grid != nullptr);

    // --- 2. 创建被测试的CellData对象 ---
    auto cell_data = std::make_shared<Cell::CellData<double, MemorySpace, Cabana::Grid::UniformMesh<double, 3>>>(local_grid);

    // --- 3. 定义初始化数据 ---
    FieldInitData<double, 3> field_data;
    field_data.velocity[0] = 1.0;
    field_data.velocity[1] = 0.0;
    field_data.velocity[2] = 0.0;
    field_data.temperature = 300.0;
    field_data.density = 1.225;
    field_data.Fn = 1e15;
    field_data.max_collision_rate = 1e6;

    // --- 定义分子种类列表 ---
    Particle::SpeciesList<MemorySpace> species_list("species_list", 1);
    species_list(0).mass = 6.63e-26; // 质量 (kg) - 氮气分子
    species_list(0).diameter = 3.7e-10; // 直径 (m)
    species_list(0).omega = 0.74; // 粘性指数
    species_list(0).Tref = 300.0; // 参考温度 (K)
    species_list(0).Zrot = 1.0; // 转动碰撞数

    SECTION("Initialize CellData Fields")
    {
        initializeField<ExecutionSpace>(
            field_data,
            cell_data,
            species_list
        );

        // check some values in cell_data arrays
        auto velocity = cell_data->Velocity->view();
        auto temperature = cell_data->Temperature->view();
        auto density = cell_data->Density->view();
        auto num_particles = cell_data->Num_particles->view();
        auto fn = cell_data->Fn->view();
        auto max_collision_rate = cell_data->Max_collision_rate->view();
        auto volume = cell_data->Volume->view();

        auto owned_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());
        int test_i = owned_space.min(0);
        int test_j = owned_space.min(1);
        int test_k = owned_space.min(2);
        REQUIRE(velocity(test_i, test_j, test_k, 0) == Approx(field_data.velocity[0]));
        REQUIRE(velocity(test_i, test_j, test_k, 1) == Approx(field_data.velocity[1]));
        REQUIRE(velocity(test_i, test_j, test_k, 2) == Approx(field_data.velocity[2]));
        REQUIRE(temperature(test_i, test_j, test_k, 0) == Approx(field_data.temperature));
        REQUIRE(density(test_i, test_j, test_k, 0) == Approx(field_data.density));
        REQUIRE(fn(test_i, test_j, test_k, 0) == Approx(field_data.Fn));
        REQUIRE(max_collision_rate(test_i, test_j, test_k, 0) == Approx(field_data.max_collision_rate));
        REQUIRE(volume(test_i, test_j, test_k, 0) == Approx(0.001)); // 体积在初始化中被设为0.001

        std::cout << "Velocity: (" 
                  << velocity(test_i, test_j, test_k, 0) << ", "
                  << velocity(test_i, test_j, test_k, 1) << ", "
                  << velocity(test_i, test_j, test_k, 2) << ")\n";
        std::cout << "Temperature: " << temperature(test_i, test_j, test_k, 0) << "\n";
        std::cout << "Density: " << density(test_i, test_j, test_k, 0) << "\n";
        std::cout << "Num Particles: " << num_particles(test_i, test_j, test_k, 0) << "\n";
        std::cout << "Fn: " << fn(test_i, test_j, test_k, 0) << "\n";
        std::cout << "Max Collision Rate: " << max_collision_rate(test_i, test_j, test_k, 0) << "\n";
        std::cout << "Volume: " << volume(test_i, test_j, test_k, 0) << "\n";
    }
}