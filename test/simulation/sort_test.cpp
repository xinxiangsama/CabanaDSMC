#include "../../src/boundary/boundary.hpp"
#include "../../src/mesh/mesh.hpp"
#include "../src/simulation/initialize.hpp"
#include "../../src/move.hpp"
#include <catch2/catch_all.hpp>
#include <Cabana_Core.hpp>

using namespace Catch;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using namespace CabanaDSMC;
TEST_CASE("Sort Test", "[sort]")
{
    std::array<double, 3> low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> high_corner = {1.0, 1.0, 1.0};
    std::array<int, 3> num_cells = {2, 2, 2};
    std::array<bool, 3> periodic = {false, false, false};
    unsigned int halo_width = 0;
    
    auto mesh_manager = std::make_shared<Mesh::MeshManager<MemorySpace>>(
        low_corner, high_corner, num_cells, MPI_COMM_WORLD, periodic, halo_width
    );
    auto local_grid = mesh_manager->getLocalGrid();
   // --- 2. 创建被测试的CellData对象 ---
    auto cell_data = std::make_shared<Cell::CellData<double, MemorySpace, Cabana::Grid::UniformMesh<double, 3>>>(local_grid);

    // --- 3. 定义初始化数据 ---
    FieldInitData<double, 3> field_data;
    field_data.velocity[0] = 1.0;
    field_data.velocity[1] = 0.0;
    field_data.velocity[2] = 0.0;
    field_data.temperature = 300.0;
    field_data.density = 1.225e-3;
    field_data.Fn = 1e18;
    field_data.max_collision_rate = 1e6;
    field_data.time_step = 1e-4;

    // --- 定义分子种类列表 ---
    Particle::SpeciesList<MemorySpace> species_list("species_list", 1);
    species_list(0).mass = 6.63e-26; // 质量 (kg) - 氮气分子
    species_list(0).diameter = 3.7e-10; // 直径 (m)
    species_list(0).omega = 0.74; // 粘性指数
    species_list(0).Tref = 300.0; // 参考温度 (K)
    species_list(0).Zrot = 1.0; // 转动碰撞数
    initializeField<ExecutionSpace>(
        field_data,
        cell_data,
        species_list
    );

    //create particle list
    Particle::GridParticleList<MemorySpace, 16> particle_list("particles");
    using particle_type = typename Particle::GridParticleList<MemorySpace, 16>::particle_type;
    auto InitFunctor = ParticleFactory<particle_type>{};
    createParticles(
        Cabana::InitRandom {},
        ExecutionSpace {},
        InitFunctor,
        particle_list,
        local_grid,
        cell_data,
        species_list,
        0, 12345
    );
    SECTION("migration test")
    {   
        auto position = particle_list.slice(Particle::Field::Position {});
        auto local_grid = mesh_manager->getLocalGrid();
        auto local_mesh = mesh_manager->getLocalMesh();
        double grid_min[3], grid_max[3], grid_delta[3];
        for (int d = 0; d < 3; ++d) {
            grid_min[d] = local_mesh->lowCorner(Cabana::Grid::Ghost(), d);
            grid_max[d] = local_mesh->highCorner(Cabana::Grid::Ghost(), d);
            grid_delta[d]  = local_grid->globalGrid().globalMesh().cellSize(d);
        }
        auto linked_cell_list = Cabana::createLinkedCellList<MemorySpace>(
            position, grid_delta, grid_min, grid_max);
        Cabana::permute(linked_cell_list, particle_list.aosoa());
        Kokkos::fence();
        // for(int i {}; i < local_grid->globalGrid().ownedNumCell(0); ++i)
        //     for(int j {}; j < local_grid->globalGrid().ownedNumCell(1); ++j)
        //         for(int k {}; k < local_grid->globalGrid().ownedNumCell(2); ++k)
        //         {
        //             std::cout << "Cell (" << i << ", " << j << ", " << k << "): ";
        //             std::cout <<"num particles = " << linked_cell_list.binSize(i,j,k) << "\n";
        //             std::cout <<"particle start index = " << linked_cell_list.binOffset(i,j,k) << "\n";
        //         }
        
    }
}