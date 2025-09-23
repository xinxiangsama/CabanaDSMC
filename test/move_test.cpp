#include "../src/move.hpp"
#include "../src/simulation/initialize.hpp"
#include "mesh/mesh.hpp"
#include "../src/cell/cell.hpp"
#include <catch2/catch_all.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>
using namespace Catch;
using namespace CabanaDSMC;

// 定义测试使用的设备类型 (在CPU上测试)
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

TEST_CASE("Particle move Test", "[particle move]")
{
    std::array<double, 3> low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> high_corner = {1.0, 1.0, 1.0};
    std::array<int, 3> num_cells = {2, 2, 2};
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

    SECTION("test move module")
    {   
        auto position = particle_list.slice(Particle::Field::Position {});
        auto velocity = particle_list.slice(Particle::Field::Velocity {});
        std::cout << "Before move, first 5 particle positions and velocites:" << std::endl;
        for(int i = 0; i < 5 && i < particle_list.size(); ++i){
            std::cout << "Particle " << i << ": ("
                      << position(i,0) << ", "
                      << position(i,1) << ", "
                      << position(i,2) << ") with velocity ("
                      << velocity(i,0) << ", "
                      << velocity(i,1) << ", "
                      << velocity(i,2) << ")" << std::endl;
        }
        for(int i = 0; i < 100; ++i){
        // move particles
            moveParticles(
                ExecutionSpace {},
                particle_list,
                local_grid,
                cell_data
            );

            // Kokkos::fence("After Move");
            // std::cout << "After move, first 5 particle positions:" << std::endl;
            // for(int i = 0; i < 5 && i < particle_list.size(); ++i){
            //     std::cout << "Particle " << i << ": ("
            //             << position(i,0) << ", "
            //             << position(i,1) << ", "
            //             << position(i,2) << ")" << std::endl;
            // }
        }
    }
}