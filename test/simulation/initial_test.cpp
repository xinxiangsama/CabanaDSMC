#include "../../src/simulation/initialize.hpp"
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
    std::array<int, 3> num_cells = {2, 2, 2};
    std::array<bool, 3> periodic = {false, false, false};
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
    field_data.time_step = 1e-7;

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

    SECTION("Initialize CellData Fields")
    {
        // check some values in cell_data arrays
        auto velocity = cell_data->Velocity->view();
        auto temperature = cell_data->Temperature->view();
        auto density = cell_data->Density->view();
        auto num_particles = cell_data->Num_particles->view();
        auto fn = cell_data->Fn->view();
        auto max_collision_rate = cell_data->Max_collision_rate->view();
        auto volume = cell_data->Volume->view();

        auto owned_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());


        for (int i = owned_space.min(0); i < owned_space.max(0); ++i)
        {
            for (int j = owned_space.min(1); j < owned_space.max(1); ++j)
            {
            for (int k = owned_space.min(2); k < owned_space.max(2); ++k)
            {
                REQUIRE(velocity(i, j, k, 0) == Approx(field_data.velocity[0]));
                REQUIRE(velocity(i, j, k, 1) == Approx(field_data.velocity[1]));
                REQUIRE(velocity(i, j, k, 2) == Approx(field_data.velocity[2]));
                REQUIRE(temperature(i, j, k, 0) == Approx(field_data.temperature));
                REQUIRE(density(i, j, k, 0) == Approx(field_data.density));
                REQUIRE(fn(i, j, k, 0) == Approx(field_data.Fn));
                REQUIRE(max_collision_rate(i, j, k, 0) == Approx(field_data.max_collision_rate));
            }
            }
        }
    }
    Kokkos::fence("Before Particle Initialization");


    //create particle list
    Particle::GridParticleList<MemorySpace, 16> particle_list("particles");
    using particle_type = typename Particle::GridParticleList<MemorySpace, 16>::particle_type;

    auto InitFunctor = ParticleFactory<particle_type>{};
    SECTION("Initialize ParticleLists")
    {
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

        auto& aosoa = particle_list.aosoa();
        auto num_particles = aosoa.size();
        std::cout << "Number of particles created: " << num_particles << std::endl;
    }
}