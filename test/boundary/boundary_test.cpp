#include "../../src/boundary/boundary.hpp"
#include "../../src/mesh/mesh.hpp"
#include "../src/simulation/initialize.hpp"
#include "../../src/move.hpp"
#include <catch2/catch_all.hpp>

using namespace Catch;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using namespace CabanaDSMC;
TEST_CASE("Boundary Test", "[boundary]")
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
    SECTION("create complex boundary")
    {   
        using periodicBoundary_t = Boundary::PeriodicBoundary<double, particle_type>;
        double normal[3] = {1.0, 0.0, 0.0};
        using periodicBoundaryFactory_t = Boundary::BoundaryFactory<periodicBoundary_t,
                    typename Mesh::MeshManager<MemorySpace>::mesh_type> ;
        auto boundary1 = periodicBoundaryFactory_t::create(0.0, normal, mesh_manager->getGlobalMesh());
        auto boundary2 = periodicBoundaryFactory_t::create(1.0, normal, mesh_manager->getGlobalMesh());
        auto boundary3 = periodicBoundaryFactory_t::create(0.0, normal, mesh_manager->getGlobalMesh());
        auto complex_boundary = Boundary::makeComplexBoundary(boundary1, boundary2, boundary3);
        REQUIRE(complex_boundary.num_boundary == 3);
    }

    SECTION("test if apply work")
    {
        using periodicBoundary_t = Boundary::PeriodicBoundary<double, particle_type>;
        using periodicBoundaryFactory_t = Boundary::BoundaryFactory<periodicBoundary_t,
                    typename Mesh::MeshManager<MemorySpace>::mesh_type> ;
        double normal1[3] = {1.0, 0.0, 0.0};
        auto boundary1 = periodicBoundaryFactory_t::create(0.0, normal1, mesh_manager->getGlobalMesh(), 0.0, 1.0);

        double normal2[3] = {0.0, 1.0, 0.0};
        auto boundary2 = periodicBoundaryFactory_t::create(0.0, normal2, mesh_manager->getGlobalMesh(), 0.0, 1.0);

        double normal3[3] = {0.0, 0.0, -1.0};
        auto boundary3 = periodicBoundaryFactory_t::create(1.0, normal3, mesh_manager->getGlobalMesh(), 0.0, 1.0);

        double normal4[3] = {-1.0, 0.0, 0.0};
        auto boundary4 = periodicBoundaryFactory_t::create(1.0, normal4, mesh_manager->getGlobalMesh(), 0.0, 1.0);
        double normal5[3] = {0.0, -1.0, 0.0};
        auto boundary5 = periodicBoundaryFactory_t::create(1.0, normal5, mesh_manager->getGlobalMesh(), 0.0, 1.0);
        double normal6[3] = {0.0, 0.0, 1.0};
        auto boundary6 = periodicBoundaryFactory_t::create(0.0, normal6, mesh_manager->getGlobalMesh(), 0.0, 1.0);


        auto complex_boundary = Boundary::makeComplexBoundary(boundary1, boundary2, boundary3, boundary4, boundary5, boundary6);
        REQUIRE(complex_boundary.num_boundary == 6);
        // create a particle
        particle_type particle;
        Cabana::get(particle, Particle::Field::Position(), 0) = -0.1;
        Cabana::get(particle, Particle::Field::Position(), 1) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 2) = 0.5;
        complex_boundary.apply(particle);
        REQUIRE(Cabana::get(particle, Particle::Field::Position(), 0) == Approx(0.9));

        Cabana::get(particle, Particle::Field::Position(), 0) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 1) = - 0.5;
        Cabana::get(particle, Particle::Field::Position(), 2) = 0.5;
        complex_boundary.apply(particle);   
        REQUIRE(Cabana::get(particle, Particle::Field::Position(), 1) == Approx(0.5));

        Cabana::get(particle, Particle::Field::Position(), 0) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 1) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 2) = 1.5;
        complex_boundary.apply(particle);
        REQUIRE(Cabana::get(particle, Particle::Field::Position(), 2) == Approx(0.5));

        Cabana::get(particle, Particle::Field::Position(), 0) = 1.5;
        Cabana::get(particle, Particle::Field::Position(), 1) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 2) = 0.5;
        complex_boundary.apply(particle);
        REQUIRE(Cabana::get(particle, Particle::Field::Position(), 0) == Approx(0.5));

        Cabana::get(particle, Particle::Field::Position(), 0) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 1) = 1.5;
        Cabana::get(particle, Particle::Field::Position(), 2) = 0.5;
        complex_boundary.apply(particle);
        REQUIRE(Cabana::get(particle, Particle::Field::Position(), 1) == Approx(0.5));

        Cabana::get(particle, Particle::Field::Position(), 0) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 1) = 0.5;
        Cabana::get(particle, Particle::Field::Position(), 2) = -0.5;
        complex_boundary.apply(particle);
        REQUIRE(Cabana::get(particle, Particle::Field::Position(), 2) == Approx(0.5));


    }

    SECTION("test whether work with move")
    {
        using periodicBoundary_t = Boundary::PeriodicBoundary<double, particle_type>;
        using periodicBoundaryFactory_t = Boundary::BoundaryFactory<periodicBoundary_t,
                    typename Mesh::MeshManager<MemorySpace>::mesh_type> ;
        double normal1[3] = {1.0, 0.0, 0.0};
        auto boundary1 = periodicBoundaryFactory_t::create(0.0, normal1, mesh_manager->getGlobalMesh(), 0.0, 1.0);

        double normal2[3] = {0.0, 1.0, 0.0};
        auto boundary2 = periodicBoundaryFactory_t::create(0.0, normal2, mesh_manager->getGlobalMesh(), 0.0, 1.0);

        double normal3[3] = {0.0, 0.0, -1.0};
        auto boundary3 = periodicBoundaryFactory_t::create(1.0, normal3, mesh_manager->getGlobalMesh(), 0.0, 1.0);


        double normal4[3] = {-1.0, 0.0, 0.0};
        auto boundary4 = periodicBoundaryFactory_t::create(1.0, normal4, mesh_manager->getGlobalMesh(), 0.0, 1.0);
        double normal5[3] = {0.0, -1.0, 0.0};
        auto boundary5 = periodicBoundaryFactory_t::create(1.0, normal5, mesh_manager->getGlobalMesh(), 0.0, 1.0);
        double normal6[3] = {0.0, 0.0, 1.0};
        auto boundary6 = periodicBoundaryFactory_t::create(0.0, normal6, mesh_manager->getGlobalMesh(), 0.0, 1.0);

        auto complex_boundary1 = Boundary::makeComplexBoundary(boundary1, boundary2, boundary3);
        auto complex_boundary2 = Boundary::makeComplexBoundary(boundary4, boundary5, boundary6);
        auto position = particle_list.slice(Particle::Field::Position {});
        auto velocity = particle_list.slice(Particle::Field::Velocity {});
        std::cout << "Before move, first 5 particle positions and velocites:" << std::endl;
        for(int i = 0; i < 100; ++i){
            // move particles
            moveParticles(
                ExecutionSpace {},
                particle_list,
                local_grid,
                cell_data,
                complex_boundary1,
                complex_boundary2
            );
            Kokkos::fence("After Move");

            for(int i = 0; i < 5 && i < particle_list.size(); ++i){
                REQUIRE(position(i,0) >= 0.0);
                REQUIRE(position(i,0) <= 1.0);
                REQUIRE(position(i,1) >= 0.0);
                REQUIRE(position(i,1) <= 1.0);
                REQUIRE(position(i,2) >= 0.0);
                REQUIRE(position(i,2) <= 1.0);
            }
        }

        for(int i = 0; i < 5 && i < particle_list.size(); ++i){
            std::cout << "Particle " << i << ": ("
                      << position(i,0) << ", "
                      << position(i,1) << ", "
                      << position(i,2) << ") with velocity ("
                      << velocity(i,0) << ", "
                      << velocity(i,1) << ", "
                      << velocity(i,2) << ")" << std::endl;
        }
    }
}