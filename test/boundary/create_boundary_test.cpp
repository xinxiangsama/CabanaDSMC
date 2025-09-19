#include "../../src/boundary/boundary.hpp"
#include "../../src/mesh/mesh.hpp"
#include <catch2/catch_all.hpp>

using namespace Catch;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using namespace CabanaDSMC;
TEST_CASE("create Boundary Test", "[boundary]")
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
    Particle::GridParticleList<MemorySpace, 16> particle_list("particles");
    using particle_type = typename Particle::GridParticleList<MemorySpace, 16>::particle_type;
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


        auto complex_boundary = Boundary::makeComplexBoundary(boundary1, boundary2, boundary3);
        REQUIRE(complex_boundary.num_boundary == 3);
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
    }
}