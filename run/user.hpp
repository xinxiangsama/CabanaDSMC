#pragma once
#include <cstddef>
#include <Kokkos_Core.hpp>
#include <Cabana_Grid.hpp>
#include "../src/particle.hpp"
#include "../src/boundary/boundaryImpl.hpp"
#include "../src/simulation/initialize.hpp"
#include "../src/collide/collideImpl.hpp"
namespace CabanaDSMC{
namespace UserSpecfic{

constexpr size_t dim = 3;

using scalar_type = double;

using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = exec_space::memory_space;

using mesh_type = Cabana::Grid::UniformMesh<scalar_type, dim>;

//===============================
//  decltype
//===============================

//----------------------------
// mesh
//----------------------------
using mesh_factory_type = Mesh::GlobalMeshFactory<mesh_type>;
//----------------------------
// particle
//----------------------------
using particle_list_type = Particle::GridParticleList<memory_space>;
using particle_type = typename particle_list_type::particle_type;
using particle_factory_t = ParticleFactory<particle_type>;
//----------------------------
// boundary
//----------------------------
using periodicBoundary_t = Boundary::PeriodicBoundary<scalar_type, particle_type>;
using wallBoundary_t = Boundary::WallBoundary<scalar_type, particle_type>;

using boundaryVariant_t = std::variant<std::monostate, wallBoundary_t, periodicBoundary_t>;
using boundaryContainer_t = std::array<boundaryVariant_t, dim * 2>;

using periodicBoundaryFactory_t = Boundary::BoundaryFactory<periodicBoundary_t, mesh_type>;
using wallBoundaryFactory_t = Boundary::BoundaryFactory<wallBoundary_t, mesh_type>;

using xminBoundary_t = periodicBoundary_t;
using xmaxBoundary_t = periodicBoundary_t;
using yminBoundary_t = periodicBoundary_t;
using ymaxBoundary_t = periodicBoundary_t;
using zminBoundary_t = periodicBoundary_t;
using zmaxBoundary_t = periodicBoundary_t;

//----------------------------
// collision model
//----------------------------
using collisionModel = VHSCollision<exec_space, scalar_type, particle_type>;
}
}