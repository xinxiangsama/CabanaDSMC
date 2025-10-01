#pragma once
#include "../simulation/initialize.hpp"
#include "../particle.hpp"
#include "../mesh/mesh.hpp"
#include "../../run/user.hpp"
#include  "../boundary/boundary.hpp"
#include "../geometry/geo.hpp"
#include <yaml-cpp/yaml.h>
namespace CabanaDSMC{
namespace Input{

struct SimulationConfig{

    static constexpr int dim = UserSpecfic::dim;
    using scalar_type = UserSpecfic::scalar_type;
    using exec_space = UserSpecfic::exec_space;
    using memory_space = UserSpecfic::memory_space;
    using h_memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using h_species_list_type = Particle::SpeciesList<h_memory_space>;
    using boundary_config_type = Boundary::BoundaryConfig<scalar_type>;
    using h_stl_type = Geometry::Stl<h_memory_space, scalar_type, dim>;

    FieldInitData<scalar_type, dim> initial;
    Mesh::MeshConfig<scalar_type, dim> mesh_config;
    std::array<boundary_config_type, dim * 2> boundary_config;
    h_species_list_type species_list;
    h_stl_type h_stl;

    uint8_t seed;
    size_t steps;
    scalar_type dt;
};

class InputReader{
public:
    using h_stl_type = SimulationConfig::h_stl_type;
static SimulationConfig read(const std::string& filename);
static h_stl_type readStl(const std::string& filename);
};
}
}