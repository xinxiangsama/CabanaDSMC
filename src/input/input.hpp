#pragma once
#include "../simulation/initialize.hpp"
#include "../particle.hpp"
#include "../../run/user.hpp"
#include "yaml/Yaml.hpp"
namespace CabanaDSMC{
namespace Input{

struct SimulationConfig{

    static constexpr int dim = UserSpecfic::dim;
    using scalar_type = UserSpecfic::scalar_type;
    using exec_space = UserSpecfic::exec_space;
    using memory_space = UserSpecfic::memory_space;
    using h_meory_space = Kokkos::DefaultHostExecutionSpace::memory_space;

    FieldInitData<scalar_type, dim> initial;
    Particle::SpeciesList<h_meory_space> species_list;
};

class InputReader{
public:
static SimulationConfig read(const std::string& filename);

};
}
}