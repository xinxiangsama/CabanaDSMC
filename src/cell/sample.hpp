#pragma once
#include "sampleImpl.hpp"


namespace CabanaDSMC{
namespace Cell{
template<class ParticleListType, class CellDataType, class... SampleTypes>
void sample(
    const uint32_t& current_step,
    const uint32_t& average_steps,
    ParticleListType& particle_list,
    const std::shared_ptr<CellDataType>& cell_data,
    const Particle::SpeciesList<typename ParticleListType::memory_space>& species_list,
    const SampleTypes&... samplings
)
{
    auto sample_utility= std::make_tuple(samplings...);

    // std::apply(
    //     [&](const auto&... sampling){
    //         (sampling.accumulate(particle_list, cell_data, species_list), ...);
    //     }, sample_utility);
    (samplings.accumulate(particle_list, cell_data, species_list), ...);
    
    if(current_step % average_steps == 0 && current_step !=0){
        std::apply(
            [&](const auto&... sampling){
                (sampling.finalize(cell_data, average_steps), ...);
            }, sample_utility);
    }
}
}
}