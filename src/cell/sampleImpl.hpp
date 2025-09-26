#pragma once
#include "cell.hpp"
#include "../particle.hpp"
#include <Kokkos_Core.hpp>
/*
use different methold to sample micro property
*/


namespace CabanaDSMC{
namespace Cell{
template<class Derived, class ExecutionSpace, class ParticleListType, class CellDataType>
class SamplerBase{
public:
    using execution_space = ExecutionSpace;
    using particle_list_type = ParticleListType;
    using cell_data_type = CellDataType;
    using species_list_type = Particle::SpeciesList<typename ExecutionSpace::memory_space>;
    using scalar_type = typename CellDataType::array_type::value_type;
    
    SamplerBase() = default;
    
    KOKKOS_INLINE_FUNCTION
    void accumulate(
        const particle_list_type& particle_list,
        const std::shared_ptr<cell_data_type>& cell_data,
        const species_list_type& species_list
    )
    {
        static_cast<Derived*>(this)->accumulateImpl(particle_list, cell_data, species_list);
    }
    
    KOKKOS_INLINE_FUNCTION
    void finalize(
        const std::shared_ptr<cell_data_type>& cell_data,
        const species_list_type& species_list
    )
    {
        static_cast<Derived*>(this)->finalizeImpl(cell_data, species_list);
    }
    
protected:
};

/*
Basic sampler that accumulates velocity temperature pressure density for each cell
*/
template<class ExecutionSpace, class ParticleListType, class CellDataType>
class BasicSampler : public SamplerBase<BasicSampler<ExecutionSpace, ParticleListType, CellDataType>, 
                                       ExecutionSpace, ParticleListType, CellDataType> {
public:
    using base_type = SamplerBase<BasicSampler<ExecutionSpace, ParticleListType, CellDataType>, 
                                 ExecutionSpace, ParticleListType, CellDataType>;
    using execution_space = ExecutionSpace;
    using particle_list_type = ParticleListType;
    using cell_data_type = CellDataType;
    using species_list_type = Particle::SpeciesList<typename ExecutionSpace::memory_space>;
    using scalar_type = typename CellDataType::array_type::value_type;
    
    BasicSampler() = default;
    
    // sample is about grid level parallel, not particle level . because if it is, it must concern about data racing and should use atomic lock, this must reduce performace.
    KOKKOS_INLINE_FUNCTION
    void accumulateImpl(
        const particle_list_type& particle_list,
        const std::shared_ptr<cell_data_type>& cell_data,
        const species_list_type& species_list
    ) const
    {
        // Get local grid and arrays
        auto local_grid = cell_data->localgrid();
        auto velocity_view = cell_data->Velocity->view();
        auto density_view = cell_data->Density->view();
        auto temperature_view = cell_data->Temperature->view();
        auto particle_num_view = cell_data->Num_particles->view();
        auto offset_index_view = cell_data->Offset_particle_idx->view();

        //get local set of owned cell indices
        auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
        auto global_grid = local_grid->globalGrid();

        Cabana::Grid::grid_parallel_for(
            "accumulate cell micro property",
            execution_space {},
            owned_cells,
            KOKKOS_LAMBDA(const int i, const int j, const int k){
                auto particle_num = particle_num_view(i, j, k, 0);
                auto particle_offset_idx = offset_index_view(i, j, k, 0);

                for(int i = 0; i < particle_num; ++i){
                    auto particle = particle_list.getParticle(particle_offset_idx + i);
                    scalar_type velocity[3];
                    for(uint16_t d = 0; d < 3; ++d){
                        velocity[d] = Cabana::get(particle, Particle::Field::Velocity(), d);
                    }
                    auto is_active = Cabana::get(particle, Particle::Field::IsActive());
                    auto e_rot = Cabana::get(particle, Particle::Field::RotEnergy());
                    auto e_vib = Cabana::get(particle, Particle::Field::VibEnergy());
                    auto species_id = Cabana::get(particle, Particle::Field::SpeciesID());
                    if(!is_active) continue;
                }
            }
        );
    }
    
    KOKKOS_INLINE_FUNCTION
    void finalizeImpl(
        const std::shared_ptr<cell_data_type>& cell_data,
        const species_list_type& species_list
    ) const
    {
    }
};

}
}
