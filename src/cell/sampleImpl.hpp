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
    // using scalar_type = typename CellDataType::array_type::value_type;
    using scalar_type = double;
    
    SamplerBase() = default;
    
 
    void accumulate(
        const particle_list_type& particle_list,
        const std::shared_ptr<cell_data_type>& cell_data,
        const species_list_type& species_list
    ) const
    {
        static_cast<const Derived*>(this)->accumulateImpl(particle_list, cell_data, species_list);
    }
    
 
    void finalize(
        const std::shared_ptr<cell_data_type>& cell_data,
        const uint32_t& average_steps 
    ) const
    {
        static_cast<const Derived*>(this)->finalizeImpl(cell_data, average_steps);
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
    using scalar_type = typename base_type::scalar_type;
    
    BasicSampler() = default;
    
    // sample is about grid level parallel, not particle level . because if it is, it must concern about data racing and should use atomic lock, this must reduce performace.
 
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
        auto fn_view = cell_data->Fn->view();
        auto volume_view = cell_data->Volume->view();
        //get local set of owned cell indices
        auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
        auto global_grid = local_grid->globalGrid();

        Cabana::Grid::grid_parallel_for(
            "accumulate cell micro property",
            execution_space {},
            owned_cells,
            KOKKOS_LAMBDA(const int i, const int j, const int k){
                auto particle_num = particle_num_view(i, j, k, 0);
                if(particle_num == 0) return;
                auto particle_offset_idx = offset_index_view(i, j, k, 0);
                auto fn  = fn_view(i, j, k, 0);
                auto volume = volume_view(i, j, k, 0);
                scalar_type V_sum [3] {};
                scalar_type E_sum {}; 
                scalar_type M_sum {};

                for(uint64_t i = 0; i < particle_num; ++i){
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

                    V_sum[0] += velocity[0];
                    V_sum[1] += velocity[1];
                    V_sum[2] += velocity[2];

                    E_sum += 0.5 * (
                            velocity[0] * velocity[0]
                          + velocity[1] * velocity[1]
                          + velocity[2]* velocity[2]
                        );

                    M_sum += species_list(species_id).mass * fn;
                }
                scalar_type E_avg = E_sum / particle_num;
                density_view(i, j, k, 0) += M_sum / volume;
                // velocity_view(i, j, k, 0) += V_sum[0] / particle_num;
                scalar_type V_avg[3];
                for(int d = 0 ; d < 3; ++d){
                    V_avg[d] = V_sum[d] / particle_num;
                    velocity_view(i, j, k, d) += V_avg[d];
                }
                constexpr double kB = 1.380649e-23; // m2 kg s-2 K-1
                scalar_type V_avg_sq = 0.5 * (V_avg[0] * V_avg[0]
                                            + V_avg[1] * V_avg[1]
                                            + V_avg[2] * V_avg[2]);
                temperature_view(i, j, k, 0) += species_list(0).mass * (2.0/3.0) * (E_avg - V_avg_sq) / kB;

            }
        );
    }
    
 
    void finalizeImpl(
        const std::shared_ptr<cell_data_type>& cell_data,
        const uint32_t& average_steps
    ) const
    {
        if (average_steps == 0) return;
        
        auto local_grid = cell_data->localgrid();
        auto velocity_view = cell_data->Velocity->view();
        auto density_view = cell_data->Density->view();
        auto temperature_view = cell_data->Temperature->view();
        auto pressure_view = cell_data->Pressure->view();
        
        auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
        
        scalar_type inv_steps = 1.0 / static_cast<scalar_type>(average_steps);
        
        Cabana::Grid::grid_parallel_for(
            "finalize cell micro property averages",
            execution_space {},
            owned_cells,
            KOKKOS_LAMBDA(const int i, const int j, const int k){
                // Average the accumulated values
                for(int d = 0; d < 3; ++d) {
                    velocity_view(i, j, k, d) *= inv_steps;
                }
                density_view(i, j, k, 0) *= inv_steps;
                temperature_view(i, j, k, 0) *= inv_steps;
                pressure_view(i, j, k, 0) *= inv_steps;
            }
        );
    }
};

}
}
