#pragma once
#include "../particle.hpp"

namespace CabanaDSMC{
template<class ExecutionSpace, class ParticleListType, class LocalGridType, class CellDataType, class CollideType>
requires (Cabana::is_particle_list<ParticleListType>::value ||
            Cabana::Grid::is_particle_list<ParticleListType>::value)
void collide(
    const ExecutionSpace& exec_space, ParticleListType& particle_list,
    const std::shared_ptr<LocalGridType>& local_grid, 
    const std::shared_ptr<CellDataType>& cell_data,
    CollideType& collide_model
)
{

}
}