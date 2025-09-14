#pragma once
#include "../particle.hpp"
namespace CabanaDSMC{

template<class ExecutionSpace = Kokkos::DefaultExecutionSpace,
            typename MemorySpace = typename ExecutionSpace::memory_space>
class Run{
public:
    Run() = default;
protected:
    Particle::GridParticleList<MemorySpace, 16> m_particles {"particles"};
};

}