#pragma once
#include "../particle.hpp"
#include <Kokkos_Core.hpp>
namespace CabanaDSMC{

/*
this module has two main functions:
1. compute collision rate for each collision pair
2. perform collision for each collision pair
*/

template <class Derived, class ExecutionSpace, class Scalar, class ParticleType>
class CollisionModelBase{
public:
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    using species_list_type = Particle::SpeciesList<typename ExecutionSpace::memory_space>;
    using generator_type = Kokkos::Random_XorShift64<ExecutionSpace>;
    CollisionModelBase() = default;
    //don't need virtual destructor

    KOKKOS_INLINE_FUNCTION
    scalar_type computeCollisionRate(
        const particle_type& particle1, 
        const particle_type& particle2,
        const species_list_type& species_list
    ) const
    {
        return static_cast<const Derived*>(this)->computeCollisionRateImpl(particle1, particle2, species_list);
    }

    KOKKOS_INLINE_FUNCTION
    void collide(particle_type& particle1, particle_type& particle2, generator_type& rng) const
    {
        static_cast<const Derived*>(this)->collideImpl(particle1, particle2, rng);
    }
};


/*
Hard Sphere Collision Model
*/
template <class ExecutionSpace, class Scalar, class ParticleType>
class HardSphereCollision : public CollisionModelBase<HardSphereCollision<ExecutionSpace, Scalar, ParticleType>, ExecutionSpace, Scalar, ParticleType>{
public:
    using base_type = CollisionModelBase<HardSphereCollision<ExecutionSpace, Scalar, ParticleType>, ExecutionSpace, Scalar, ParticleType>;
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    using species_list_type = Particle::SpeciesList<typename ExecutionSpace::memory_space>;
    using generator_type = Kokkos::Random_XorShift64<ExecutionSpace>;
    HardSphereCollision() = default;

    KOKKOS_INLINE_FUNCTION
    scalar_type computeCollisionRateImpl(
        const particle_type& particle1, 
        const particle_type& particle2,
        const species_list_type& species_list
    ) const
    {
        // get species id
        auto species_id1 = Cabana::get(particle1, Particle::Field::SpeciesID{});
        auto species_id2 = Cabana::get(particle2, Particle::Field::SpeciesID{});

        // get species data
        auto species1 = species_list(species_id1);
        auto species2 = species_list(species_id2);

        // get relative velocity
        scalar_type rel_vel[3];
        for(int d = 0; d < 3; ++d){
            rel_vel[d] = Cabana::get(particle1, Particle::Field::Velocity{}, d) - Cabana::get(particle2, Particle::Field::Velocity{}, d);
        }
        scalar_type rel_speed = sqrt(rel_vel[0]*rel_vel[0] + rel_vel[1]*rel_vel[1] + rel_vel[2]*rel_vel[2]);

        // compute collision rate
        scalar_type d_avg = 0.5 * (species1.diameter + species2.diameter);
        scalar_type collision_rate = M_PI * d_avg * d_avg * rel_speed;

        return collision_rate;
    }

    KOKKOS_INLINE_FUNCTION
    void collideImpl(particle_type& particle1, particle_type& particle2, generator_type& rng) const
    {
        scalar_type ave_vel [3];
        scalar_type rel_vel[3];
        for(int d = 0; d < 3; ++d){
            ave_vel[d] = 0.5 * (Cabana::get(particle1, Particle::Field::Velocity{}, d) + Cabana::get(particle2, Particle::Field::Velocity{}, d));
            rel_vel[d] = Cabana::get(particle1, Particle::Field::Velocity{}, d) - Cabana::get(particle2, Particle::Field::Velocity{}, d);
        }
        scalar_type rel_speed = sqrt(rel_vel[0]*rel_vel[0] + rel_vel[1]*rel_vel[1] + rel_vel[2]*rel_vel[2]);

        auto cos_r = Kokkos::rand<decltype(rng), double>::draw(rng, 0.0, 1.0);
        auto sin_r = Kokkos::sqrt(1 - cos_r * cos_r);
        auto phi = 2 * M_PI * Kokkos::rand<decltype(rng), double>::draw(rng, 0.0, 1.0);

        scalar_type rel_vel_new [3];
        rel_vel_new[0] = rel_speed * sin_r * Kokkos::cos(phi);
        rel_vel_new[1] = rel_speed * sin_r * Kokkos::sin(phi);
        rel_vel_new[2] = rel_speed * cos_r;

        for(int d = 0; d < 3; ++d){
            Cabana::get(particle1, Particle::Field::Velocity(), d) = ave_vel[d] + rel_vel_new[d];
            Cabana::get(particle2, Particle::Field::Velocity(), d) = ave_vel[d] - rel_vel_new[d];
        }
    } 
};


}