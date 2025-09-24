#pragma once

namespace CabanaDSMC{

/*
this module has two main functions:
1. compute collision rate for each collision pair
2. perform collision for each collision pair
*/

template <class Derived, class Scalar, class ParticleType>
class CollisionModelBase{
public:
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    CollisionModelBase() = default;
    //don't need virtual destructor

    KOKKOS_INLINE_FUNCTION
    template<class SpeciesListType>
    scalar_type computeCollisionRate(
        const particle_type& particle1, 
        const particle_type& particle2,
        const SpeciesListType& species_list
    ) const
    {
        return static_cast<const Derived*>(this)->computeCollisionRateImpl(particle1, particle2, species_list);
    }

    KOKKOS_INLINE_FUNCTION
    void collide(particle_type& particle1, particle_type& particle2) const
    {
        static_cast<const Derived*>(this)->collideImpl(particle1, particle2);
    }
};


/*
Hard Sphere Collision Model
*/
template <class Scalar, class ParticleType>
class HardSphereCollision : public CollisionModelBase<HardSphereCollision<Scalar, ParticleType>, Scalar, ParticleType>{
public:
    using base_type = CollisionModelBase<HardSphereCollision<Scalar, ParticleType>, Scalar, ParticleType>;
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    HardSphereCollision() = default;

    KOKKOS_INLINE_FUNCTION
    template<class SpeciesListType>
    scalar_type computeCollisionRateImpl(
        const particle_type& particle1, 
        const particle_type& particle2,
        const SpeciesListType& species_list
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
};


}