#pragma once
#include <Cabana_Grid.hpp>
#include "../particle.hpp"
namespace CabanaDSMC{
namespace Boundary{
template<class Derived, class Scalar, class ParticleType>
class BoudnaryBase{
public:
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    BoudnaryBase() = default;
    BoudnaryBase(const scalar_type& position, const scalar_type normal[3], const scalar_type domain_extent[3], const scalar_type& start, const scalar_type& end)
        : _position(position), _start(start), _end(end)
    {
        _normal[0] = normal[0];
        _normal[1] = normal[1];
        _normal[2] = normal[2];

        _domain_extent[0] = domain_extent[0];
        _domain_extent[1] = domain_extent[1];
        _domain_extent[2] = domain_extent[2];
    }

    //don't need virtual destructor

    KOKKOS_INLINE_FUNCTION
    void apply(particle_type& particle, const scalar_type time_step) const
    {
        static_cast<const Derived*>(this)->applyImpl(particle, time_step);
    }

    KOKKOS_INLINE_FUNCTION
    bool checkIfHitBoundary(const particle_type& particle) const
    {
        return static_cast<const Derived*>(this)->checkIfHitBoundaryImpl(particle);
    }

protected:
    scalar_type _position; // the position of the boundary. ie 0, L1 , L2, L3.....
    scalar_type _normal [3]; // the normal direction of the boundary
    scalar_type _start;
    scalar_type _end;
    scalar_type _domain_extent [3]; //domian extent in each direction
};  
template <class Scalar, class ParticleType>
class PeriodicBoundary : public BoudnaryBase<PeriodicBoundary<Scalar, ParticleType>, Scalar, ParticleType>{
public:
    using base_type = BoudnaryBase<PeriodicBoundary<Scalar, ParticleType>, Scalar, ParticleType>;
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    PeriodicBoundary(const scalar_type& position, const scalar_type normal[3], const scalar_type domain_extent[3], const scalar_type& start, const scalar_type& end)
        : base_type(position, normal, domain_extent, start, end)
    {}

    KOKKOS_INLINE_FUNCTION
    void applyImpl(particle_type& particle, const scalar_type& time_step) const
    {   
        if(checkIfHitBoundaryImpl(particle))
        {
            
            for (int d = 0; d < 3; ++d)
            {   
                auto& p_d = Cabana::get(particle, Particle::Field::Position(), d);
                if (this->_normal[d] > 0) // positive direction
                {
                    if (p_d < this->_position)
                        p_d += this->_domain_extent[d];
                }
                else if (this->_normal[d] < 0) // negative direction
                {
                    if (p_d > this->_position)
                        p_d -= this->_domain_extent[d];
                }
            }
        }
    }

    
    KOKKOS_INLINE_FUNCTION
    bool checkIfHitBoundaryImpl(const particle_type& particle) const
    {
        for (int d = 0; d < 3; ++d)
        {   
            auto& p_d = Cabana::get(particle, Particle::Field::Position(), d);
            if (this->_normal[d] > 0) // positive direction
            {
                if (p_d < this->_position)
                    return true;
            }
            else if (this->_normal[d] < 0) // negative direction
            {
                if (p_d > this->_position)
                    return true;
            }
        }
        return false;
    }
};


template <class Scalar, class ParticleType>
class WallBoundary : public BoudnaryBase<PeriodicBoundary<Scalar, ParticleType>, Scalar, ParticleType>{
public:
    using base_type = BoudnaryBase<PeriodicBoundary<Scalar, ParticleType>, Scalar, ParticleType>;
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    WallBoundary(const scalar_type& position, const scalar_type normal[3], const scalar_type domain_extent[3], const scalar_type& start, const scalar_type& end, const scalar_type& temperature) : base_type(position, normal, domain_extent, start, end), _temperature(temperature)
    {}
private:
    scalar_type _temperature;

};

// other boundary types to be implemented

// sfinae for boundary types
template <class T>
struct is_boundary : public std::false_type
{
};

template <class Scalar, class ParticleType>
struct is_boundary<PeriodicBoundary<Scalar, ParticleType>> : public std::true_type
{
};

template <class Scalar, class ParticleType>
struct is_boundary<WallBoundary<Scalar, ParticleType>> : public std::true_type
{
};



// boundary factory

template <class BoundaryType, class MeshType>
struct BoundaryFactory
{
};

template <class Scalar, class ParticleType, class MeshType>
requires is_boundary<PeriodicBoundary<Scalar, ParticleType>>::value
struct BoundaryFactory<PeriodicBoundary<Scalar, ParticleType>, MeshType>
{
    using boundary_type = PeriodicBoundary<Scalar, ParticleType>;
    using scalar_type = Scalar;
    using particle_type = ParticleType;
    using global_mesh_type = Cabana::Grid::GlobalMesh<MeshType>;
    static boundary_type create(
        const Scalar& position, const Scalar normal[3], const std::shared_ptr<global_mesh_type>& global_mesh, const Scalar& start = -std::numeric_limits<Scalar>::infinity(), const Scalar& end = std::numeric_limits<Scalar>::infinity())
    {   
        Scalar domain_extent[3];
        domain_extent[0] = global_mesh->extent(0);
        domain_extent[1] = global_mesh->extent(1);
        domain_extent[2] = global_mesh->extent(2);
        return boundary_type(position, normal, domain_extent, start, end);
    }
};
}
}