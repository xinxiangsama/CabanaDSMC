#pragma once
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include "boundaryImpl.hpp"
#include "../../run/user.hpp"

namespace CabanaDSMC{
namespace Boundary{



// fot manager all the boundary(complex) in the simulation field
template <class MemorySpace, class MeshType = Cabana::Grid::UniformMesh<double, 3>, uint16_t NumBoundary = 6>
class BoundaryManager{
public:
using memory_space = MemorySpace;
using mesh_type = MeshType;
};


// combination of some typical boundary conditions
template<class ... BoundaryTypes>
requires (is_boundary<BoundaryTypes>::value && ...)
struct ComplexBoundary
{
    static constexpr uint16_t num_boundary = sizeof...(BoundaryTypes);
    using tuple_type = std::tuple<BoundaryTypes...>;

    ComplexBoundary(const tuple_type& boundaries)
        : _boundaries(boundaries)
    {}

    template <class ParticleType>
    KOKKOS_INLINE_FUNCTION
    void apply(ParticleType& particle) const
    {
        // applyImpl(particle, std::make_index_sequence<num_boundary>{});
        std::apply([&](const auto&... boundary) {
            (boundary.apply(particle), ...);
        }, _boundaries);
    }


    tuple_type _boundaries;
};

template<class... BoundaryTypes>
auto makeComplexBoundary(const BoundaryTypes &... boundaries)
{
    return ComplexBoundary<BoundaryTypes...>(std::make_tuple(boundaries...));
}


template<class Scalar, class ParticleType, class MeshType>
UserSpecfic::boundaryVariant_t createBoundary(const BoundaryConfig<Scalar>& cfg,
                    const std::shared_ptr<Cabana::Grid::GlobalMesh<MeshType>>& global_mesh)
{
    switch (cfg.boundary_type)
    {
        case BoundaryType::Periodic:
        {
            return BoundaryFactory<PeriodicBoundary<Scalar, ParticleType>, MeshType>::create(
                cfg.position, cfg.normal, global_mesh);
        }

        case BoundaryType::Wall:
        {
            return BoundaryFactory<WallBoundary<Scalar, ParticleType>, MeshType>::create(
                cfg.position, cfg.normal, global_mesh,
                -std::numeric_limits<Scalar>::infinity(),
                 std::numeric_limits<Scalar>::infinity(),
                 cfg.temperature);
        }

            // TODO: 以后可以加 MaxwellWall, Inflow, Outflow
        default:
            throw std::runtime_error("Unsupported boundary type in createBoundary()");
    }
}
}


}
