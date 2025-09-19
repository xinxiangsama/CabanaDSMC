#pragma once
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include "boundaryImpl.hpp"
namespace CabanaDSMC{
namespace Boundary{

enum class BoundaryType{
    Inflow,
    Outflow,
    Wall,
    MaxwellWall,
    Periodic
};

// fot manager all the boundary(complex) in the simulation field
template <class MemorySpace, class MeshType = Cabana::Grid::UniformMesh<double, 3>, uint16_t NumBoundary = 6>
class BoundaryManager{
public:
using memory_space = MemorySpace;
using mesh_type = MeshType;
};


// combination of some typical boundary conditions
template<class ... BoundaryTypes>
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
            // 对解包出来的每一个 boundary 对象调用 .apply()
            (boundary.apply(particle, 0.0), ...);
        }, _boundaries);
    }

    // template <class ParticleType, std::size_t... Is>
    // KOKKOS_INLINE_FUNCTION
    // void applyImpl(ParticleType& particle, std::index_sequence<Is...>) const
    // {
    //     ( std::get<Is>(_boundaries).apply(particle, 0.0), ... );
    // }

    tuple_type _boundaries;
};

template<class... BoundaryTypes>
auto makeComplexBoundary(const BoundaryTypes &... boundaries)
{
    return ComplexBoundary<BoundaryTypes...>(std::make_tuple(boundaries...));
}
}


}