#pragma once
#include "geo.hpp"
// =========================================================================
// Winding Number / Point in Polygon/Polyhedron
// =========================================================================
namespace CabanaDSMC {
namespace Geometry {
namespace Utilities {

//---------------------------
// compute solid angle (point to triangle )
//-----------------------------
template<typename Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
Scalar computeSolidAngle(const Point<Scalar, Dim>& point, const Triangle<Scalar, Dim>& triangle)
requires(Dim == 3)
{
    const auto& ver1 = triangle.vertices()[0];
    const auto& ver2 = triangle.vertices()[1];
    const auto& ver3 = triangle.vertices()[2];

    // compute the vector from point to three vertice
    const auto a = ver1 - point;
    const auto b = ver2 - point;
    const auto c = ver3 - point;

    // compute the module of three vector
    auto module_a = Kokkos::sqrt(dot(a, a));
    auto module_b = Kokkos::sqrt(dot(b, b));
    auto module_c = Kokkos::sqrt(dot(c, c));

    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    if (module_a < epsilon || module_b < epsilon || module_c < epsilon) {
        return 0.0; // point on the vertice , solid angle is zero
    }

    const Scalar triple_product = dot(a, cross(b, c));
    const Scalar denominator = (module_a * module_b * module_c) +
                               (dot(a, b) * module_c) +
                               (dot(a, c) * module_b) +
                               (dot(b, c) * module_a);

    return 2.0 * Kokkos::atan2(triple_product, denominator);
}

//---------------------------
// compute solid angle (point to Segment )
//-----------------------------
template<typename Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
Scalar computeSolidAngle(const Point<Scalar, Dim>& point, const Triangle<Scalar, Dim>& triangle)
requires(Dim == 2)
{
    const auto& ver1 = triangle.vertices()[0];
    const auto& ver2 = triangle.vertices()[1];

    // compute the vector from point to three vertice
    const auto a = ver1 - point;
    const auto b = ver2 - point;

    return Kokkos::atan2(a.x() * b.y() - a.y() * b.x(),
                         dot(a, b));
}

//--------------------------------------
// winding number test
//---------------------------------------
template<class MemorySpace, typename Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
bool winding_number_test(const Point<Scalar, Dim>& point, const Stl<MemorySpace, Scalar, Dim>& stl)
{
    using scalar_type = Scalar;
    scalar_type total_solid_angle {};

    for (size_t i = 0; i < stl.extent(0); ++i) {
        total_solid_angle += computeSolidAngle(point, stl(i));
    }
    if constexpr (Dim == 3) {
        constexpr scalar_type two_pi = 2.0 * std::numbers::pi_v<scalar_type>;
        return Kokkos::abs(total_solid_angle) > two_pi;
    }else if (Dim == 2){
        constexpr scalar_type pi = std::numbers::pi_v<scalar_type>;
        return Kokkos::abs(total_solid_angle) > pi;
    }else {
        std::__throw_bad_cast();
    }

}

}
}
}