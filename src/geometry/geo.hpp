#pragma once
#include <Cabana_Core.hpp>

#include "../../run/user.hpp"

namespace CabanaDSMC{
namespace Geometry {

//--------------------------
// point
//--------------------------
template<class Scalar, size_t Dim>
struct Point {
    using scalar_type = Scalar;
    static constexpr size_t dim = Dim;
    std::array<scalar_type, Dim> coords;

    KOKKOS_INLINE_FUNCTION
    scalar_type& x() requires (Dim >= 1) {
        return coords[0];
    }
    const scalar_type& x() const requires (Dim >= 1) {
        return coords[0];
    }
    KOKKOS_INLINE_FUNCTION
    scalar_type& y() requires (Dim >= 2) {
        return coords[1];
    }

    KOKKOS_INLINE_FUNCTION
    const scalar_type& y() const requires (Dim >= 2) {
        return coords[1];
    }

    KOKKOS_INLINE_FUNCTION
    scalar_type& z() requires (Dim >= 3) {
        return coords[2];
    }

    KOKKOS_INLINE_FUNCTION
    const scalar_type& z() const requires (Dim >= 3) {
        return coords[2];
    }
};


template<class Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
Point<Scalar, Dim> operator-(const Point<Scalar, Dim>& a, const Point<Scalar, Dim>& b) {
    Point<Scalar, Dim> result;
    for (size_t i = 0; i < Dim; ++i) {
        result.coords[i] = a.coords[i] - b.coords[i];
    }
    return result;
}

template<class Scalar>
KOKKOS_INLINE_FUNCTION
Point<Scalar, 3> cross(const Point<Scalar, 3>& a, const Point<Scalar, 3>& b) {
    Point<Scalar, 3> result;
    result.x() = a.y() * b.z() - a.z() * b.y();
    result.y() = a.z() * b.x() - a.x() * b.z();
    result.z() = a.x() * b.y() - a.y() * b.x();
    return result;
}

template<class Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
Scalar dot(const Point<Scalar, Dim>& a, const Point<Scalar, Dim>& b) {
    Scalar result = 0;
    for (size_t i = 0; i < Dim; ++i) {
        result += a.coords[i] * b.coords[i];
    }
    return result;
}

//--------------------------
// triangle in 3d (or segment in 2d, actually it represent discrete element)
//--------------------------
template<class Scalar, size_t Dim>
requires (Dim > 1)
struct Triangle {
    using scalar_type = Scalar;
    static constexpr size_t dim = Dim;
    using point_type = Point<Scalar, dim>;

    std::array<point_type, dim> points;
    std::array<scalar_type, dim> normal;
};

//--------------------------
// segment (just segment)
//--------------------------
template<class Scalar, size_t Dim>
requires (Dim> 1)
struct Segment {
    using scalar_type = Scalar;
    static constexpr size_t dim = Dim;
    using point_type = Point<Scalar, Dim>;

    std::array<point_type, 2> points;

};

// ------------------------
// STL geometry object
// -------------------------
template<class MemorySpace, class Scalar, size_t Dim>
using Stl = Kokkos::View<Triangle<Scalar, Dim>*, MemorySpace>;

// ========================================================
//  some helpful geometry function
// =========================================================
namespace Utilities {

// -----------------------------------------
// segment intersect with segment
// -------------------------------------------
template<class Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
std::optional<Point<Scalar, Dim>> segmentIntersectWithTriangle(
    const Segment<Scalar, Dim>& segment,
    const Triangle<Scalar, Dim>& triangle) requires(Dim == 2)
{
    using scalar_type = Scalar;
    const auto& p1 = segment.points[0];
    const auto& p2 = segment.points[1];
    const auto& p3 = triangle.points[0];
    const auto& p4 = triangle.points[1];

    const scalar_type v1x = p2.x() - p1.x();
    const scalar_type v1y = p2.y() - p1.y();
    const scalar_type v2x = p4.x() - p3.x();
    const scalar_type v2y = p4.y() - p3.y();

    const scalar_type denominator = v1x * v2y - v1y * v2x;

    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();
    if (std::abs(denominator) < epsilon) {
        // The segments are parallel or collinear. No unique intersection exists.
        return std::nullopt;
    }

    const scalar_type deltaPx = p1.x() - p3.x();
    const scalar_type deltaPy = p1.y() - p3.y();

    // Solve for parameters t and u for the parametric equations of the lines.
    const scalar_type t = (deltaPy * v2x - deltaPx * v2y) / denominator;
    const scalar_type u = (deltaPy * v1x - deltaPx * v1y) / denominator;

    // If t and u are both between 0 and 1, the intersection point
    // lies within the bounds of both line segments.
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        Point<Scalar, Dim> intersection_point;
        intersection_point.x() = p1.x() + t * v1x;
        intersection_point.y() = p1.y() + t * v1y;

        // Return the optional containing the calculated point.
        return intersection_point;
    }

    // The lines intersect, but not within the segments' bounds.
    return std::nullopt;
}


// -----------------------------------------
// segment intersect with triangle
// -------------------------------------------
template<class Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
std::optional<Point<Scalar, Dim>> segmentIntersectWithTriangle(
    const Segment<Scalar, Dim>& segment,
    const Triangle<Scalar, Dim>& triangle) requires(Dim == 3)
{
    using point_type = Point<Scalar, Dim>;
    constexpr auto epsilon = std::numeric_limits<Scalar>::epsilon();

    const point_type& p0 = segment.points[0];
    const point_type& p1 = segment.points[1];
    const point_type dir = p1 - p0;

    const point_type& v0 = triangle.points[0];
    const point_type& v1 = triangle.points[1];
    const point_type& v2 = triangle.points[2];

    const point_type edge1 = v1 - v0;
    const point_type edge2 = v2 - v0;

    const point_type pvec = cross(dir, edge2);
    const Scalar det = dot(edge1, pvec);

    // If the determinant is near zero, the segment is parallel to the triangle plane.
    if (std::abs(det) < epsilon) {
        return std::nullopt;
    }

    const Scalar inv_det = static_cast<Scalar>(1.0) / det;
    const point_type tvec = p0 - v0;

    // Calculate the U barycentric coordinate and test its bounds.
    const Scalar u = dot(tvec, pvec) * inv_det;
    if (u < 0 || u > 1) {
        return std::nullopt;
    }

    const point_type qvec = cross(tvec, edge1);

    // Calculate the V barycentric coordinate and test its bounds.
    const Scalar v = dot(dir, qvec) * inv_det;
    if (v < 0 || u + v > 1) {
        return std::nullopt;
    }

    // Calculate t, the parameter for the intersection point on the segment.
    const Scalar t = dot(edge2, qvec) * inv_det;

    // Check if the intersection is within the segment's bounds [0, 1].
    if (t > epsilon && t < 1 + epsilon) {
        point_type intersection_point;
        intersection_point.x() = p0.x() + t * dir.x();
        intersection_point.y() = p0.y() + t * dir.y();
        intersection_point.z() = p0.z() + t * dir.z();
        return intersection_point;
    }

    return std::nullopt;
}
}
}
}
