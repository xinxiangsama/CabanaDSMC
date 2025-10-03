#pragma once
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
namespace CabanaDSMC{
namespace Geometry {

//--------------------------
// point
//--------------------------
template<class Scalar, size_t Dim>
struct Point {
    using scalar_type = Scalar;
    static constexpr size_t dim = Dim;
    scalar_type coords [dim];

    KOKKOS_INLINE_FUNCTION
    Point()
    {
        // Explicitly initialize all coordinates to zero.
        for (size_t i = 0; i < Dim; ++i) {
            coords[i] = 0.0;
        }
    }

    template<class... Args>
    KOKKOS_INLINE_FUNCTION
    Point(Args... args) requires (sizeof...(Args) == Dim)
        : coords{ static_cast<scalar_type>(args)... } {}

    KOKKOS_INLINE_FUNCTION
    scalar_type& x() requires (Dim >= 1) {
        return coords[0];
    }

    KOKKOS_INLINE_FUNCTION
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
    KOKKOS_INLINE_FUNCTION
    const point_type* vertices() const {
        return _vertices;
    }
    KOKKOS_INLINE_FUNCTION
    point_type* vertices() {
        return _vertices;
    }
    KOKKOS_INLINE_FUNCTION
    auto size() const {
        return dim;
    }

    point_type _vertices [dim];
    point_type _normal;
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

    point_type _vertices [2]; 

    KOKKOS_INLINE_FUNCTION
    const point_type* vertices() const {
        return _vertices;
    }
    KOKKOS_INLINE_FUNCTION
    point_type* vertices() {
        return _vertices;
    }

    KOKKOS_INLINE_FUNCTION
    auto size() const {
        return 2;
    }
};

//--------------------------
// Square (as a 2D or 3D Quad)
//--------------------------
template<class Scalar, size_t Dim>
requires (Dim == 2 || Dim == 3)
struct Square {
    using scalar_type = Scalar;
    static constexpr size_t dim = Dim;
    using point_type = Point<Scalar, Dim>;
    using segment_type = Segment<Scalar, Dim>;

    point_type _vertices [4];

    KOKKOS_INLINE_FUNCTION
    const point_type* vertices() const {
        return _vertices;
    }

    KOKKOS_INLINE_FUNCTION
    point_type* vertices() {
        return _vertices;
    }

    KOKKOS_INLINE_FUNCTION
    segment_type edge(int idx) const
    {
        segment_type s;
        s.vertices[0] = _vertices[idx];
        s.vertices[1] = _vertices[(idx + 1) % 4];
        return s;
    }

    KOKKOS_INLINE_FUNCTION
    point_type center() const
    {
        point_type c;
        for (size_t d = 0; d < dim; ++d)
        {
            c.coords[d] = (_vertices[0].coords[d] + _vertices[1].coords[d] +
                           _vertices[2].coords[d] + _vertices[3].coords[d]) * 0.25;
        }
        return c;
    }

    KOKKOS_INLINE_FUNCTION
    auto size() const {
        return 4;
    }

};
//--------------------------
// Cube (3D only)
//--------------------------
template<class Scalar>
struct Cube {
    using scalar_type = Scalar;
    static constexpr size_t dim = 3;
    using point_type = Point<Scalar, dim>;
    using square_type = Square<Scalar, dim>;


    point_type _vertices [8];
    point_type _center;
    scalar_type _side_length;

    KOKKOS_INLINE_FUNCTION
    Cube(const point_type& center, scalar_type side_length) : _center(center), _side_length(side_length)
    {
        const scalar_type half_side = side_length / 2.0;
        // build order: first bottom then top
        // Z= -half_side
        _vertices[0] = {center.x() - half_side, center.y() - half_side, center.z() - half_side};
        _vertices[1] = {center.x() + half_side, center.y() - half_side, center.z() - half_side};
        _vertices[2] = {center.x() + half_side, center.y() + half_side, center.z() - half_side};
        _vertices[3] = {center.x() - half_side, center.y() + half_side, center.z() - half_side};
        // Z= +half_side
        _vertices[4] = {center.x() - half_side, center.y() - half_side, center.z() + half_side};
        _vertices[5] = {center.x() + half_side, center.y() - half_side, center.z() + half_side};
        _vertices[6] = {center.x() + half_side, center.y() + half_side, center.z() + half_side};
        _vertices[7] = {center.x() - half_side, center.y() + half_side, center.z() + half_side};
    }

    KOKKOS_INLINE_FUNCTION
    const point_type* vertices() const {
        return _vertices;
    }

    KOKKOS_INLINE_FUNCTION
    point_type* vertices() {
        return _vertices;
    }

    KOKKOS_INLINE_FUNCTION
    auto center() const {
        return _center;
    }


    KOKKOS_INLINE_FUNCTION
    auto size() const {
        return 8;
    }

    KOKKOS_INLINE_FUNCTION
    auto sideLength() const {
        return _side_length;
    }
};
//--------------------------
// Bounding Box
//--------------------------
template<class Scalar, size_t Dim>
struct BoundingBox {
    Point<Scalar, Dim> min_corner;
    Point<Scalar, Dim> max_corner;
};

//--------------------------
// a piece
//--------------------------
template<class Scalar>
struct Interval {
    Scalar min;
    Scalar max;
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
// compute bounding box
// -------------------------------------------
template<class Scalar, size_t Dim, class Entitytype>
KOKKOS_INLINE_FUNCTION
BoundingBox<Scalar, Dim>
getBoundingBox(const Entitytype& entity)
requires requires(const Entitytype& e) {
e.vertices();
}
{
    BoundingBox<Scalar, Dim> bbox;
    for (size_t d = 0; d < Dim; ++d) {
        bbox.min_corner.coords[d] = std::numeric_limits<Scalar>::max();
        bbox.max_corner.coords[d] = std::numeric_limits<Scalar>::lowest();
    }
    const auto& vertices = entity.vertices();
    const auto& size = entity.size();
    for (size_t i = 0; i < size; ++i)
    {
        const auto& vertex = vertices[i];

        for (size_t d = 0; d < Dim; ++d)
        {
            bbox.min_corner.coords[d] = Kokkos::min(bbox.min_corner.coords[d], vertex.coords[d]);
            bbox.max_corner.coords[d] = Kokkos::max(bbox.max_corner.coords[d], vertex.coords[d]);
        }
    }

    return bbox;
}
// -----------------------------------------
// check bounding box intersection
// -------------------------------------------
template<class Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
bool checkBoundingBoxIntersection(const BoundingBox<Scalar, Dim>& bbox_a,
                     const BoundingBox<Scalar, Dim>& bbox_b)
{
    for (size_t d = 0; d < Dim; ++d)
    {
        if (bbox_a.max_corner.coords[d] < bbox_b.min_corner.coords[d] ||
            bbox_b.max_corner.coords[d] < bbox_a.min_corner.coords[d])
        {
            return false;
        }
    }
    return true;
}


//----------------------------------------------
// project one set of point to a axis
//----------------------------------------------
template<class Scalar, size_t Dim, class EntityType>
KOKKOS_INLINE_FUNCTION
Interval<Scalar> getInterval(const EntityType& entity,
                           const Point<Scalar, Dim>& axis)
{
    Interval<Scalar> interval;
    interval.min = std::numeric_limits<Scalar>::max();
    interval.max = std::numeric_limits<Scalar>::lowest();

    const auto& vertices = entity.vertices();
    for (size_t i = 0; i < entity.size(); ++i)
    {
        const Scalar proj = dot(vertices[i], axis);
        interval.min = Kokkos::min(interval.min, proj);
        interval.max = Kokkos::max(interval.max, proj);
    }
    return interval;
}

//----------------------------------------------
// project a cube to a axis
//----------------------------------------------
template<class Scalar>
KOKKOS_INLINE_FUNCTION
Interval<Scalar> getInterval(const Cube<Scalar>& cube,
                           const Point<Scalar, 3>& axis)
{
    const auto center = cube.center();
    // 直接使用存储的边长来计算半边长
    const Scalar half_side = cube.sideLength() * 0.5;
    
    // 投影中心点
    const Scalar c = dot(center, axis);
    
    // 投影半径是三个半边长在轴上投影的绝对值之和
    const Scalar r = half_side * (Kokkos::abs(axis.x()) +
                                  Kokkos::abs(axis.y()) +
                                  Kokkos::abs(axis.z()));

    return {c - r, c + r};
}
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
    const auto& p1 = segment.vertices()[0];
    const auto& p2 = segment.vertices()[1];
    const auto& p3 = triangle.vertices()[0];
    const auto& p4 = triangle.vertices()[1];

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

    const point_type& p0 = segment.vertices()[0];
    const point_type& p1 = segment.vertices()[1];
    const point_type dir = p1 - p0;

    const point_type& v0 = triangle.vertices()[0];
    const point_type& v1 = triangle.vertices()[1];
    const point_type& v2 = triangle.vertices()[2];

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
