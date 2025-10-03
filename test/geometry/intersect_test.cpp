#include <catch2/catch_all.hpp>
#include "../../src/geometry/geo.hpp"

// Define aliases for convenience in the test file
using Point2D = CabanaDSMC::Geometry::Point<double, 2>;
using Segment2D = CabanaDSMC::Geometry::Segment<double, 2>;
using Triangle2D = CabanaDSMC::Geometry::Triangle<double, 2>;

using Point3D = CabanaDSMC::Geometry::Point<double, 3>;
using Segment3D = CabanaDSMC::Geometry::Segment<double, 3>;
using Triangle3D = CabanaDSMC::Geometry::Triangle<double, 3>;

TEST_CASE("2D Geometry Utilities", "[geometry]") {

    SECTION("Segment-Triangle Intersection in 2D") {

        // Test: Simple intersection
        SECTION("Simple intersection") {
            Segment2D seg;
            seg.vertices()[0] = {0.0, 0.0};
            seg.vertices()[1] = {2.0, 2.0};

            Triangle2D tri_as_seg;
            tri_as_seg.vertices()[0] = {0.0, 2.0};
            tri_as_seg.vertices()[1] = {2.0, 0.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(1.0));
            REQUIRE(result->y() == Catch::Approx(1.0));
        }

        // Test: Lines intersect but segments do not
        SECTION("Lines intersect but segments do not") {
            Segment2D seg;
            seg.vertices()[0] = {0.0, 0.0};
            seg.vertices()[1] = {1.0, 1.0};

            Triangle2D tri_as_seg;
            tri_as_seg.vertices()[0] = {0.0, 3.0};
            tri_as_seg.vertices()[1] = {3.0, 0.0}; // The line intersects at (1.5, 1.5)

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            REQUIRE_FALSE(result.has_value());
        }

        // Test: Parallel segments
        SECTION("Parallel segments") {
            Segment2D seg;
            seg.vertices()[0] = {0.0, 0.0};
            seg.vertices()[1] = {2.0, 0.0};

            Triangle2D tri_as_seg;
            tri_as_seg.vertices()[0] = {0.0, 1.0};
            tri_as_seg.vertices()[1] = {2.0, 1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            REQUIRE_FALSE(result.has_value());
        }

        // Test: Collinear segments with no overlap
        SECTION("Collinear segments with no overlap") {
            Segment2D seg;
            seg.vertices()[0] = {0.0, 0.0};
            seg.vertices()[1] = {1.0, 0.0};

            Triangle2D tri_as_seg;
            tri_as_seg.vertices()[0] = {2.0, 0.0};
            tri_as_seg.vertices()[1] = {3.0, 0.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            // No intersection as collinear but no overlap
            REQUIRE_FALSE(result.has_value());
        }

        // Test: Intersection at an endpoint
        SECTION("Intersection at an endpoint") {
            Segment2D seg; // A horizontal segment
            seg.vertices()[0] = {0.0, 1.0};
            seg.vertices()[1] = {4.0, 1.0};

            Triangle2D tri_as_seg; // A diagonal segment touching the endpoint of the first
            tri_as_seg.vertices()[0] = {2.0, -1.0};
            tri_as_seg.vertices()[1] = {4.0, 1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(4.0));
            REQUIRE(result->y() == Catch::Approx(1.0));
        }

        // Test: Perpendicular intersection (T-junction)
        SECTION("Perpendicular intersection (T-junction)") {
            Segment2D seg; // Vertical segment
            seg.vertices()[0] = {1.0, -1.0};
            seg.vertices()[1] = {1.0, 1.0};

            Triangle2D tri_as_seg; // Horizontal segment
            tri_as_seg.vertices()[0] = {0.0, 0.0};
            tri_as_seg.vertices()[1] = {2.0, 0.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(1.0));
            REQUIRE(result->y() == Catch::Approx(0.0));
        }

        // Test: Segment doesn't touch triangle but is collinear in 2D
        SECTION("Collinear segment doesn't touch triangle") {
            Segment2D seg;
            seg.vertices()[0] = {0.0, 0.0};
            seg.vertices()[1] = {5.0, 0.0}; // A long horizontal segment

            Triangle2D tri_as_seg;
            tri_as_seg.vertices()[0] = {1.0, 1.0};
            tri_as_seg.vertices()[1] = {4.0, 1.0}; // A triangle above the line

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(seg, tri_as_seg);

            // The segment does not intersect the triangle
            REQUIRE_FALSE(result.has_value());
        }
    }

    SECTION("Segment-Triangle Intersection in 3D") {

        // Triangle on the XY plane for most tests
        Triangle3D triangle;
        triangle.vertices()[0] = {0.0, 0.0, 0.0};
        triangle.vertices()[1] = {5.0, 0.0, 0.0};
        triangle.vertices()[2] = {0.0, 5.0, 0.0};

        // Test: Simple perpendicular intersection in 3D
        SECTION("Simple perpendicular intersection") {
            Segment3D segment;
            segment.vertices()[0] = {1.0, 1.0, 1.0};
            segment.vertices()[1] = {1.0, 1.0, -1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(1.0));
            REQUIRE(result->y() == Catch::Approx(1.0));
            REQUIRE(result->z() == Catch::Approx(0.0));
        }

        // Test: No intersection, segment parallel to triangle
        SECTION("No intersection, segment parallel to triangle") {
            Segment3D segment;
            segment.vertices()[0] = {1.0, 1.0, 1.0};
            segment.vertices()[1] = {2.0, 1.0, 1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE_FALSE(result.has_value());
        }

        // Test: No intersection, segment crosses plane but misses triangle
        SECTION("No intersection, segment crosses plane but misses triangle") {
            Segment3D segment;
            segment.vertices()[0] = {4.0, 4.0, 1.0};
            segment.vertices()[1] = {4.0, 4.0, -1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE_FALSE(result.has_value());
        }

        // Test: No intersection, segment is entirely behind the triangle
        SECTION("No intersection, segment is entirely behind the triangle") {
            Segment3D segment;
            segment.vertices()[0] = {1.0, 1.0, -1.0};
            segment.vertices()[1] = {1.0, 1.0, -2.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE_FALSE(result.has_value());
        }

        // Test: Intersection at a vertex
        SECTION("Intersection at a vertex") {
            Segment3D segment;
            segment.vertices()[0] = {0.0, 0.0, 1.0};
            segment.vertices()[1] = {0.0, 0.0, -1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(0.0));
            REQUIRE(result->y() == Catch::Approx(0.0));
            REQUIRE(result->z() == Catch::Approx(0.0));
        }

        // Test: Intersection on an edge
        SECTION("Intersection on an edge") {
            Segment3D segment;
            segment.vertices()[0] = {2.5, 0.0, 1.0};
            segment.vertices()[1] = {2.5, 0.0, -1.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(2.5));
            REQUIRE(result->y() == Catch::Approx(0.0));
            REQUIRE(result->z() == Catch::Approx(0.0));
        }

        // Test: Angled intersection
        SECTION("Angled intersection") {
            Segment3D segment;
            segment.vertices()[0] = {0.0, 0.0, 1.0};
            segment.vertices()[1] = {2.0, 2.0, -1.0}; // Intersects at (1,1,0)

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            REQUIRE(result.has_value());
            REQUIRE(result->x() == Catch::Approx(1.0));
            REQUIRE(result->y() == Catch::Approx(1.0));
            REQUIRE(result->z() == Catch::Approx(0.0));
        }

        // Test: Coplanar segment, no intersection
        SECTION("Coplanar segment, no intersection") {
            Segment3D segment;
            segment.vertices()[0] = {1.0, 1.0, 0.0};
            segment.vertices()[1] = {3.0, 3.0, 0.0};

            auto result = CabanaDSMC::Geometry::Utilities::segmentIntersectWithTriangle(segment, triangle);

            // The algorithm returns nullopt because the determinant is zero.
            REQUIRE_FALSE(result.has_value());
        }
    }
}
