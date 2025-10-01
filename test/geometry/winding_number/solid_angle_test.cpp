// file: test_solid_angle.cpp
#include <catch2/catch_all.hpp>
#include <numbers>    // For std::numbers::pi
#include "../../../src/geometry//winding_number.hpp"

// 为方便测试，定义类型别名
using Point3D = CabanaDSMC::Geometry::Point<double, 3>;
using Triangle3D = CabanaDSMC::Geometry::Triangle<double, 3>;


using Point3D = CabanaDSMC::Geometry::Point<double, 3>;
using Triangle3D = CabanaDSMC::Geometry::Triangle<double, 3>;

TEST_CASE("3D Solid Angle Calculation", "[geometry][solid_angle]") {

    SECTION("Known case: Point at origin for a triangle covering one octant") {
        Triangle3D octant_triangle;
        // 顶点顺序 {v1, v2, v3} 从原点看是逆时针 (CCW)
        octant_triangle.points[0] = {1.0, 0.0, 0.0};
        octant_triangle.points[1] = {0.0, 1.0, 0.0};
        octant_triangle.points[2] = {0.0, 0.0, 1.0};

        Point3D origin_point = {0.0, 0.0, 0.0};

        // 预期: (4*pi)/8 = pi/2
        double expected_angle = std::numbers::pi / 2.0;
        double solid_angle = CabanaDSMC::Geometry::Utilities::computeSolidAngle(origin_point, octant_triangle);

        REQUIRE(solid_angle == Catch::Approx(expected_angle));
    }

    SECTION("Sign changes when point moves to the other side of the triangle") {
        Triangle3D triangle_ccw; // 使用逆时针顺序 (CCW)
        // 原始顺序: {1,1,0}, {-1,1,0}, {0,-1,0} 是顺时针(CW)
        // 修正后顺序: {-1,1,0}, {1,1,0}, {0,-1,0} 是逆时针(CCW)
        triangle_ccw.points[0] = {-1.0, 1.0, 0.0};
        triangle_ccw.points[1] = {1.0, 1.0, 0.0};
        triangle_ccw.points[2] = {0.0, -1.0, 0.0};

        Point3D point_positive_z = {0.0, 0.0, 1.0};
        Point3D point_negative_z = {0.0, 0.0, -1.0};

        double angle_pos = CabanaDSMC::Geometry::Utilities::computeSolidAngle(point_positive_z, triangle_ccw);
        double angle_neg = CabanaDSMC::Geometry::Utilities::computeSolidAngle(point_negative_z, triangle_ccw);

        // 现在从 z>0 看是 CCW, 结果应为正
        REQUIRE(angle_pos > 0);
        // 从 z<0 看是 CW, 结果应为负
        REQUIRE(angle_neg < 0);
        REQUIRE(angle_pos == Catch::Approx(-angle_neg));
    }

    SECTION("Angle approaches zero as point moves far away") {
        Triangle3D triangle;
        triangle.points[0] = {1.0, 0.0, 0.0};
        triangle.points[1] = {0.0, 1.0, 0.0};
        triangle.points[2] = {0.0, 0.0, 1.0};

        Point3D far_point = {1000.0, 1000.0, 1000.0};

        double solid_angle = CabanaDSMC::Geometry::Utilities::computeSolidAngle(far_point, triangle);
        // 这里的符号取决于点在哪一侧，但其绝对值应趋近于0
        REQUIRE(solid_angle == Catch::Approx(0.0).margin(1e-5));
    }

    SECTION("Sum of angles for a closed polyhedron (tetrahedron) is -4*PI due to inward-facing normals") {
        Point3D v0 = { 1.0,  1.0,  1.0};
        Point3D v1 = { 1.0, -1.0, -1.0};
        Point3D v2 = {-1.0,  1.0, -1.0};
        Point3D v3 = {-1.0, -1.0,  1.0};

        // 这些面的顶点顺序 (从外部看) 都是顺时针的，因此它们的法向量指向内部。
        Triangle3D face1; face1.points = {v0, v2, v1};
        Triangle3D face2; face2.points = {v0, v1, v3};
        Triangle3D face3; face3.points = {v0, v3, v2};
        Triangle3D face4; face4.points = {v1, v2, v3};

        Triangle3D faces[] = {face1, face2, face3, face4};

        Point3D center_point = {0.0, 0.0, 0.0};
        double total_solid_angle = 0.0;

        for (const auto& face : faces) {
            total_solid_angle += CabanaDSMC::Geometry::Utilities::computeSolidAngle(center_point, face);
        }

        // 因为法向量都指向内部，所以内部点的总固体角是 -4*PI。
        double expected_total_angle = -4.0 * std::numbers::pi;
        REQUIRE(total_solid_angle == Catch::Approx(expected_total_angle));
    }
}