// file: test_sat.cpp
#include <catch2/catch_all.hpp>
#include "geometry/geo.hpp"       // 包含 Cube, Triangle
#include "geometry/cut.hpp"  // 包含 SAT 函数

// 别名
using Point3D = CabanaDSMC::Geometry::Point<double, 3>;
using Triangle3D = CabanaDSMC::Geometry::Triangle<double, 3>;
using Cube3D = CabanaDSMC::Geometry::Cube<double>;

// 辅助函数，用于创建带法线的三角形
Triangle3D createTriangle(const Point3D& v0, const Point3D& v1, const Point3D& v2)
{
    Triangle3D tri;
    tri.vertices()[0] = v0;
    tri.vertices()[1] = v1;
    tri.vertices()[2] = v2;
    auto edge1 = v1 - v0;
    auto edge2 = v2 - v0;
    tri._normal = CabanaDSMC::Geometry::cross(edge1, edge2);
    // 归一化法线 (可选，但推荐)
    double norm = std::sqrt(CabanaDSMC::Geometry::dot(tri._normal, tri._normal));
    if (norm > 1e-9) {
        tri._normal.x() /= norm;
        tri._normal.y() /= norm;
        tri._normal.z() /= norm;
    }
    return tri;
}

TEST_CASE("3D SAT Triangle-Cube Intersection", "[geometry][sat]") {

    // 创建一个中心在原点、边长为2的立方体 (-1 to 1 on all axes)
    Cube3D cube({0.0, 0.0, 0.0}, 2.0);

    SECTION("No Intersection Cases") {

        SECTION("Completely separate (AABB does not overlap)") {
            Triangle3D tri = createTriangle({3,0,0}, {4,0,0}, {3.5,1,0});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == false);
        }

        SECTION("AABB overlaps but objects do not (passes by)") {
            // 一个竖直的三角形从立方体旁边经过
            Triangle3D tri = createTriangle({-0.5, 2, -0.5}, {0.5, 2, -0.5}, {0, 2, 0.5});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == false);
        }

        SECTION("Triangle plane separates from cube") {
            // 三角形所在平面平行于立方体的一个面，但分离开
            Triangle3D tri = createTriangle({-2, -2, 1.1}, {2, -2, 1.1}, {0, 2, 1.1});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == false);
        }
    }

    SECTION("Intersection Cases") {

        SECTION("Simple piercing intersection") {
            // 三角形从中间穿过立方体
            Triangle3D tri = createTriangle({-2, 0, 0}, {2, 0, 0}, {0, 0, 2});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }

        SECTION("Triangle vertex pierces cube face") {
            // 一个角刺入立方体
            Triangle3D tri = createTriangle({0.5, 0.5, 1.5}, {2, 0.5, -0.5}, {2, -0.5, -0.5});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }

        SECTION("Triangle edge intersects cube") {
            // 一条边穿过立方体
            Triangle3D tri = createTriangle({-2, 0.5, 0.5}, {2, 0.5, 0.5}, {-2, 2.5, 0.5});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }

        SECTION("Triangle is fully inside the cube") {
            Triangle3D tri = createTriangle({-0.5, 0, 0}, {0.5, 0, 0}, {0, 0.5, 0});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }

        SECTION("Cube corner pierces triangle") {
            // 一个大的斜三角形，立方体的一个角穿过它
            Triangle3D tri = createTriangle({-2, -2, 0}, {2, -2, 0}, {0, 2, 2});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }
    }

    SECTION("Touching Cases (should be considered intersection)") {

        SECTION("Triangle vertex touches cube face") {
            Triangle3D tri = createTriangle({0, 0, 1}, {2, 0, 1}, {1, 1, 2});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }

        SECTION("Triangle edge touches cube edge") {
            Triangle3D tri = createTriangle({1, 1, 0}, {1, 1, 2}, {2, 1, 1});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }

        SECTION("Triangle face touches cube face (coplanar)") {
            Triangle3D tri = createTriangle({-0.5, -0.5, 1}, {0.5, -0.5, 1}, {0, 0.5, 1});
            REQUIRE(CabanaDSMC::Geometry::SAT(tri, cube) == true);
        }
    }
}