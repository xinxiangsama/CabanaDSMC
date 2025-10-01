// file: test_winding_number.cpp
#include <catch2/catch_all.hpp>
#include <Kokkos_Core.hpp>
#include "../../../src/geometry/geo.hpp"             // 你的核心几何定义
#include "../../../src/geometry/winding_number.hpp"  // 包含 winding_number_test 的文件

// 为方便测试，定义类型别名
using Point2D = CabanaDSMC::Geometry::Point<double, 2>;
using Triangle2D = CabanaDSMC::Geometry::Triangle<double, 2>;
using Stl2D = CabanaDSMC::Geometry::Stl<Kokkos::HostSpace, double, 2>;

using Point3D = CabanaDSMC::Geometry::Point<double, 3>;
using Triangle3D = CabanaDSMC::Geometry::Triangle<double, 3>;
using Stl3D = CabanaDSMC::Geometry::Stl<Kokkos::HostSpace, double, 3>;

TEST_CASE("Winding Number Point Inclusion Test", "[geometry][winding_number]") {


    SECTION("2D Winding Number Test with a Square") {
        // 创建一个 2x2 的正方形，中心在 (1,1)
        Point2D v0 = {0.0, 0.0};
        Point2D v1 = {2.0, 0.0};
        Point2D v2 = {2.0, 2.0};
        Point2D v3 = {0.0, 2.0};

        // 定义4条边，顺序为逆时针 (CCW)
        Stl2D square("square_segments", 4);
        auto host_square = Kokkos::create_mirror_view(square);

        host_square(0).points = {v0, v1}; // Bottom
        host_square(1).points = {v1, v2}; // Right
        host_square(2).points = {v2, v3}; // Top
        host_square(3).points = {v3, v0}; // Left

        Kokkos::deep_copy(square, host_square);

        // 测试内部点
        Point2D inside_point = {1.0, 1.0};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(inside_point, square) == true);

        // 测试外部点
        Point2D outside_point = {3.0, 3.0};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(outside_point, square) == false);

        // 测试边界点 (点在边上)
        Point2D on_edge_point = {1.0, 0.0};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(on_edge_point, square) == false);

        // 测试顶点
        Point2D on_vertex_point = {0.0, 0.0};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(on_vertex_point, square) == false);
    }

    SECTION("3D Winding Number Test with a Cube") {
        // ... (立方体的顶点和面的定义保持不变) ...
        Point3D v[8] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };
        Stl3D cube("cube_triangles", 12);
        auto host_cube = Kokkos::create_mirror_view(cube);
        // ... (12个三角形的填充代码保持不变) ...
        host_cube(0).points = {v[0], v[2], v[1]}; host_cube(1).points = {v[0], v[3], v[2]};
        host_cube(2).points = {v[4], v[5], v[6]}; host_cube(3).points = {v[4], v[6], v[7]};
        host_cube(4).points = {v[0], v[1], v[5]}; host_cube(5).points = {v[0], v[5], v[4]};
        host_cube(6).points = {v[3], v[6], v[2]}; host_cube(7).points = {v[3], v[7], v[6]};
        host_cube(8).points = {v[0], v[7], v[3]}; host_cube(9).points = {v[0], v[4], v[7]};
        host_cube(10).points = {v[1], v[2], v[6]}; host_cube(11).points = {v[1], v[6], v[5]};
        Kokkos::deep_copy(cube, host_cube);

        // --- 测试明确的内部和外部点 (这些测试应该总是通过) ---
        Point3D inside_point = {0.5, 0.5, 0.5};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(inside_point, cube) == true);

        Point3D outside_point = {2.0, 2.0, 2.0};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(outside_point, cube) == false);

        // --- 鲁棒的边界测试 ---
        // 定义一个小的偏移量来测试边界附近
        const double epsilon = 1e-9;

        // 测试靠近面的点
        Point3D on_face_center = {0.5, 0.5, 0.0};
        Point3D just_inside_from_face = {0.5, 0.5, epsilon};
        Point3D just_outside_from_face = {0.5, 0.5, -epsilon};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(just_inside_from_face, cube) == true);
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(just_outside_from_face, cube) == false);

        // 测试靠近边的点
        Point3D on_edge_center = {0.5, 0.0, 0.0};
        Point3D just_inside_from_edge = {0.5, epsilon, epsilon};
        Point3D just_outside_from_edge = {0.5, -epsilon, -epsilon};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(just_inside_from_edge, cube) == true);
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(just_outside_from_edge, cube) == false);

        // 测试靠近顶点的点
        Point3D on_vertex = {1.0, 1.0, 1.0};
        Point3D just_inside_from_vertex = {1.0 - epsilon, 1.0 - epsilon, 1.0 - epsilon};
        Point3D just_outside_from_vertex = {1.0 + epsilon, 1.0 + epsilon, 1.0 + epsilon};
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(just_inside_from_vertex, cube) == true);
        REQUIRE(CabanaDSMC::Geometry::Utilities::winding_number_test(just_outside_from_vertex, cube) == false);
    }
}