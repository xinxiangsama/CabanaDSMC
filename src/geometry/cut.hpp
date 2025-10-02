#pragma once
#include "geo.hpp"
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
// =========================================================================
// Cut Cell (Triangle With Cube / Segment with Square)
// =========================================================================
namespace CabanaDSMC {
namespace Geometry {
// ----------------------------------------------
// cut cell 3d(based on Separating Axis Theorem)
template<class Scalar, size_t Dim>
KOKKOS_INLINE_FUNCTION
bool SAT(const Triangle<Scalar, Dim>& triangle, const Cube<Scalar>& cube)
requires(Dim == 3)
{
    using scalar_type = Scalar;
    constexpr size_t dim = Dim;
    using point_type = Point<Scalar, Dim>;

    // check bounding box
    auto tri_box = Utilities::getBoundingBox<scalar_type, dim>(triangle);
    auto cube_box = Utilities::getBoundingBox<scalar_type, dim>(cube);
    if (!Utilities::checkBoundingBoxIntersection(tri_box, cube_box)) {
        return false;
    }

    // check SAT
    const auto& tri_verts = triangle.vertices();
    const Point<Scalar, Dim> tri_edges[3] = {
        tri_verts[1] - tri_verts[0],
        tri_verts[2] - tri_verts[1],
        tri_verts[0] - tri_verts[2]
    };

    // test axis
    point_type axes[13];
    int num_axes = 0;

    // 1. three face normal of cube
    axes[num_axes++] = {1.0, 0.0, 0.0};
    axes[num_axes++] = {0.0, 1.0, 0.0};
    axes[num_axes++] = {0.0, 0.0, 1.0};

    // 2. normal of tri
    axes[num_axes++] = triangle.normal;

    // 3. nine cross axis (cube edge X tri edge)
    const point_type cube_edges_dirs[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            axes[num_axes++] = cross(cube_edges_dirs[i], tri_edges[j]);
        }
    }

    // --- check projection and overlapping

    for (int i = 0; i < num_axes; ++i)
    {
        auto& axis = axes[i];

        // Ignore degenerate axes
        const Scalar len_sq = dot(axis, axis);
        if (len_sq < 1e-12) continue;

        // projection tri and cube to axis
        const auto tri_interval = Utilities::getInterval(tri_verts, axis);
        const auto cube_interval = Utilities::getInterval(cube, axis);

        // check if overlapping
        if (tri_interval.max < cube_interval.min || cube_interval.max < tri_interval.min)
        {
            return false;
        }
    }

    // must intersect
    return true;
}
// -------------------------------------------------


template<class ExecutionSpace, class StlType, class CellDataType>
void cutcell(
    const ExecutionSpace& exec,
    const StlType& stl,
    const std::shared_ptr<CellDataType>& cell_data
)
{
    using memory_space = typename ExecutionSpace::memory_space;
    using scalar_type = typename CellDataType::mesh_type::scalar_type;
    constexpr int dim = CellDataType::mesh_type::num_space_dim;

    auto local_grid = cell_data->localgrid();
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>(*local_grid);
    auto global_mesh = local_grid->globalGrid().globalMesh();
    auto cell_size = global_mesh.cellSize(0);


    auto num_cut_view = cell_data->Num_cut_faces->view();
    auto offset_cut_view = cell_data->Offset_cut_faces->view();

    auto owned_cells = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());
    auto policy = Cabana::Grid::createExecutionPolicy(owned_cells, exec);

    Kokkos::parallel_for("cut cell", policy,
    KOKKOS_LAMBDA(const int i, const int j, const int k) {
        int ijk[3] = {i, j, k};
        scalar_type cell_center[dim];
        local_mesh.coordinates(Cabana::Grid::Cell(), ijk, cell_center);
        
        Geometry::Point<scalar_type, dim> center_point;
        for (int d=0; d<dim; ++d) center_point.coords[d] = cell_center[d];

        Cube<scalar_type> cell_cube(center_point, cell_size);

        uint64_t local_count = 0;
        for (std::size_t t = 0; t < stl.extent(0); ++t)
        {
            if (Geometry::SAT(stl(t), cell_cube))
            {
                local_count++;

            }
        }

        // Kokkos::printf("INFO: Cell (%d, %d, %d) is a cut-cell. Intersections: %llu\n", i, j, k, local_count);
        num_cut_view(i, j, k, 0) = local_count;
    });
}
}
}
