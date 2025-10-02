#pragma once
#include "winding_number.hpp"
#include "../cell/node.hpp"

namespace CabanaDSMC {
namespace Geometry {
template <class ExecutionSpace, class StlType, class NodeDataType>
void distinguish_node(
    const ExecutionSpace& exec,
    const StlType& stl,
    const std::shared_ptr<NodeDataType>& node_data
)
{
    using memory_space = typename ExecutionSpace::memory_space;
    using scalar_type = typename NodeDataType::mesh_type::scalar_type;
    static constexpr int dim = NodeDataType::mesh_type::num_space_dim;
    using point_type = Point<scalar_type, dim>;

    auto local_grid = node_data->localgrid();
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *local_grid );
    auto is_inside_view = node_data->is_inside->view();

    auto owned_nodes = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());
    auto policy = Cabana::Grid::createExecutionPolicy(owned_nodes, ExecutionSpace {});

    Kokkos::parallel_for("distinguish which node in the stl ", policy,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
             int ijk [3] = {i, j, k};
            scalar_type node_coords[dim];
            local_mesh.coordinates(Cabana::Grid::Node {}, ijk, node_coords);

            point_type current_point;
            for (int d = 0; d < dim; ++d) {
                current_point.coords[d] = node_coords[d];
            }

            auto is_in = Utilities::winding_number_test(current_point, stl);

            is_inside_view(i, j, k, 0) = (is_in) ? 1 : 0;
        });
}
}
}