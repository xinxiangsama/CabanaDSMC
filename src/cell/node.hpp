#pragma once

#include "cell.hpp"

namespace CabanaDSMC {
namespace Node {

template<class Scalar, class MemorySpace, class MeshType>
class NodeData {
public:
    using mesh_type = MeshType;
    using memory_space = MemorySpace;
    using local_grid_type = Cabana::Grid::LocalGrid<MeshType>;

    using entity_type = Cabana::Grid::Node;
    using array_layout = Cabana::Grid::ArrayLayout<entity_type, MeshType>;

    using bool_array_type = Cabana::Grid::Array<int, entity_type, MeshType, MemorySpace>;
    using bool_factory = Cell::ArrayFactory<bool_array_type>;

    NodeData(const std::shared_ptr<local_grid_type>& local_grid)
        : _local_grid(local_grid)
    {
        is_inside = bool_factory::create("is_inside", Cabana::Grid::createArrayLayout(local_grid, 1, entity_type()));
    }

    auto localgrid() {
        return _local_grid;
    }

    void setAllNodesToOutside()
    {
        Cabana::Grid::ArrayOp::assign(*is_inside, 0, Cabana::Grid::Ghost());
    }

    std::shared_ptr<bool_array_type> is_inside;

private:
    std::shared_ptr<local_grid_type> _local_grid;
};

} // namespace Node
} // namespace CabanaDSMC