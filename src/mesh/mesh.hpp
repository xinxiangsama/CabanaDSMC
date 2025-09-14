#pragma once
#include <Cabana_Grid.hpp>

/*
mesh is contain geometry infomation,
grid is contain topological infomation
*/


namespace CabanaDSMC{
namespace Mesh{
using namespace Cabana::Grid;

template<class MeshType>
struct GlobalMeshFactory 
{

};

template<class Scalar, std::size_t NumSpaceDim>
struct GlobalMeshFactory<UniformMesh<Scalar, NumSpaceDim>>
{
    static std::shared_ptr<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>
    create(
        const std::array<Scalar, NumSpaceDim>& global_low_corner,
        const std::array<Scalar, NumSpaceDim>& global_high_corner,
        const Scalar cell_size )
    {
        return std::make_shared<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>(
            global_low_corner, global_high_corner, cell_size );
    }


    static std::shared_ptr<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>
    create(
        const std::array<Scalar, NumSpaceDim>& global_low_corner,
        const std::array<Scalar, NumSpaceDim>& global_high_corner,
        const std::array<Scalar, NumSpaceDim>& cell_size )
    {
        return std::make_shared<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>(
            global_low_corner, global_high_corner, cell_size );
    }

    static std::shared_ptr<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>
    create(
        const std::array<Scalar, NumSpaceDim>& global_low_corner,
        const std::array<Scalar, NumSpaceDim>& global_high_corner,
        const std::array<int, NumSpaceDim>& global_num_cell )
    {
        return std::make_shared<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>(
            global_low_corner, global_high_corner, global_num_cell );
    }
};


template<class MemorySpace, class MeshType = Cabana::Grid::UniformMesh<double, 3>>
requires Cabana::Grid::isMeshType<MeshType>::value
class MeshManager{
public:
    using mesh_type = MeshType;
    using memory_space = MemorySpace;
    using scalar_type = MeshType::scalar_type;
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;
    using global_mesh_type = Cabana::Grid::GlobalMesh<MeshType>;
    using global_mesh_factory = GlobalMeshFactory<MeshType>;
    using global_grid_type = Cabana::Grid::GlobalGrid<MeshType>;
    using local_grid_type = Cabana::Grid::LocalGrid<MeshType>;
    using local_mesh_type = Cabana::Grid::LocalMesh<MemorySpace, MeshType>;

    MeshManager(
        const std::array<scalar_type, num_space_dim>& global_low_corner,
        const std::array<scalar_type, num_space_dim>& global_high_corner,
        const std::array<int, num_space_dim>& global_num_cell,
        MPI_Comm comm,
        const std::array<bool, num_space_dim>& periodic,
        const unsigned int halo_cell_width = 0
        )
    {   
        // create global mesh
        m_global_mesh = global_mesh_factory::create(
            global_low_corner,
            global_high_corner,
            global_num_cell
        );

        // create global grid
        auto partitioner = Cabana::Grid::DimBlockPartitioner<3> {};
        m_global_grid = Cabana::Grid::createGlobalGrid(
            comm, 
            m_global_mesh,
            periodic,
            partitioner
        );

        // create local grid
        m_local_grid = Cabana::Grid::createLocalGrid(
            m_global_grid,
            halo_cell_width
        ); 

        // create local mesh
        m_local_mesh = std::make_shared<local_mesh_type>(*m_local_grid);
    }
    auto getGlobalMesh() const { return m_global_mesh; }
    auto getGlobalGrid() const { return m_global_grid; }
    auto getLocalGrid() const { return m_local_grid; }
    auto getLocalMesh() const { return m_local_mesh; }
protected:
    std::shared_ptr<global_mesh_type> m_global_mesh;
    std::shared_ptr<global_grid_type> m_global_grid;
    std::shared_ptr<local_grid_type> m_local_grid;
    std::shared_ptr<local_mesh_type> m_local_mesh;

};


template <class>
struct is_mesh_manager : public std::false_type
{

};

template <class MemorySpace, class MeshType>
struct is_mesh_manager<MeshManager<MemorySpace, MeshType>> : public std::true_type
{

};
}
};



