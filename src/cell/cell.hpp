#pragma once
#include "../particle.hpp"
#include <Cabana_Grid_Array.hpp>
/*
cell is contain field macro variables
*/

namespace CabanaDSMC{

// forward declaration
template<class Scalar, int dim>
struct FieldInitData;

namespace Cell{

template <class>
struct ArrayFactory
{

};

template <class Scalar, class... Params, class EntityType, class MeshType>
struct ArrayFactory<Cabana::Grid::Array<Scalar, EntityType, MeshType, Params...>>
{
    static std::shared_ptr<Cabana::Grid::Array<Scalar, EntityType, MeshType, Params...>>
    create( const std::string& label,
            const std::shared_ptr<Cabana::Grid::ArrayLayout<EntityType, MeshType>>& layout )
    {
        return std::make_shared<Cabana::Grid::Array<Scalar, EntityType, MeshType, Params...>>(
            label, layout );
    }   
};


template<class Scalar, class MemorySpace, class MeshType>
class CellData{
public:
    using mesh_type = MeshType;
    using memory_space = MemorySpace;
    using local_grid_type = Cabana::Grid::LocalGrid<MeshType>;
    using array_layout = Cabana::Grid::ArrayLayout<Cabana::Grid::Cell, MeshType>;
    using array_type = Cabana::Grid::Array<Scalar, Cabana::Grid::Cell, MeshType, MemorySpace>;
    using uint_array_type = Cabana::Grid::Array<uint32_t, Cabana::Grid::Cell, MeshType, MemorySpace>;
    using array_factory = ArrayFactory<array_type>;
    using uint_factory= ArrayFactory<uint_array_type>;

    CellData(const std::shared_ptr<local_grid_type>& local_grid) : _local_grid(local_grid)
    {
        Velocity = array_factory::create("velocity", Cabana::Grid::createArrayLayout(local_grid, 3, Cabana::Grid::Cell()));
        Temperature = array_factory::create("temperature", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Pressure = array_factory::create("pressure", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Density = array_factory::create("density", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Max_collision_rate = array_factory::create("max collision rate", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Volume = array_factory::create("volume", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Num_particles = uint_factory::create("num particles", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Fn = uint_factory::create("Fn", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));

    }

    template<class FieldInitDataType>
    void initialize(
        const FieldInitDataType& field_data,
        const Particle::SpeciesList<memory_space>& species_list
    )
    {
        // to be implemented
    }

    auto localgrid() {
        return _local_grid;
    }

// protected:
    std::shared_ptr<array_type> Velocity;
    std::shared_ptr<array_type> Temperature;
    std::shared_ptr<array_type> Pressure;
    std::shared_ptr<array_type> Density;
    std::shared_ptr<array_type> Max_collision_rate;
    std::shared_ptr<array_type> Volume;
    std::shared_ptr<uint_array_type> Num_particles;
    std::shared_ptr<uint_array_type> Fn; // one simulation particle represents Fn real particles
private:
    std::shared_ptr<local_grid_type> _local_grid;
};


}

}