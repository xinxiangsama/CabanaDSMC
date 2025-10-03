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
    using entity_type = Cabana::Grid::Cell;
    using local_grid_type = Cabana::Grid::LocalGrid<MeshType>;
    using array_layout = Cabana::Grid::ArrayLayout<Cabana::Grid::Cell, MeshType>;
    using array_type = Cabana::Grid::Array<Scalar, Cabana::Grid::Cell, MeshType, MemorySpace>;
    using uint_array_type = Cabana::Grid::Array<uint64_t, Cabana::Grid::Cell, MeshType, MemorySpace>;
    using int_array_type = Cabana::Grid::Array<int, Cabana::Grid::Cell, MeshType, MemorySpace>;
    using array_factory = ArrayFactory<array_type>;
    using uint_factory= ArrayFactory<uint_array_type>;
    using int_factory= ArrayFactory<int_array_type>;

    static constexpr size_t MAX_FACES_PER_CELL = 16;

    CellData(const std::shared_ptr<local_grid_type>& local_grid) : _local_grid(local_grid)
    {
        Velocity = array_factory::create("velocity", Cabana::Grid::createArrayLayout(local_grid, 3, Cabana::Grid::Cell()));
        Temperature = array_factory::create("temperature", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Pressure = array_factory::create("pressure", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Density = array_factory::create("density", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Max_collision_rate = array_factory::create("max collision rate", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Volume = array_factory::create("volume", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Num_particles = uint_factory::create("num particles", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Offset_particle_idx = uint_factory::create("offset particle idx", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Fn = uint_factory::create("Fn", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Dt = array_factory::create("local time step", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));

        // ============= cut cell  ================================== //
        Num_cut_faces = int_factory::create("num cut faces", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
        Cut_face_ids = uint_factory::create("num cut faces", Cabana::Grid::createArrayLayout(local_grid, MAX_FACES_PER_CELL, Cabana::Grid::Cell()));
        // Offset_cut_faces = uint_factory::create("cut faces offset ids", Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Cell()));
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

    void setToZero()
    {
        Cabana::Grid::ArrayOp::assign(*Velocity, 0.0, Cabana::Grid::Ghost());
        Cabana::Grid::ArrayOp::assign(*Temperature, 0.0, Cabana::Grid::Ghost());
        Cabana::Grid::ArrayOp::assign(*Pressure, 0.0, Cabana::Grid::Ghost());
        Cabana::Grid::ArrayOp::assign(*Density, 0.0, Cabana::Grid::Ghost());
    }
// protected:
    // macroscopic field data
    std::shared_ptr<array_type> Velocity;
    std::shared_ptr<array_type> Temperature;
    std::shared_ptr<array_type> Pressure;
    std::shared_ptr<array_type> Density;
    // simulation related field data
    std::shared_ptr<array_type> Max_collision_rate;
    std::shared_ptr<array_type> Volume;
    std::shared_ptr<uint_array_type> Num_particles;
    std::shared_ptr<uint_array_type> Offset_particle_idx; // offset index of particles in each cell
    std::shared_ptr<uint_array_type> Fn; // one simulation particle represents Fn real particles
    std::shared_ptr<array_type> Dt; // local time step for each cell

    //----------------------------
    // cut cell 
    //----------------------------
    std::shared_ptr<uint_array_type> Cut_face_ids;
    std::shared_ptr<int_array_type> Num_cut_faces; // bit mask for cut cell faces
    // std::shared_ptr<uint_array_type> Offset_cut_faces;
private:
    std::shared_ptr<local_grid_type> _local_grid;
};


}

}