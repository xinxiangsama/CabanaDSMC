#pragma once
#include "../mesh/mesh.hpp"
#include "../cell/cell.hpp"
#include <Cabana_Grid.hpp>
namespace CabanaDSMC{


template<class Scalar, int dim>
struct FieldInitData{
    Scalar velocity [dim];
    Scalar temperature;
    Scalar density;
    u_int32_t Fn; // one simulation particle represents Fn real particles
    Scalar max_collision_rate;
};


template <class ExecutionSpace, class InitFunctor, class ParticleListType, class LocalGridType,
          class CellDataType>
requires Cabana::is_particle_list<ParticleListType>::value
int createParticles(
    Cabana::InitRandom, const ExecutionSpace& exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    const std::shared_ptr<LocalGridType>& local_grid, 
    const std::shared_ptr<CellDataType>& cell_data,
    const Particle::SpeciesList<typename ParticleListType::memory_space>& species_list,
    const std::size_t previous_num_particles = 0,
    const uint64_t seed = 123456
)
{   
    using memory_space = typename ParticleListType::memory_space;
    
    //create local mesh
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *local_grid );

    //get global grid
    const auto& global_grid = local_grid->globalGrid();

    //get local set of owned cell indices
    auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    // Create a random number generator.
    const auto local_seed =
        global_grid.blockId() + ( seed % ( global_grid.blockId() + 1 ) );
    Kokkos::Random_XorShift64_Pool<ExecutionSpace> pool;
    pool.init( local_seed, owned_cells.size() );

    // Allocate enough space for the case the particles consume the entire
    // sum local grid total particle num
    size_t total_paritcle_num = 0;
    auto num_particles = cell_data->Num_particles->view();
    Kokkos::parallel_reduce("sum local grid total particle num",
        Cabana::Grid::createExecutionPolicy(owned_cells, exec_space),
        KOKKOS_LAMBDA(const int i, const int j, const int k, size_t& update){
            update += num_particles(i,j,k,0);
        }, 
        total_paritcle_num
    );

    auto& aosoa = particle_list.aosoa();
    aosoa.resize( previous_num_particles + total_paritcle_num );

    // Initialize particles.
    auto velocity = cell_data->Velocity->view();
    auto temperature = cell_data->Temperature->view();

    auto count = Kokkos::View<int*, memory_space>( "particle_count", 1 );
    Kokkos::deep_copy( count, previous_num_particles );
    grid_parallel_for(
        "Cabana::Grid::ParticleInit::Random", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Cabana::Grid::Dim::I );
            int j_own = j - owned_cells.min( Cabana::Grid::Dim::J );
            int k_own = k - owned_cells.min( Cabana::Grid::Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Cabana::Grid::Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Cabana::Grid::Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Cabana::Grid::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cabana::Grid::Node(), high_node, high_coords );

            // Random number generator.
            auto rand = pool.get_state( cell_id );

            // get cell properties
            double cell_velocity[3] = {velocity(i,j,k,0), velocity(i,j,k,1), velocity(i,j,k,2)};
            double cell_temperature = temperature(i,j,k,0);
            auto particles_per_cell = num_particles(i,j,k,0);
            
            // get species properties
            // assume only one species for now
            auto species = species_list(0);

            // compute maxwell properties
            double kB = 1.380649e-23; // m2 kg s-2 K-1
            double v_std = Kokkos::sqrt(2.0 * kB * cell_temperature / species.mass);

            // Create particles.
            double position[3] {};
            double velocity[3] {};
            double e_rot {}, e_vib {};
            for ( int p = 0; p < particles_per_cell; ++p )
            {
                // Local particle id.
                int pid =
                    previous_num_particles + cell_id * particles_per_cell + p;

                // Select a random point in the cell for the particle
                // location. These coordinates are logical.
                for ( int d = 0; d < 3; ++d )
                {
                    position[d] = Kokkos::rand<decltype( rand ), double>::draw(
                        rand, low_coords[d], high_coords[d] );
                }

                // Select a random velocity from the local cell
                double random_number {};
                for ( int d = 0; d < 3; ++d )
                {
                    random_number = Kokkos::rand<decltype( rand ), double>::draw(rand, 0.0, 1.0);
                    auto a1 = Kokkos::sqrt(-Kokkos::log( random_number ));
                    random_number = Kokkos::rand<decltype( rand ), double>::draw(rand, 0.0, 1.0);
                    auto a2 = Kokkos::sin( 2.0 * M_PI * random_number );
                    velocity[d] = v_std * a1 * a2 + cell_velocity[d];
                }
                // Create a new particle with the given logical coordinates.
                auto particle = particle_list.getParticle( pid );
                bool created = create_functor( 
                    pid, 
                    position, velocity, 
                    e_rot, e_vib, 
                    cell_id, 
                    particle 
                );

                // If we created a new particle insert it into the list.
                if ( created )
                {
                    auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                    particle_list.setParticle( particle, c );
                }
            }
        }
    );
    return 0;
}

template<class ExecutionSpace, class FieldInitDataType, class CellDataType>
void initializeField(
    const FieldInitDataType& field_data,
    const std::shared_ptr<CellDataType>& cell_data,
    const Particle::SpeciesList<typename CellDataType::memory_space>& species_list
)
{
    auto local_grid = cell_data->localgrid();

    auto velocity = cell_data->Velocity->view();
    auto temperature = cell_data->Temperature->view();
    auto density = cell_data->Density->view();
    auto num_particles = cell_data->Num_particles->view();
    auto fn = cell_data->Fn->view();
    auto max_collision_rate = cell_data->Max_collision_rate->view();

    // create local mesh
    auto local_mesh = Cabana::Grid::createLocalMesh<typename CellDataType::memory_space>( *local_grid );

    //get local set of owned cell indices
    auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    // compute cell volume
    auto volume = cell_data->Volume->view();
    Cabana::Grid::grid_parallel_for(
        "compute cell volume",
        ExecutionSpace {},
        owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            volume(i,j,k,0) = local_mesh.measure(Cabana::Grid::Cell(), low_node );
            // volume(i,j,k,0) = 0.01;
        }
    );

    // initialize field data
    Cabana::Grid::grid_parallel_for("initialize cell data", 
        ExecutionSpace {},
        owned_cells,
        KOKKOS_LAMBDA(const int i, const int j, const int k){
            velocity(i,j,k,0) = field_data.velocity[0];
            velocity(i,j,k,1) = field_data.velocity[1];
            velocity(i,j,k,2) = field_data.velocity[2];
            temperature(i,j,k,0) = field_data.temperature;
            max_collision_rate(i,j,k,0) = field_data.max_collision_rate;
            density(i,j,k,0) = field_data.density;
            fn(i,j,k,0) = field_data.Fn;
        }
    );

    // initalize num particles
    // assume only one species for now
    auto species = species_list(0);

    Cabana::Grid::grid_parallel_for("initialize num particles", 
        ExecutionSpace {},
        owned_cells,
        KOKKOS_LAMBDA(const int i, const int j, const int k){
            // compute num particles in each cell
            double volume_cell = volume(i,j,k,0);
            double n = density(i,j,k,0) / species.mass; // number density
            double num_real_particles = n * volume_cell;
            num_particles(i,j,k,0) = static_cast<u_int32_t>(num_real_particles / fn(i,j,k,0));
        }
    );


}


// template
template <class ParticleType>
struct ParticleFactory
{
    void operator() (
        const int pid, 
        const double position[3], const double velocity[3], 
        const double e_rot, const double e_vib, 
        const int cell_id,
        ParticleType& particle
    ) const
    {
        Cabana::get(particle, Particle::Field::Position(), 0) = position[0];
        Cabana::get(particle, Particle::Field::Position(), 1) = position[1];
        Cabana::get(particle, Particle::Field::Position(), 2) = position[2];

        Cabana::get(particle, Particle::Field::Velocity(), 0) = velocity[0];
        Cabana::get(particle, Particle::Field::Velocity(), 1) = velocity[1];
        Cabana::get(particle, Particle::Field::Velocity(), 2) = velocity[2];

        Cabana::get(particle, Particle::Field::RotEnergy(), 0) = e_rot;

        Cabana::get(particle, Particle::Field::VibEnergy(), 0) = e_vib;

        Cabana::get(particle, Particle::Field::SpeciesID(), 0) = 0; //assume only one species for now

        Cabana::get(particle, Particle::Field::CellID(), 0) = cell_id;

        Cabana::get(particle, Particle::Field::GlobalID(), 0) = pid;
        Cabana::get(particle, Particle::Field::IsActive(), 0) = true;
    }
};

}