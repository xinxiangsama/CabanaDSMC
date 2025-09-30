#pragma once
#include "collideImpl.hpp"
#include <Kokkos_Core.hpp>
namespace CabanaDSMC{
template<class ExecutionSpace, class ParticleListType, class LocalGridType, class CellDataType, class CollideType, class RandomPoolType>
requires (Cabana::is_particle_list<ParticleListType>::value ||
            Cabana::Grid::is_particle_list<ParticleListType>::value)
void collide(
    const ExecutionSpace& exec_space, ParticleListType& particle_list,
    const std::shared_ptr<LocalGridType>& local_grid, 
    const std::shared_ptr<CellDataType>& cell_data,
    const Particle::SpeciesList<typename ParticleListType::memory_space>& species_list,
    const CollideType& collide_model,
    const  RandomPoolType& pool
)
{   
    using memory_space = typename ExecutionSpace::memory_space;

    // get cell status
    auto num_particles = cell_data->Num_particles->view();
    auto offset_particle_idx = cell_data->Offset_particle_idx->view();
    auto max_collision_rate = cell_data->Max_collision_rate->view();
    auto volume = cell_data->Volume->view();
    auto dt = cell_data->Dt->view();
    auto fn = cell_data->Fn->view();

    //get local set of owned cell indices
    auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    auto global_grid = local_grid->globalGrid();
    //create local mesh
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *local_grid );
    // Create a random number generator.
    // const auto local_seed =
    //     global_grid.blockId() + ( seed % ( global_grid.blockId() + 1 ) );
    //
    // Kokkos::Random_XorShift64_Pool<ExecutionSpace> pool( local_seed, owned_cells.size() );
    //
    // collide particles in each cell
    std::cout <<"start collision"<<std::endl;
    Cabana::Grid::grid_parallel_for("collide particles in each cell",
        exec_space,
        owned_cells,
        KOKKOS_LAMBDA(const int i, const int j, const int k){
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Cabana::Grid::Dim::I );
            int j_own = j - owned_cells.min( Cabana::Grid::Dim::J );
            int k_own = k - owned_cells.min( Cabana::Grid::Dim::K );
            int ijk [3] = {i_own, j_own, k_own};
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

            // compute possible collision pairs
            auto np = num_particles(i,j,k,0);
            auto offset_idx = offset_particle_idx(i,j,k,0);
            auto delta_t = dt(i,j,k,0);
            auto srcmax = max_collision_rate(i,j,k,0);
            auto cell_volume = volume(i,j,k,0);
            auto cell_fn = fn(i,j,k,0);

            double expected_num_collision = 0.5 * static_cast<double>(np) * static_cast<double>(np - 1) * srcmax * delta_t * cell_fn / cell_volume;

            // // log for debug
            // printf("Cell (%d,%d,%d): np=%lu, offset_idx=%lu, dt=%f, srcmax=%f, volume=%f, fn=%lu, expected_collisions=%f\n",
            //        i, j, k, np, offset_idx, delta_t, srcmax, cell_volume, cell_fn, expected_num_collision);

            int num_collision = static_cast<int>(expected_num_collision);
            double prob_collision = expected_num_collision - num_collision;
            if(num_collision < 0) return; // no particles or no collision
            // Random number generator.
            auto rand = pool.get_state( cell_id );

            double max_collision_rate_next {srcmax};
            for(int n = 0; n < num_collision; ++n){
                // randomly select two particles
                int idx1 = offset_idx + Kokkos::rand<decltype( rand ), int>::draw(rand, 0, np - 1);
                int idx2 = offset_idx + Kokkos::rand<decltype( rand ), int>::draw(rand, 0, np - 1);
                while(idx2 == idx1){ // ensure two different particles
                    idx2 = offset_idx + Kokkos::rand<decltype( rand ), int>::draw(rand, 0, np - 1);
                }
                auto particle1 = particle_list.getParticle(idx1);
                auto particle2 = particle_list.getParticle(idx2);

                // debug check select particle whether in this cell
                // for(int d = 0; d < 3; ++d){
                //     if(Cabana::get(particle1, Particle::Field::Position(), d) < low_coords[d] || Cabana::get(particle2, Particle::Field::Position(), d) < low_coords[d] || Cabana::get(particle1, Particle::Field::Position(), d) > high_coords[d] || Cabana::get(particle2, Particle::Field::Position(), d) > high_coords[d]){
                //         printf("error!!!");
                //     }
                // }


                double collision_rate = collide_model.computeCollisionRate(particle1, particle2, species_list);
                // printf("Collision rate: %e\n", collision_rate);
                max_collision_rate_next = Kokkos::max(max_collision_rate_next, collision_rate);

                // accept or reject
                double accept_prob = collision_rate / srcmax;
                double r = Kokkos::rand<decltype( rand ), double>::draw(rand, 0.0, 1.0);
                if(r < accept_prob){
                    // collide
                    collide_model.collide(particle1, particle2, rand);

                    // update particles
                    particle_list.setParticle(particle1, idx1);
                    particle_list.setParticle(particle2, idx2);
                }
            }
            max_collision_rate(i,j,k,0) = max_collision_rate_next;
            pool.free_state(rand);
        }
    );
}
}
