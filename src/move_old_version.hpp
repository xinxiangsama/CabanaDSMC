#pragma once
#include "particle.hpp"
#include "geometry/geo.hpp"
namespace CabanaDSMC{
template <class ExecutionSpace, class ParticleListType, class LocalGridType,
          class CellDataType, class StlType, class ... BoundaryTypes>
requires (Cabana::is_particle_list<ParticleListType>::value ||
            Cabana::Grid::is_particle_list<ParticleListType>::value)
void moveParticles(
    const ExecutionSpace& exec_space, ParticleListType& particle_list,
    const std::shared_ptr<LocalGridType>& local_grid, 
    const std::shared_ptr<CellDataType>& cell_data,
    const StlType& stl,
    const BoundaryTypes&... boundaries
)
{   
    // using memory_space = typename ParticleListType::memory_space;
    using mesh_type  = typename CellDataType::mesh_type;
    constexpr size_t dim = mesh_type::num_space_dim;
    using point_type = Geometry::Point<double, dim>;
    // using boundary_tuple_type = std::tuple<BoundaryTypes...>;
    // if constexpr (sizeof...(BoundaryTypes) > 0){
    //     boundary_conditions = std::make_tuple(boundaries...);
    // }
    auto boundary_conditions = std::make_tuple(boundaries...);

    //get global grid
    // const auto& global_grid = local_grid->globalGrid();

    //get local set of owned cell indices
    // auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    // get particle data view
    auto position = particle_list.slice(Particle::Field::Position {});
    auto velocity = particle_list.slice(Particle::Field::Velocity {});
    auto cell_id = particle_list.slice(Particle::Field::CellID {});
    auto remain_time  = particle_list.slice(Particle::Field::RemainTime {});

    // get cell data view
    // to be thought: what we need from cell data 
    // 1. cell time step (even get from ghost cell)
    // 2. cell cut cell faces
    auto cell_dt_view = cell_data->Dt->view();
    auto num_cut_view = cell_data->Num_cut_faces->view();
    auto face_ids_view = cell_data->Cut_face_ids->view();

    // move particles
    Kokkos::parallel_for("move particles",
        Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace {}, 0, particle_list.size()),
        KOKKOS_LAMBDA(const int idx){
            auto remain_dt = cell_dt_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0);

            while (remain_dt > 1e-12) {
                point_type old_position = {position(idx,0), position(idx,1), position(idx,2)};
                // move particle
                auto attempt_dt = remain_dt;
                for(size_t d = 0; d < dim; ++d){
                    position(idx, d) += velocity(idx, d) * attempt_dt;
                }

                // cut cell
                point_type impact_normal {};
                bool hit_surface = false;
                if (num_cut_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0) > 0) {
                    const int num_faces_in_cell = num_cut_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0);

                    for (int f = 0; f < num_faces_in_cell; ++f){
                        const auto tri_id = face_ids_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), f);
                        const auto& triangle = stl(tri_id);

                        auto hit_position_opt = Geometry::Utilities::segmentIntersectWithTriangle(
                            {old_position, {position(idx,0), position(idx,1), position(idx,2)}},
                            triangle
                        );

                        if (hit_position_opt.has_value()) {
                            auto hit_position = hit_position_opt.value();
                            double t_hit =  attempt_dt * (Kokkos::abs(hit_position.x() - old_position.x()) / Kokkos::abs(position(idx,0) - old_position.x()));

                            if (t_hit < remain_dt) {
                                attempt_dt = t_hit;
                                impact_normal = triangle.normal();
                                hit_surface = true;
                                for (size_t d = 0; d < dim; ++d) {
                                    position(idx, d)=  hit_position.coords[d];
                                }
                                break;
                            }
                        }

                    }


                    if (hit_surface) {
                        point_type old_velocity = {velocity(idx,0), velocity(idx,1), velocity(idx,2)};
                        auto v_dot_n = Geometry::dot(old_velocity, impact_normal);

                        if (v_dot_n < 0) {
                            velocity(idx, 0) -= 2.0 * v_dot_n * impact_normal.x();
                            velocity(idx, 1) -= 2.0 * v_dot_n * impact_normal.y();
                            velocity(idx, 2) -= 2.0 * v_dot_n * impact_normal.z();
                        }
                    }
                    // position(idx, 0) = 0.2;

                }
                remain_dt -= attempt_dt;
                remain_time(idx) = remain_dt;
            }


            // apply boundary conditions
            auto particle = particle_list.getParticle(idx);
            std::apply([&](const auto&... boundary){
                (boundary.apply(particle), ...);
            }, boundary_conditions);
            particle_list.setParticle(particle, idx);
        }
    );

    // to be implemented:
    // 1. check boundary condition
    // 2. variable time step
    // 3. hit with cut cell faces        
}

}
