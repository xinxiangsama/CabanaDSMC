#pragma once
#include "particle.hpp"
#include "geometry/geo.hpp"

namespace CabanaDSMC {

template <class ExecutionSpace, class ParticleListType, class LocalGridType,
          class CellDataType, class StlType, class ... BoundaryTypes>
requires (Cabana::is_particle_list<ParticleListType>::value ||
            Cabana::Grid::is_particle_list<ParticleListType>::value)
void moveParticles(
    const ExecutionSpace& exec_space, ParticleListType& particle_list,
    const std::shared_ptr<LocalGridType>& local_grid,
    const std::shared_ptr<CellDataType>& cell_data,
    const StlType& stl,
    const BoundaryTypes&... boundaries)
{
    auto boundary_conditions = std::make_tuple(boundaries...);

    // Get particle data slices
    auto position = particle_list.slice(Particle::Field::Position{});
    auto velocity = particle_list.slice(Particle::Field::Velocity{});
    auto cell_id = particle_list.slice(Particle::Field::CellID{});
    auto remain_time_slice = particle_list.slice(Particle::Field::RemainTime{});

    // Get cell data views
    auto cell_dt_view = cell_data->Dt->view();
    auto num_cut_view = cell_data->Num_cut_faces->view();
    auto face_ids_view = cell_data->Cut_face_ids->view();

    // Move particles in parallel
    Kokkos::parallel_for("move_particles",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, particle_list.size()),
        KOKKOS_LAMBDA(const int idx) {

            // Get the total time this particle needs to move in this step
            double remain_dt = cell_dt_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0);

            // Sub-cycle until the particle has moved for its full time
            while (remain_dt > 1e-12) { // Use a small epsilon for floating point comparison

                double time_to_impact = remain_dt;
                Geometry::Point<double, 3> impact_normal = {0.0, 0.0, 0.0};
                bool hit_surface = false;

                // 1. COLLISION DETECTION with immersed boundary (STL)
                // Only perform this expensive check if the particle is in a cut cell.
                if (num_cut_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0) > 0)
                {
                    const int num_faces_in_cell = num_cut_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0);

                    Geometry::Point<double, 3> current_pos = {position(idx,0), position(idx,1), position(idx,2)};
                    Geometry::Point<double, 3> current_vel = {velocity(idx,0), velocity(idx,1), velocity(idx,2)};

                    // Check against all candidate faces in this cell
                    for (int f = 0; f < num_faces_in_cell; ++f)
                    {
                        const uint64_t tri_id = face_ids_view(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), f);
                        const auto& triangle = stl(tri_id);

                        // Calculate time `t` to hit this triangle
                        auto hit_time_opt = Geometry::Utilities::timeToHitTriangle(current_pos, current_vel, triangle);

                        if (hit_time_opt.has_value()) {
                            double t_hit = hit_time_opt.value();
                            // If this is the earliest collision we've found so far in this sub-step...
                            if (t_hit < time_to_impact) {
                                time_to_impact = t_hit;
                                impact_normal = triangle._normal; // Use the pre-computed normal
                                hit_surface = true;
                            }
                        }
                    }
                }

                // 2. MOVE THE PARTICLE
                // The actual time step for this move is the smaller of the remaining
                // time or the time to the earliest impact.
                double attempt_dt = time_to_impact;

                for (int d = 0; d < 3; ++d) {
                    position(idx, d) += velocity(idx, d) * attempt_dt;
                }

                // 3. HANDLE COLLISION RESPONSE
                if (hit_surface) {
                    // Normalize the impact normal (important for reflection math)
                    double norm_mag = Kokkos::sqrt(dot(impact_normal, impact_normal));
                    if (norm_mag > 1e-9) {
                        impact_normal.x() /= norm_mag;
                        impact_normal.y() /= norm_mag;
                        impact_normal.z() /= norm_mag;
                    }

                    Geometry::Point<double, 3> v_old = {velocity(idx,0), velocity(idx,1), velocity(idx,2)};

                    // Specular reflection: v_new = v_old - 2 * dot(v_old, n) * n
                    double v_dot_n = dot(v_old, impact_normal);

                    // Ensure particle is moving towards the surface before reflecting
                    if (v_dot_n < 0) {
                        velocity(idx, 0) -= 2.0 * v_dot_n * impact_normal.x();
                        velocity(idx, 1) -= 2.0 * v_dot_n * impact_normal.y();
                        velocity(idx, 2) -= 2.0 * v_dot_n * impact_normal.z();
                    }
                }

                // 4. UPDATE REMAINING TIME
                remain_dt -= attempt_dt;
            } // End of sub-cycling while loop

            // Store any tiny leftover time for the next step (optional but good practice)
            remain_time_slice(idx) = remain_dt;

            // 5. APPLY DOMAIN BOUNDARY CONDITIONS
            // This is applied after all sub-cycling is complete.
            auto particle = particle_list.getParticle(idx);
            std::apply([&](const auto&... boundary){
                (boundary.apply(particle), ...);
            }, boundary_conditions);
            particle_list.setParticle(particle, idx);
        }
    );
}

} // namespace CabanaDSMC