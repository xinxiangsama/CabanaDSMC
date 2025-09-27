#include "../../src/cell/sample.hpp"
#include "../../src/cell/cell.hpp"
#include "../../src/particle.hpp"
#include "../../src/simulation/initialize.hpp"
#include "../../src/collide/collide.hpp"
#include "../../src/sort/sort.hpp"
#include "mesh/mesh.hpp"
#include <catch2/catch_all.hpp>
#include <Kokkos_Core.hpp>

using namespace CabanaDSMC;
using namespace Catch;

// 定义测试使用的设备类型
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using MeshType = Cabana::Grid::UniformMesh<double, 3>;
using CellDataType = Cell::CellData<double, MemorySpace, MeshType>;
using ParticleListType = Particle::GridParticleList<MemorySpace, 16>;

TEST_CASE("BasicSampler Functionality", "[sampler]")
{
    
    // 创建网格管理器
    std::array<double, 3> low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> high_corner = {1.0, 1.0, 1.0};
    std::array<int, 3> num_cells = {50, 50, 50};
    std::array<bool, 3> periodic = {false, false, false};
    unsigned int halo_width = 1;
    
    auto mesh_manager = std::make_shared<Mesh::MeshManager<MemorySpace>>(
        low_corner, high_corner, num_cells, MPI_COMM_WORLD, periodic, halo_width
    );
    auto local_grid = mesh_manager->getLocalGrid();
    REQUIRE(local_grid != nullptr);

    // 创建CellData
    auto cell_data = std::make_shared<Cell::CellData<double, MemorySpace, MeshType>>(local_grid);
    
    // --- 3. 定义初始化数据 ---
    FieldInitData<double, 3> field_data;
    field_data.velocity[0] = 100.0;
    field_data.velocity[1] = 0.0;
    field_data.velocity[2] = 0.0;
    field_data.temperature = 300.0;
    field_data.density = 1.225e-3;
    field_data.Fn = 1e16;
    field_data.max_collision_rate = 1.0e-20;
    field_data.time_step = 1e-4;

    // --- 定义分子种类列表 ---
    Particle::SpeciesList<MemorySpace> species_list("species_list", 1);
    species_list(0).mass = 6.63e-26; // 质量 (kg) - 氮气分子
    species_list(0).diameter = 3.7e-10; // 直径 (m)
    species_list(0).omega = 0.74; // 粘性指数
    species_list(0).Tref = 300.0; // 参考温度 (K)
    species_list(0).Zrot = 1.0; // 转动碰撞数
    initializeField<ExecutionSpace>(
        field_data,
        cell_data,
        species_list
    );

    //create particle list
    Particle::GridParticleList<MemorySpace, 16> particle_list("particles");
    using particle_type = typename Particle::GridParticleList<MemorySpace, 16>::particle_type;
    auto InitFunctor = ParticleFactory<particle_type>{};
    createParticles(
        Cabana::InitRandom {},
        ExecutionSpace {},
        InitFunctor,
        particle_list,
        local_grid,
        cell_data,
        species_list,
        0, 12345
    );
    cell_data->setToZero();
    SECTION("采样结果应与初始场数据基本一致")
    {
        Cell::BasicSampler<ExecutionSpace, ParticleListType, CellDataType> sampler;
        
        // 设置粒子数量数组和偏移数组
        auto num_particles_view = cell_data->Num_particles->view();
        auto offset_view = cell_data->Offset_particle_idx->view();
        
        CabanaDSMC::sort(
            ExecutionSpace {},
            particle_list,
            cell_data
        );
        Kokkos::fence();
        // 执行采样（average_steps设为1）
        sampler.accumulate(particle_list, cell_data, species_list);
        sampler.finalize(cell_data, 1);

        // 验证采样结果
        auto vel_view = cell_data->Velocity->view();
        auto den_view = cell_data->Density->view();
        auto temp_view = cell_data->Temperature->view();
        
        auto host_vel = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vel_view);
        auto host_den = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), den_view);
        auto host_temp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temp_view);
        
        auto owned_space = local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());
        
        // 检查所有网格的宏观物理量
        for (int i = owned_space.min(0); i < owned_space.max(0); ++i) {
            for (int j = owned_space.min(1); j < owned_space.max(1); ++j) {
                for (int k = owned_space.min(2); k < owned_space.max(2); ++k) {
                    // // 速度应该在初始值附近（由于随机性会有波动）
                    // REQUIRE(host_vel(i, j, k, 0) == Approx(1.0).margin(0.5)); // 允许±0.5的误差
                    // REQUIRE(host_vel(i, j, k, 1) == Approx(0.0).margin(0.5));
                    // REQUIRE(host_vel(i, j, k, 2) == Approx(0.0).margin(0.5));
                    
                    // // 温度应该在300K附近
                    // REQUIRE(host_temp(i, j, k, 0) == Approx(300.0).margin(50.0)); // 允许±50K的误差
                    
                    // 密度应该在初始值附近
                    REQUIRE(host_den(i, j, k, 0) == Approx(1.225e-3).epsilon(0.1)); // 允许100%的相对误差
                    // std::cout << host_den(i, j, k, 0) << std::endl;
                    std::cout << host_vel(i, j, k, 0) << std::endl;
                }
            }
        }
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            ExecutionSpace {},
            "U",
            1,
            0.1,
            *(cell_data->Velocity)
        );
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(
            ExecutionSpace {},
            "density",
            1,
            0.1,
            *(cell_data->Density)
        );
    }

}
