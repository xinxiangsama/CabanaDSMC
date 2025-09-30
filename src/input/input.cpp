#include "input.hpp"

using namespace CabanaDSMC::Input;

SimulationConfig InputReader::read(const std::string &filename)
{
    using scalar_type = UserSpecfic::scalar_type;

    // MPI rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        if (rank == 0)
            std::cerr << "[ERROR] Cannot open YAML file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    SimulationConfig config;
    YAML::Node root = YAML::LoadFile(filename);

    // --------------------------------------
    // global
    // -------------------------------------
    auto global = root["global"];
    config.dt = global["dt"].as<scalar_type>();
    config.steps = global["steps"].as<size_t>();
    config.seed = global["seed"].as<int>();

    // --------------------------------------
    // initial condition
    // -------------------------------------
    auto initial = root["initial"];
    for(int d = 0; d< SimulationConfig::dim; ++d)
        config.initial.velocity[d] = initial["velocity"][d].as<scalar_type>();

    config.initial.temperature        = initial["temperature"].as<scalar_type>();
    config.initial.density            = initial["density"].as<scalar_type>();
    config.initial.Fn                 = initial["Fn"].as<uint64_t>();
    config.initial.max_collision_rate = initial["max_collision_rate"].as<scalar_type>();
    config.initial.time_step          = root["global"]["dt"].as<scalar_type>();

    // -------------------------------
    // mesh config
    // -------------------------------
    auto grid = root["grid"];
    for(int d = 0; d< SimulationConfig::dim; ++d){
        config.mesh_config.global_num_cell[d] = grid["cells"][d].as<int>();
        config.mesh_config.global_low_corner[d] = grid["low_corner"][d].as<scalar_type>();
        config.mesh_config.global_high_corner[d] = grid["high_corner"][d].as<scalar_type>();
        config.mesh_config.periodic[d] = grid["periodic"][d].as<bool>();
    }
    config.mesh_config.halo_cell_width=  grid["halo_cell_width"].as<uint8_t>();

    // -------------------------------
    // boundary config
    // -------------------------------
    auto boundaries = root["boundary"];
    auto boundary_num = boundaries.size();

    if (boundary_num != SimulationConfig::dim * 2) {
        if (rank == 0)
            std::cerr << "[ERROR] Number of boundary conditions ("
                      << boundary_num << ") does not match dim*2 ("
                      << SimulationConfig::dim * 2 << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (size_t i = 0; i < boundary_num; ++i) {
        auto b = boundaries[i];
        auto& bc = config.boundary_config[i];

        std::string type  = b["type"].as<std::string>();
        std::string axis  = b["axis"].as<std::string>();
        std::string slide = b["slide"].as<std::string>();

        auto boundary_type = Boundary::boundaryTypedict.find(type);
        if (boundary_type != Boundary::boundaryTypedict.end()) {
            bc.boundary_type = boundary_type->second;
        }

        int axis_idx = (axis == "x") ? 0 : (axis == "y") ? 1 : 2;

        bc.normal[0] = bc.normal[1] = bc.normal[2] = scalar_type(0);
        bc.normal[axis_idx] = (slide == "min") ? 1.0 : -1.0;

        bc.position = (slide == "min")
            ? config.mesh_config.global_low_corner[axis_idx]
            : config.mesh_config.global_high_corner[axis_idx];

        if (b["temperature"])
            bc.temperature = b["temperature"].as<scalar_type>();
        else
            bc.temperature = scalar_type(0.0);

        if (rank == 0) {
            std::cout << "[INFO] Boundary " << i
                      << " type=" << type
                      << ", axis=" << axis
                      << ", slide=" << slide
                      << ", pos=" << bc.position
                      << ", normal=(" << bc.normal[0] << "," << bc.normal[1] << "," << bc.normal[2] << ")"
                      << ", T=" << bc.temperature
                      << std::endl;
        }
    }

    // -------------------------------
    // species list
    // -------------------------------
    using h_species_list_type = SimulationConfig::h_species_list_type;
    auto species_list = root["species"];
    auto species_num = species_list.size();
    config.species_list = h_species_list_type("host species list",  species_num);
    for (size_t i = 0; i < species_num; ++i)
    {
        auto s = species_list[i];

        auto name = s["name"].as<std::string>();

        config.species_list(i).mass   = s["mass"].as<scalar_type>();
        config.species_list(i).diameter  = s["diam"].as<scalar_type>();
        config.species_list(i).omega  = s["omega"].as<scalar_type>();
        config.species_list(i).Tref   = s["Tref"].as<scalar_type>();
        config.species_list(i).Zrot   = s["Zrot"].as<uint32_t>();

        if (rank == 0)
        {
            std::cout << "[INFO] Added species " << name
                    << " (mass=" << config.species_list(i).mass
                    << ", diam=" << config.species_list(i).diameter
                    << ", omega=" << config.species_list(i).omega
                    << ", Tref=" << config.species_list(i).Tref
                    << ", Zrot=" << config.species_list(i).Zrot
                    << ")" << std::endl;
        }
    }
    return config;
}