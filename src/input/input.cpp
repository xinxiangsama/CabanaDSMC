#include "input.hpp"

using namespace CabanaDSMC::Input;

SimulationConfig InputReader::read(const std::string &filename)
{
    SimulationConfig config;
    Yaml::Node root;
    Yaml::Parse(root, filename);

    return config;
}