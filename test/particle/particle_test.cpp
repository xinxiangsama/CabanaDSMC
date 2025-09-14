// test/trivial.cpp
#include <catch2/catch_all.hpp>  // 只要这一头即可
#include "../../src/particle.hpp"
using namespace CabanaDSMC::Particle;
using namespace Catch;
TEST_CASE("add species type", "[species]")
{
    SpeciesList<Kokkos::DefaultHostExecutionSpace::memory_space> species ("species_list", 1);
    REQUIRE(species.size() == 1);
}
