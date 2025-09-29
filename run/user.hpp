#pragma once
#include <stddef.h>
#include <Kokkos_Core.hpp>
namespace CabanaDSMC{
namespace UserSpecfic{

constexpr int dim = 3;

using scalar_type = double;

using exec_space = Kokkos::DefaultExecutionSpace;
using memory_space = exec_space::memory_space;
}
}