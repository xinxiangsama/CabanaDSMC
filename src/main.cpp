#include "./simulation/run.hpp"
//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {   
        Kokkos::ScopeGuard scope_guard( argc, argv );
        Kokkos::printf("hello world from cabanaDSMC \n");
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//