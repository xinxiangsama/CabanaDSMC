#include "./simulation/run.hpp"
#include "./input/input.hpp"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {
        Kokkos::ScopeGuard scope_guard(argc, argv);
        Kokkos::printf("hello world from cabanaDSMC \n");

        if (argc < 2) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                fprintf(stderr, "Error: No input file provided.\n");
                fprintf(stderr, "Usage: %s <config.toml>\n", argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        std::string input_file = argv[1];
        auto config = CabanaDSMC::Input::InputReader::read(input_file);

        CabanaDSMC::Run run{};
        run.init();
        run.run();
    }
    MPI_Finalize();

    return 0;
}
//---------------------------------------------------------------------------//
