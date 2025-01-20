#define CATCH_CONFIG_RUNNER
// Copied and adapted from
// https://stackoverflow.com/questions/58289895/is-it-possible-to-use-catch2-for-testing-an-mpi-code
#include <mpi.h>
#include "catch2/catch.hpp"

#include <sstream>

int main( int argc, char* argv[] ) {
    MPI_Init( &argc, &argv );

    std::stringstream ss;

    /* save old buffer and redirect output to string stream */
    auto cout_buf = std::cout.rdbuf( ss.rdbuf() );

    int result = Catch::Session().run( argc, argv );

    /* reset buffer */
    std::cout.rdbuf( cout_buf );

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);


    std::stringstream printRank;
    printRank << "Rank ";
    printRank.width(2);
    printRank << std::right << rank << ":\n";

    for ( int i{1}; i<size; ++i ){
        MPI_Barrier(MPI_COMM_WORLD);
        if ( i == rank ){
            /* if all tests are passed, it's enough if we hear that from
             * the master. Otherwise, print results */
            if ( ss.str().rfind("All tests passed") == std::string::npos )
                std::cout << printRank.str() + ss.str();
        }
    }
    /* have master print last, because it's the one with the most assertions */
    MPI_Barrier(MPI_COMM_WORLD);
    if ( rank == 0 )
        std::cout << printRank.str() + ss.str();

    MPI_Finalize();
    return result;
}
