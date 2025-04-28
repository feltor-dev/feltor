// #define CATCH_CONFIG_RUNNER // no longer needed in catch2 v3
// Copied and adapted from
// https://stackoverflow.com/questions/58289895/is-it-possible-to-use-catch2-for-testing-an-mpi-code
#include <mpi.h>
#include <catch2/catch_session.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h> //declare THRUST_DEVICE_SYSTEM

#include <sstream>

int main( int argc, char* argv[] ) {

    Catch::Session session;

    int err = session.applyCommandLine(argc, argv);
    if (err != 0) return err;

    // If only listing tests, don't initialize CUDA/MPI etc
    auto& config_data = session.configData();
    if (config_data.listTests || config_data.listTags || config_data.listReporters) {
        return session.run();
    }

    // a copy of dg::mpi_init
#ifdef _OPENMP
    int provided, error;
    error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( error == MPI_SUCCESS && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices == 0)
    {
        std::cerr << "# No CUDA capable devices found on rank "<<rank<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    // The default error handler makes it very difficult to debug...
    MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    //std::stringstream ss;

    /* save old buffer and redirect output to string stream */
    //auto cout_buf = std::cout.rdbuf( ss.rdbuf() );

    int result = session.run();

    /* reset buffer */
    //std::cout.rdbuf( cout_buf );

    // The problem with catching the cout is that debugging becomes hard
    // cause it's not printing when expected

    //std::stringstream printRank;
    //printRank << "Rank ";
    //printRank.width(2);
    //printRank << std::right << rank << ":\n";

    //for ( int i{1}; i<size; ++i ){
    //    MPI_Barrier(MPI_COMM_WORLD);
    //    if ( i == rank ){
    //        /* if all tests are passed, it's enough if we hear that from
    //         * the master. Otherwise, print results */
    //        if ( ss.str().rfind("All tests passed") == std::string::npos )
    //            std::cout << printRank.str() + ss.str();
    //    }
    //}
    ///* have master print last, because it's the one with the most assertions */
    //MPI_Barrier(MPI_COMM_WORLD);
    //if ( rank == 0 )
    //    std::cout << printRank.str() + ss.str();

    MPI_Finalize();
    return result;
}
