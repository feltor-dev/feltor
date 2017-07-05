#ifndef _DG_TIMER_
#define _DG_TIMER_
//the <thrust/device_vector.h> header must be included before this file for the THRUST_DEVICE_SYSTEM macros to work
namespace dg
{
#if (THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA) //if we don't use a GPU
#ifdef MPI_VERSION //(mpi.h is included)
///@cond
class Timer
{
  public:
    /**
    * @brief Start timer using MPI_Wtime
    *
    * @param comm the communicator 
    * @note uses MPI_Barrier(comm)
    */
    void tic( MPI_Comm comm = MPI_COMM_WORLD ){ MPI_Barrier(comm); start = MPI_Wtime();}
    /**
    * @brief Stop timer using MPI_Wtime
    *
    * @param comm the communicator 
    * @note uses MPI_Barrier(comm)
    */
    void toc( MPI_Comm comm = MPI_COMM_WORLD ){ MPI_Barrier(comm); stop = MPI_Wtime(); }
    double diff(){ return stop - start; }
  private:
    double start, stop;
};
///@endcond

#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP //MPI_VERSION
#include "omp.h"

/*! @brief Very simple tool for performance measuring
 * 
 * @code
   dg::Timer t;
   t.tic();
   some_function_to_benchmark();
   t.toc();
   std::cout << "Function took "<<t.diff()<<"s\n";
 * @endcode
 * @ingroup timer
 * @note The Timer knows what hardware you are on!
 */
class Timer
{
  public:
    /**
    * @brief Start timer 
    */
    void tic( ){ start = omp_get_wtime();}
    /**
    * @brief Stop timer 
    */
    void toc( ){ stop = omp_get_wtime(); }
    /*! \brief Return time elapsed between tic and toc
     *
     * \return Time in seconds between calls of tic and toc*/
    double diff(){ return stop - start; }
  private:
    double start, stop;
};
#else

///@cond
#include <sys/time.h>
class Timer
{
    timeval start;
    timeval stop;
    public:
    /*! @brief Start timer using gettimeofday */
    void tic(){ gettimeofday( &start, NULL);}
    /*! @brief Stop timer using gettimeofday */
    void toc(){ gettimeofday( &stop, NULL);}
    /*! \brief Return time elapsed between tic and toc
     *
     * \return Time in seconds between calls of tic and toc*/
    double diff(){ return ((stop.tv_sec - start.tv_sec)*1000000u + (stop.tv_usec - start.tv_usec))/1e6;}
};
///@endcond
#endif //MPI_VERSION
#else //THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
#ifdef MPI_VERSION

///@cond
class Timer
{
  public:
    Timer(){
        cudaEventCreate( &cu_sync);
    }
    /**
    * @brief Start timer using MPI_Wtime
    *
    * @param comm the communicator 
    * @note uses MPI_Barrier(comm)
    */
    void tic( MPI_Comm comm = MPI_COMM_WORLD ){ 
    MPI_Barrier(comm); 
    start = MPI_Wtime();}
    /**
    * @brief Stop timer using MPI_Wtime
    *
    * @param comm the communicator 
    * @note uses MPI_Barrier(comm)
    */
    void toc( MPI_Comm comm = MPI_COMM_WORLD ){ 
    cudaEventRecord( cu_sync, 0); //place event in stream
    cudaEventSynchronize( cu_sync); //sync cpu  on event
    MPI_Barrier(comm); //sync other cpus on event
    stop = MPI_Wtime(); }
    double diff(){ return stop - start; }
  private:
    double start, stop;
    cudaEvent_t cu_sync;
};
///@endcond


#else //MPI_VERSION

///@cond
class Timer
{
  public:
    Timer(){
        cudaEventCreate( &start); 
        cudaEventCreate( &stop);
    }
    /**
    * @brief Start timer using cudaEventRecord
    *
    * @param stream the stream in which the Event is placed
    */
    void tic( cudaStream_t stream = 0){ cudaEventRecord( start, stream);}
    /**
    * @brief Stop timer using cudaEventRecord and Synchronize
    *
    * @param stream the stream in which the Event is placed
    */
    void toc( cudaStream_t stream = 0){ 
        cudaEventRecord( stop, stream);
        cudaEventSynchronize( stop);
    }
    float diff(){ 
        float time; 
        cudaEventElapsedTime( &time, start, stop);
        return time/1000.;
    }
  private:
    cudaEvent_t start, stop;
};
///@endcond
#endif //MPI_VERSION
#endif //THRUST

} //namespace dg

#endif //_DG_TIMER_
