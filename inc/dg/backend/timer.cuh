#ifndef _DG_TIMER_
#define _DG_TIMER_

namespace dg
{
#if (THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA)
#ifdef MPI_VERSION
class Timer
{
  public:
    /**
    * @brief Start timer using cudaEventRecord
    *
    * @param stream the stream in which the Event is placed
    */
    void tic( MPI_Comm comm = MPI_COMM_WORLD ){ MPI_Barrier(comm); start = MPI_Wtime();}
    /**
    * @brief Stop timer using cudaEventRecord and Synchronize
    *
    * @param stream the stream in which the Event is placed
    */
    void toc( MPI_Comm comm = MPI_COMM_WORLD ){ MPI_Barrier(comm); stop = MPI_Wtime(); }
    /*! \brief Return time elapsed between tic and toc
     *
     * \return Time in seconds between calls of tic and toc*/
    double diff(){ return stop - start; }
  private:
    double start, stop;
};
#else //MPI_VERSION

#include "../../toefl/timer.h"
    /**
     * @brief If we compute on the host we use the toefl timer
     */
    typedef toefl::Timer Timer;
#endif //MPI_VERSION
#else //THRUST

/*! @brief Very simple tool for performance measurements using CUDA-API 
 * @ingroup utilities
 */
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
    /*! \brief Return time elapsed between tic and toc
     *
     * \return Time in seconds between calls of tic and toc*/
    float diff(){ 
        float time; 
        cudaEventElapsedTime( &time, start, stop);
        return time/1000.;
    }
  private:
    cudaEvent_t start, stop;
};
#endif //THRUST

} //namespace dg

#endif //_DG_TIMER_
