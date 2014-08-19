#ifndef _DG_TIMER_
#define _DG_TIMER_

namespace dg
{
#if (THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA) //if we don't use a GPU
#ifdef MPI_VERSION //(mpi.h is included)
/*! @brief Very simple tool for performance measuring
 * @ingroup utilities
 */
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
    /*! \brief Return time elapsed between tic and toc
     *
     * \return Time in seconds between calls of tic and toc*/
    double diff(){ return stop - start; }
  private:
    double start, stop;
};
#else //MPI_VERSION

/*! @brief Very simple tool for performance measuring
 * @ingroup utilities
 */
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
    double diff(){ return (stop.tv_sec - start.tv_sec) + 1e-6*(stop.tv_usec - start.tv_usec);}
};
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
