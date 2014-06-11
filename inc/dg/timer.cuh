#ifndef _DG_TIMER_
#define _DG_TIMER_

#if (THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP || THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CPP)
#include "../toefl/timer.h"
namespace dg
{
    /**
     * @brief If we compute on the host we use the toefl timer
     */
    typedef toefl::Timer Timer;
}
#else

namespace dg{
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

}
#endif

#endif //_DG_TIMER_
