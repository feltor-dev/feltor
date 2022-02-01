#ifndef _DG_TIMER_
#define _DG_TIMER_

///@cond
#include "thrust/device_vector.h"
//the <thrust/device_vector.h> header must be included for the THRUST_DEVICE_SYSTEM macros to work
#if (THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA) //if we don't use a GPU
#ifdef MPI_VERSION //(mpi.h is included)
namespace dg
{
class Timer //CPU/ OMP + MPI
{
  public:
    void tic( MPI_Comm comm = MPI_COMM_WORLD ){ MPI_Barrier(comm); start = MPI_Wtime();}
    void toc( MPI_Comm comm = MPI_COMM_WORLD ){ MPI_Barrier(comm); stop = MPI_Wtime(); }
    double diff()const{ return stop - start; }
  private:
    double start = 0., stop = 0.;
};
}//namespace dg
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP //MPI_VERSION not defined and THRUST ==  OMP
#include "omp.h"
namespace dg
{
class Timer //OMP non-MPI
{
  public:
    void tic( ){ start = omp_get_wtime();}
    void toc( ){ stop = omp_get_wtime(); }
    double diff()const{ return stop - start; }
  private:
    double start = 0., stop = 0.;
};
}//namespace dg
#else // MPI_VERSION not defined and THRUST == CPU

///@cond
#if defined _MSC_VER //we are on windows, God help us
#include <windows.h>
namespace dg{
class Timer
{
public:
    Timer()
    {
        LARGE_INTEGER timerFreq;
        QueryPerformanceFrequency(&timerFreq);
        m_freq = 1.0 / timerFreq.QuadPart;
    }
    inline void tic(void) { QueryPerformanceCounter(&m_beginTime); }
    inline void toc(void) { QueryPerformanceCounter(&m_endTime); }
    inline double diff(void) {
        return (m_endTime.QuadPart - m_beginTime.QuadPart) * m_freq;
    }
private:
    double m_freq;
    LARGE_INTEGER m_beginTime;
    LARGE_INTEGER m_endTime;
};
#else //linux
#include <sys/time.h>
namespace dg{
class Timer //CPU non-MPI
{
    timeval start;
    timeval stop;
    public:
    void tic(){ gettimeofday( &start, NULL);}
    void toc(){ gettimeofday( &stop, NULL);}
    double diff()const{ return ((stop.tv_sec - start.tv_sec)*1000000u + (stop.tv_usec - start.tv_usec))/1e6;}
};
}//namespace dg
#endif//windows or not
#endif//MPI_VERSION

#else //THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#ifdef MPI_VERSION
namespace dg{
class Timer //GPU MPI
{
  public:
    Timer(){ }
    void tic( MPI_Comm comm = MPI_COMM_WORLD ){
        cudaDeviceSynchronize();
        MPI_Barrier(comm);
        start = MPI_Wtime();
    }
    void toc( MPI_Comm comm = MPI_COMM_WORLD ){
        cudaDeviceSynchronize();
        MPI_Barrier(comm); //sync other cpus
        stop = MPI_Wtime();
    }
    double diff()const{ return stop - start; }
  private:
    double start = 0., stop = 0.;
};
}//namespace dg

#else //MPI_VERSION

namespace dg{
class Timer// GPU non-MPI
{
  public:
    Timer(){
        cudaEventCreate( &start);
        cudaEventCreate( &stop);
    }
    void tic( cudaStream_t stream = 0){ cudaEventRecord( start, stream);}
    void toc( cudaStream_t stream = 0){
        cudaEventRecord( stop, stream);
        cudaEventSynchronize( stop);
    }
    float diff()const{
        float time;
        cudaEventElapsedTime( &time, start, stop);
        return time/1000.;
    }
  private:
    cudaEvent_t start, stop;
};
} //namespace dg
#endif //MPI_VERSION
#endif //THRUST

///@endcond

#endif //_DG_TIMER_
