/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Matthias Wiesenberger, 2020, within FELTOR license
 */
/**
 *  @file fpedot_omp.h
 *  @brief OpenMP version of fpedot
 *
 *  @authors
 *    Developers : \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include <omp.h>
#include "accumulate.h"

namespace dg
{
namespace exblas{
///@cond
namespace cpu{

/**
 * \brief Final step of summation -- Parallel reduction among threads
 *
 * \param tid thread ID
 * \param tnum number of threads
 * \param acc FPE
 */
template<class T, int N>
inline void Reduction(unsigned int tid, unsigned int tnum,
    std::vector<std::array<T,N>>& acc, int* status)
{
    // Custom tree reduction
    for(unsigned int s = 1; (unsigned)(1 << (s-1)) < tnum; ++s)
    {
        // 1<<(s-1) = 0001, 0010, 0100, ... = 1,2,4,8,16,...
#pragma omp barrier //all threads are ready for level s
        if(tid % (1 << s) == 0) { //1<<s = 2,4,8,16,32,...
            //only the tid thread executes this block, tid2 just sets ready
            unsigned int tid2 = tid | (1 << (s-1)); //effectively adds 1, 2, 4,...
            if(tid2 < tnum) {
                for( unsigned i=0; i<N; i++)
                    Accumulate( acc[tid2][i], acc[tid], status);
            }
        }
    }
}
} //namespace cpu
///@endcond


/*!@brief OpenMP version of fpe generalized dot product
 *
 * @copydetails fpedot_cpu
*/
template<class T, size_t N, class Functor, class ...PointerOrValues>
void fpedot_omp(int * status, unsigned size, std::array<T,N>& fpe, Functor f, PointerOrValues ...xs_ptr)
{
    // OpenMP sum+reduction
    int maxthreads = omp_get_max_threads();
    std::vector<std::array<T, N>> acc(maxthreads);
    std::vector<int> status_i( maxthreads, 0);
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();
        std::array<T,N> & myacc = acc[tid];
        int l = tid * size / tnum;
        int r = ((tid+1) * size / tnum)  - 1;
        for( unsigned u=0; u<N; u++)
            myacc[u] = T(0);
        for(int i = l; i <= r; i++)
        {
            T res = f( cpu::get_element( xs_ptr, i)...);
            cpu::Accumulate(res, myacc, &status_i[tid]);
            // std::isfinite does not work for complex
            //if( !std::isfinite(res) ) *status = 1;
        }
        cpu::Reduction<T,N>( tid, tnum, acc, status);
    }//omp parallel
    for( unsigned i=0; i<N; i++)
        fpe[i] = acc[0][i];
    for( int i=0; i<maxthreads; i++)
    {
        if( status_i[i] == 2) *status = 2;
        if( status_i[i] == 1) *status = 1;
    }
}


}//namespace exblas
} //namespace dg
