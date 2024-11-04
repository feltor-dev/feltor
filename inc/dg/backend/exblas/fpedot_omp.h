/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

/**
 *  @file exdot_serial.h
 *  @brief Serial version of exdot
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
 * \brief Parallel reduction step
 *
 * \param step step among threads
 * \param acc1 FPE of the first thread
 * \param acc2 FPE of the second thread
 */
template<int NBFPE>
inline void ReductionStep(int step, std::array<double, NBFPE>& acc1, std::array<double, NBFPE>& acc2, int * status,
    int volatile * ready)
{
#ifndef _WITHOUT_VCL
    _mm_prefetch((char const*)ready, _MM_HINT_T0);
    // Wait for thread 2 to be ready
    while(*ready < step) {
        // wait
        _mm_pause();
    }
#endif//_WITHOUT_VCL
    for( unsigned i=0; i<NBFPE; i++)
        Accumulate( acc2[i], acc1, status);
}
/**
 * \brief Final step of summation -- Parallel reduction among threads
 *
 * \param tid thread ID
 * \param tnum number of threads
 * \param acc FPE
 */
template<int NBFPE>
inline void Reduction(unsigned int tid, unsigned int tnum, std::vector<int32_t>& ready,
    std::vector<std::array<double,NBFPE>>& acc, int const linesize, int* status)
{
    // Custom tree reduction
    for(unsigned int s = 1; (unsigned)(1 << (s-1)) < tnum; ++s)
    {
        // 1<<(s-1) = 0001, 0010, 0100, ... = 1,2,4,8,16,...
        int32_t volatile * c = &ready[tid * linesize];
        ++*c; //set: ready for level s
#ifdef _WITHOUT_VCL
#pragma omp barrier //all threads are ready for level s
#endif
        if(tid % (1 << s) == 0) { //1<<s = 2,4,8,16,32,...
            //only the tid thread executes this block, tid2 just sets ready
            unsigned int tid2 = tid | (1 << (s-1)); //effectively adds 1, 2, 4,...
            if(tid2 < tnum) {
                ReductionStep<NBFPE>(s, acc[tid], acc[tid2],
                    status, &ready[tid2 * linesize]);
            }
        }
    }
}

template<typename PointerOrValue1, typename PointerOrValue2, int NBFPE>
void fpeDOT_omp(int N, PointerOrValue1 a, PointerOrValue2 b, std::array<double, NBFPE>& fpe, int * status) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    std::vector<std::array<double, NBFPE>> acc(maxthreads);
    std::vector<int32_t> ready(maxthreads * linesize);
    std::vector<int> status_i( maxthreads, 0);
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();
        *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?
#ifndef _WITHOUT_VCL
        std::array<vcl::Vec8d, NBFPE> fpe_;
        for( int i=0; i<NBFPE; i++)
            fpe_[i] = 0.;
        int l = ((tid * int64_t(N)) / tnum) & ~7ul;// & ~7ul == round down to multiple of 8
        int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;
        for(int i = l; i < r; i+=8) {
#ifndef _MSC_VER
            asm ("# myloop");
#endif
            //vcl::Vec8d r1 ;
            //vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,i), make_vcl_vec8d(b,i), r1);
            vcl::Vec8d x  = make_vcl_vec8d(a,i)* make_vcl_vec8d(b,i);
            Accumulate(x, fpe_, &status_i[tid]);
            //Accumulate(r1, fpe_, &status_i[tid]);
            vcl::Vec8db finite = vcl::is_finite( x);
            if( !vcl::horizontal_and( finite) ) status_i[tid] = 1;
        }
        if( tid+1 == tnum && r != N-1) {
            r+=1;
            //accumulate remainder
            //vcl::Vec8d r1;
            //vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,r,N-r), make_vcl_vec8d(b,r,N-r), r1);
            vcl::Vec8d x  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
            Accumulate(x, fpe_, &status_i[tid]);
            //Accumulate(r1, fpe_, &status_i[tid]);
            vcl::Vec8db finite = vcl::is_finite( x);
            if( !vcl::horizontal_and( finite) ) status_i[tid] = 1;
        }
        //now sum everything into thread fpe
        for( int k =0; k<NBFPE; k++)
            for(unsigned int i = 0; i != 8; ++i)
                Accumulate( fpe_[k][i], acc[tid], &status_i[tid]);
#else// _WITHOUT_VCL
        int l = ((tid * int64_t(N)) / tnum);
        int r = ((((tid+1) * int64_t(N)) / tnum) ) - 1;
        for(int i = l; i <= r; i++) {
            //double r1;
            //double x = TwoProductFMA(get_element(a,i),get_element(b,i),r1);
            double x = get_element(a,i)*get_element(b,i);
            Accumulate(x, acc[tid], &status_i[tid]);
            //Accumulate(r1, acc[tid], &status_i[tid]);
            if( !std::isfinite(x) ) *status = 1;
        }
#endif// _WITHOUT_VCL
        Reduction<NBFPE>( tid, tnum, ready, acc, linesize, status);
    }//omp parallel
    for( uint i=0; i<NBFPE; i++)
        fpe[i] = acc[0][i];
    for( int i=0; i<maxthreads; i++)
    {
        if( status_i[i] == 2) *status = 2;
        if( status_i[i] == 1) *status = 1;
    }
}

template<typename PointerOrValue1, typename PointerOrValue2, typename PointerOrValue3, int NBFPE>
void fpeDOT_omp(int N, PointerOrValue1 a, PointerOrValue2 b, PointerOrValue3 c, std::array<double, NBFPE>& fpe, int * status) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int maxthreads = omp_get_max_threads();
    std::vector<std::array<double, NBFPE>> acc(maxthreads);
    std::vector<int32_t> ready(maxthreads * linesize);
    std::vector<int> status_i( maxthreads, 0);
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int tnum = omp_get_num_threads();
        *(int32_t volatile *)(&ready[tid * linesize]) = 0;  // Race here, who cares?
#ifndef _WITHOUT_VCL
        std::array<vcl::Vec8d, NBFPE> fpe_;
        for( int i=0; i<NBFPE; i++)
            fpe_[i] = 0.;
        int l = ((tid * int64_t(N)) / tnum) & ~7ul;// & ~7ul == round down to multiple of 8
        int r = ((((tid+1) * int64_t(N)) / tnum) & ~7ul) - 1;
        for(int i = l; i < r; i+=8) {
#ifndef _MSC_VER
            asm ("# myloop");
#endif
            vcl::Vec8d x1  = make_vcl_vec8d(a,i)* make_vcl_vec8d(b,i);
            vcl::Vec8d x2  = x1*make_vcl_vec8d(c,i);
            Accumulate(x2, fpe_, &status_i[tid]);
            vcl::Vec8db finite = vcl::is_finite( x2);
            if( !vcl::horizontal_and( finite) ) status_i[tid] = 1;
        }
        if( tid+1 == tnum && r != N-1) {
            r+=1;
            vcl::Vec8d x1  = make_vcl_vec8d(a,r,N-r)*make_vcl_vec8d(b,r,N-r);
            vcl::Vec8d x2  = x1*make_vcl_vec8d(c,r,N-r);
            Accumulate(x2, fpe_, &status_i[tid]);
            vcl::Vec8db finite = vcl::is_finite( x2);
            if( !vcl::horizontal_and( finite) ) status_i[tid] = 1;
        }
        //now sum everything into thread fpe
        for( int k =0; k<NBFPE; k++)
            for(unsigned int i = 0; i != 8; ++i)
                Accumulate( fpe_[k][i], acc[tid], &status_i[tid]);
#else// _WITHOUT_VCL
        for(int i = 0; i < N; i++) {
            double x1 = get_element(a,i)*get_element(b,i);
            double x2 = x1*get_element(c,i);
            Accumulate(x2, fpe, &status_i[tid]);
            if( !std::isfinite(x2) ) *status = 1;
        }
#endif// _WITHOUT_VCL
        Reduction<NBFPE>( tid, tnum, ready, acc, linesize, status);
    }//omp parallel
    for( uint i=0; i<NBFPE; i++)
        fpe[i] = acc[0][i];
    for( int i=0; i<maxthreads; i++)
    {
        if( status_i[i] == 2) *status = 2;
        if( status_i[i] == 1) *status = 1;
    }
}

}//namespace cpu
///@endcond

/*!@brief serial version of exact dot product
 *
 * Computes the exact dot \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param fpe the FPE holding the result (write-only)
 * @sa \c exblas::cpu::Round  to convert the FPE into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE>
void fpedot_omp(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, std::array<double, NBFPE>& fpe, int * status){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");

    cpu::fpeDOT_omp<PointerOrValue1,PointerOrValue2,NBFPE>(
            (int)size,x1_ptr,x2_ptr, fpe, status);
}
/*!@brief serial version of exact dot product
 *
 * Computes the exact dot \f[ \sum_{i=0}^{N-1} x_i w_i y_i\f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param x3_ptr third array
 * @param fpe the FPE holding the result (write-only)
 * @sa \c exblas::cpu::Round  to convert the FPE into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, class PointerOrValue3, size_t NBFPE>
void fpedot_omp(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, PointerOrValue3 x3_ptr, std::array<double, NBFPE>& fpe, int * status){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue3>::value, "PointerOrValue3 needs to be T or T* with T one of (const) float or (const) double");

    cpu::fpeDOT_omp<PointerOrValue1,PointerOrValue2,PointerOrValue3,NBFPE>(
            (int)size,x1_ptr,x2_ptr,x3_ptr, fpe, status);
}


}//namespace exblas
} //namespace dg
