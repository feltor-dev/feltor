/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file cpu/blas1/ExSUM.hpp
 *  \brief Provides a set of summation routines
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXSUM_HPP_
#define EXSUM_HPP_

#include "superaccumulator.hpp"
#include "ExSUM.FPE.hpp"
#include <omp.h>

#ifdef EXBLAS_MPI
    #include <mpi.h>
#endif

/**
 * \ingroup ExSUM
 * \brief Parallel summation computes the sum of elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm that relies upon 
 *     floating-point expansions of size CACHE and superaccumulators when needed
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector to start with 
 * TODO: not done for inca and offset
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
template<typename CACHE> double ExSUMFPE(int N, double *a, int inca, int offset);

#endif // EXSUM_HPP_
