
#pragma once
/*! @file
 *
 * @brief Some utility functions for the dg::evaluate routines
 */

///@brief Expands to \__host__ \__device__ if compiled with nvcc else is empty
#define DG_DEVICE
#ifdef __CUDACC__
#undef DG_DEVICE
#define DG_DEVICE __host__ __device__
#endif

namespace dg{
///@addtogroup functions
///@{

/**
 * @brief
 * \f[ f(x) = 1\f]
 *
 * Useful for integration.
 * @param x x-value
 *
 * @return 1
 */
DG_DEVICE
static inline double one( double x) {return 1;}

/**
 * @brief
 * \f[ f(x,y) = 1\f]
 *
 * @param x x-value
 * @param y y-value
 *
 * @return 1
 */
DG_DEVICE
static inline double one( double x, double y) {return 1;}

/**
 * @brief
 * \f[ f(x,y,z) = 1\f]
 *
 * @param x x-value
 * @param y x-value
 * @param z x-value
 *
 * @return 1
 */
DG_DEVICE
static inline double one( double x, double y, double z) {return 1;}

/**
 * @brief
 * \f[ f(x) = 0\f]
 *
 * @param x x-value
 *
 * @return 0
 */
DG_DEVICE
static inline double zero( double x) {return 0.;}

/**
 * @brief
 * \f[ f(x,y) = 0\f]
 *
 * @param x x-value
 * @param y y-value
 *
 * @return 0
 */
DG_DEVICE
static inline double zero( double x, double y) {return 0.;}

///@brief \f[ f(x,y,z) = 0\f]
DG_DEVICE
static inline double zero( double x, double y, double z) {return 0.;}

///@brief \f[ f(x) = x\f]
DG_DEVICE
static inline double cooX1d( double x) {return x;}
///@brief \f[ f(x,y) = x\f]
DG_DEVICE
static inline double cooX2d( double x, double y) {return x;}
///@brief \f[ f(x,y,z) = x\f]
DG_DEVICE
static inline double cooX3d( double x, double y, double z) {return x;}

///@brief \f[ f(x,y) = y\f]
DG_DEVICE
static inline double cooY2d( double x, double y) {return y;}
///@brief \f[ f(x,y,z) = y\f]
DG_DEVICE
static inline double cooY3d( double x, double y, double z) {return y;}
///@brief \f[ f(x,y,z) = z\f]
DG_DEVICE
static inline double cooZ3d( double x, double y, double z) {return z;}
} //namespace dg

///@}

