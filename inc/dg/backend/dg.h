#pragma once
//reroute host calls to omp and device system to cpu
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include "polarisation.cuh"
#include "cg.h"


///@cond DOC
typedef thrust::host_vector<double> dg_container;

struct dg_workspace
{
    dg_workspace( const dg::Grid<double>& g):size(g.size()), pol(g), cg( thrust::host_vector<double>(g.size()) , g.size() ),
    w2d( dg::create::w2d(g)), v2d( dg::create::v2d(g))
                      {}
    unsigned size;
    dg::Polarisation2dX<dg_container > pol; 
    dg::CG<dg_container > cg;
    const dg_container w2d, v2d;
    cusp::csr_matrix<int, double, cusp::host_memory> matrix; //!< most efficient host matrix format
};
///@endcond
///@addtogroup polarization
///@{

/**
 * @brief opaque workspace type
 */
typedef dg_workspace dg_workspace;

/**
 * @brief Create workspace for polarization and conjugate gradient
 *
 * @note the x - direction is assumed to be the one that lies contiguously in memory!!
 * @param Nx # of points in x direction
 * @param Ny # of points in y direction
 * @param hx grid constant in x
 * @param hy grid constant in y
 * @param bcx one of dg::PER, dg::DIR_NEU, dg::NEU_DIR and dg::NEU
 * @param bcy one of dg::PER, dg::DIR_NEU, dg::NEU_DIR and dg::NEU
 *
 * @return a pointer to a newly allocated workspace 
 */
dg_workspace* dg_create_workspace( unsigned Nx, unsigned Ny, double hx, double hy, dg::bc bcx, dg::bc bcy)
{
    dg::Grid<double> g( 0, Nx*hx, 0, Ny*hy, 1, Nx, Ny, bcx, bcy); 
    dg_workspace* w = new dg_workspace( g);
    return w;
}

/**
 * @brief Assemble first order polarization matrix 
 *
 * The term discretized is \f[ \nabla ( \chi \nabla ) \f] using a first order discretization
 * @param w A previously allocated workspace
 * @param chi Polarizability
 */
void dg_update_polarizability( dg_workspace* w,  const double* chi)
{
    dg_container chi_( chi, chi+w->size); //newly allocate 
    w->matrix = w->pol.create( chi_);
}

/**
 * @brief Solve the matrix equation
 *
 * What is solved is the equation \f[ \nabla(\chi\nabla)x = b \f]
 * to first order,
 * where the polarizability from the last call of dg_update_polarizability() is used.
 * @param w A previously allocated workspace
 * @param x contains the initial guess at the beginning, and the solution on exit
 * @param b the right hand side
 * @param eps the accuracy
 *
 * @return The number of iterations used to achieve the desired accuracy
 */
unsigned dg_solve( dg_workspace* w, double *x, const double* b, double eps)
{
    dg_container x_( x, x+w->size);
    dg_container b_( b, b+w->size);
    dg::blas2::symv( w->w2d, b_, b_);
    unsigned number = w->cg( w->matrix, x_, b_, w->v2d, eps);
    thrust::copy( x_.begin(), x_.end(), x);
    return number;
}

/**
 * @brief Should be called to release resources
 *
 * @param w The workspace to delete
 */
void dg_free_workspace( dg_workspace* w)
{
    delete w;
}
///@}
