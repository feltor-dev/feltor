#pragma once

#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#include "evaluation.cuh"

/*! @file
  @brief Function discretization routines for mpi vectors
  */
namespace dg
{


///@addtogroup evaluation
///@{

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x) on the given grid
 * @copydoc hide_binary
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate f
 *
 * @return  A MPI Vector with values
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_mpi_evaluate2d
 */
template< class BinaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( const BinaryOp& f, const aMPITopology2d& g)
{
    //since the local grid is not binary compatible we have to use this implementation
    unsigned n = g.n();
    Grid2d l = g.local();
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
    thrust::host_vector<double> absx( l.n()*l.Nx());
    thrust::host_vector<double> absy( l.n()*l.Ny());
    for( unsigned i=0; i<l.Nx(); i++)
        for( unsigned j=0; j<n; j++)
        {
            unsigned coord = i+l.Nx()*coords[0];
            double xmiddle = DG_FMA( g.hx(), (double)(coord), g.x0());
            double h2 = g.hx()/2.;
            double absj = 1.+g.dlt().abscissas()[j];
            absx[i*n+j] = DG_FMA( h2, absj, xmiddle);
        }
    for( unsigned i=0; i<l.Ny(); i++)
        for( unsigned j=0; j<n; j++)
        {
            unsigned coord = i+l.Ny()*coords[1];
            double ymiddle = DG_FMA( g.hy(), (double)(coord), g.y0());
            double h2 = g.hy()/2.;
            double absj = 1.+g.dlt().abscissas()[j];
            absy[i*n+j] = DG_FMA( h2, absj, ymiddle );
        }

    thrust::host_vector<double> w( l.size());
    for( unsigned i=0; i<l.Ny(); i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<l.Nx(); j++)
                for( unsigned r=0; r<n; r++)
                    w[ ((i*n+k)*l.Nx() + j)*n + r] = f( absx[j*n+r], absy[i*n+k]);
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
    return v;
};
///@cond
MPI_Vector<thrust::host_vector<double> > evaluate( double(f)(double, double), const aMPITopology2d& g)
{
    return evaluate<double(double, double)>( *f, g);
};
///@endcond

/**
 * @brief Evaluate a function on gaussian abscissas
 *
 * Evaluates f(x,y,z) on the given grid
 * @copydoc hide_ternary
 * @param f The function to evaluate: f = f(x,y,z)
 * @param g The 3d grid on which to evaluate f
 *
 * @return  A MPI Vector with values
 * @sa <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @copydoc hide_code_mpi_evaluate3d
 */
template< class TernaryOp>
MPI_Vector<thrust::host_vector<double> > evaluate( const TernaryOp& f, const aMPITopology3d& g)
{
    //since the local grid is not binary compatible we have to use this implementation
    unsigned n = g.n();
    //abscissas
    Grid3d l = g.local();
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
    thrust::host_vector<double> absx( l.n()*l.Nx());
    thrust::host_vector<double> absy( l.n()*l.Ny());
    thrust::host_vector<double> absz(       l.Nz());
    for( unsigned i=0; i<l.Nx(); i++)
        for( unsigned j=0; j<n; j++)
        {
            unsigned coord = i+l.Nx()*coords[0];
            double xmiddle = DG_FMA( g.hx(), (double)(coord), g.x0());
            double h2 = g.hx()/2.;
            double absj = 1.+g.dlt().abscissas()[j];
            absx[i*n+j] = DG_FMA( h2, absj, xmiddle);
        }
    for( unsigned i=0; i<l.Ny(); i++)
        for( unsigned j=0; j<n; j++)
        {
            unsigned coord = i+l.Ny()*coords[1];
            double ymiddle = DG_FMA( g.hy(), (double)(coord), g.y0());
            double h2 = g.hy()/2.;
            double absj = 1.+g.dlt().abscissas()[j];
            absy[i*n+j] = DG_FMA( h2, absj, ymiddle );
        }
    for( unsigned i=0; i<l.Nz(); i++)
    {
        unsigned coord = i+l.Nz()*coords[2];
        double zmiddle = DG_FMA( g.hz(), (double)(coord), g.z0());
        double h2 = g.hz()/2.;
        absz[i] = DG_FMA( h2, (1.), zmiddle );
    }

    thrust::host_vector<double> w( l.size());
    for( unsigned s=0; s<l.Nz(); s++)
        for( unsigned i=0; i<l.Ny(); i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned j=0; j<l.Nx(); j++)
                    for( unsigned r=0; r<n; r++)
                        w[ (((s*l.Ny()+i)*n+k)*l.Nx() + j)*n + r] = f( absx[j*n+r], absy[i*n+k], absz[s]);
    MPI_Vector<thrust::host_vector<double> > v( w, g.communicator());
    return v;
};
///@cond
MPI_Vector<thrust::host_vector<double> > evaluate( double(f)(double, double, double), const aMPITopology3d& g)
{
    return evaluate<double(double, double, double)>( *f, g);
};
///@endcond
//
///@}

/**
 * @brief Take the relevant local part of a global vector
 *
 * @param global a vector the size of the global grid
 * @param g the assumed topology
 * @return an MPI_Vector that is the distributed version of the global vector
 * @ingroup scatter
 */
MPI_Vector<thrust::host_vector<double> > global2local( const thrust::host_vector<double>& global, const aMPITopology3d& g)
{
    assert( global.size() == g.global().size());
    Grid3d l = g.local();
    thrust::host_vector<double> temp(l.size());
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
    for( unsigned s=0; s<l.Nz(); s++)
        //for( unsigned py=0; py<dims[1]; py++)
            for( unsigned i=0; i<l.n()*l.Ny(); i++)
                //for( unsigned px=0; px<dims[0]; px++)
                    for( unsigned j=0; j<l.n()*l.Nx(); j++)
                    {
                        unsigned idx1 = (s*l.n()*l.Ny()+i)*l.n()*l.Nx() + j;
                        unsigned idx2 = (((s*dims[1]+coords[1])*l.n()*l.Ny()+i)*dims[0] + coords[0])*l.n()*l.Nx() + j;
                        temp[idx1] = global[idx2];
                    }
    return MPI_Vector<thrust::host_vector<double> >(temp, g.communicator());
}
/**
 * @copydoc global2local
 * @ingroup scatter
 */
MPI_Vector<thrust::host_vector<double> > global2local( const thrust::host_vector<double>& global, const aMPITopology2d& g)
{
    assert( global.size() == g.global().size());
    Grid2d l = g.local();
    thrust::host_vector<double> temp(l.size());
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
    //for( unsigned py=0; py<dims[1]; py++)
        for( unsigned i=0; i<l.n()*l.Ny(); i++)
            //for( unsigned px=0; px<dims[0]; px++)
                for( unsigned j=0; j<l.n()*l.Nx(); j++)
                {
                    unsigned idx1 = i*l.n()*l.Nx() + j;
                    unsigned idx2 = ((coords[1]*l.n()*l.Ny()+i)*dims[0] + coords[0])*l.n()*l.Nx() + j;
                    temp[idx1] = global[idx2];
                }
    return MPI_Vector<thrust::host_vector<double> >(temp, g.communicator());
}

}//namespace dg
