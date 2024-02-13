#pragma once

#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#include "evaluation.h"

/*! @file
  @brief Function discretization routines for mpi vectors
  */
namespace dg
{
///@cond
namespace create
{
template<class real_type>
thrust::host_vector<real_type> mpi_abscissas( const RealGrid1d<real_type>& l, real_type global_x0, int mpi_coord)
{
    thrust::host_vector<real_type> abs(l.size());
    for( unsigned i=0; i<l.N(); i++)
        for( unsigned j=0; j<l.n(); j++)
        {
            unsigned coord = i+l.N()*mpi_coord;
            real_type xmiddle = DG_FMA( l.h(), (real_type)(coord), global_x0);
            real_type h2 = l.h()/2.;
            real_type absj = 1.+l.dlt().abscissas()[j];
            abs[i*l.n()+j] = DG_FMA( h2, absj, xmiddle);
        }
    return abs;
}
}//
///@endcond

/**
 * @class hide_mpi_evaluate
 * @note In the MPI version all processes in the grid communicator need to call
 * this function. Each process evaluates the function f only on the grid
 * coordinates that it owns i.e. the local part of the given grid
 */

///@addtogroup evaluation
///@{
//
/**
 * @brief %Evaluate a 1d function on mpi distributed grid coordinates
 *
 * %Evaluate is equivalent to the following:
 *
 * -# generate the list of grid coordinates \f$ x_i\f$ representing the local part of the given computational space discretization (the grid) that the calling process owns
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i)\f$ for all \f$ i\f$
 *.
 * @copydoc hide_mpi_evaluate
 * @copydoc hide_code_mpi_evaluate1d
 * @tparam UnaryOp Model of Unary Function <tt> real_type f(real_type) </tt>
 * @param f The function to evaluate: f = f(x)
 * @param g The 1d grid on which to evaluate \c f
 *
 * @return The output vector \c v as an MPI host Vector
 * @note Use the elementary function \f$ f(x) = x \f$ (\c dg::cooX1d) to generate the list of grid coordinates in \c x direction
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template< class UnaryOp,class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( const UnaryOp& f, const RealMPIGrid1d<real_type>& g)
{
    //since the local grid is not binary compatible we have to use this implementation
    RealGrid1d<real_type> l = g.local();
    int dims[1], periods[1], coords[1];
    MPI_Cart_get( g.communicator(), 1, dims, periods, coords);
    thrust::host_vector<real_type> absx( create::mpi_abscissas( l, g.x0(), coords[0]));

    thrust::host_vector<real_type> w( l.size());
    for( unsigned j=0; j<l.N(); j++)
    for( unsigned r=0; r<l.n(); r++)
        w[ j*l.n() + r] = f( absx[j*l.n()+r]);
    MPI_Vector<thrust::host_vector<real_type> > v( w, g.communicator());
    return v;
};
///@cond
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( real_type(f)(real_type, real_type), const RealMPIGrid1d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type)>( *f, g);
};
///@endcond

/**
 * @brief %Evaluate a 2d function on mpi distributed grid coordinates
 *
 * %Evaluate is equivalent to the following:
 *
 * -# generate the list of grid coordinates \f$ x_i\f$, \f$ y_i\f$ representing the local part of the given computational space discretization (the grid) that the calling process owns
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i, y_i)\f$ for all \f$ i\f$
 *.
 * @copydoc hide_mpi_evaluate
 * @copydoc hide_code_mpi_evaluate2d
 * @copydoc hide_binary
 * @param f The function to evaluate: f = f(x,y)
 * @param g The 2d grid on which to evaluate \c f
 *
 * @return The output vector \c v as an MPI host Vector
 * @note Use the elementary function \f$ f(x,y) = x \f$ (\c dg::cooX2d) to generate the list of grid coordinates in \c x direction (or analogous in \c y, \c dg::cooY2d)
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template< class BinaryOp,class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( const BinaryOp& f, const aRealMPITopology2d<real_type>& g)
{
    //since the local grid is not binary compatible we have to use this implementation
    RealGrid2d<real_type> l = g.local();
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
    thrust::host_vector<real_type> absx( create::mpi_abscissas( l.gx(), g.x0(), coords[0]));
    thrust::host_vector<real_type> absy( create::mpi_abscissas( l.gy(), g.y0(), coords[1]));

    thrust::host_vector<real_type> w( l.size());
    for( unsigned i=0; i<l.Ny(); i++)
    for( unsigned k=0; k<l.ny(); k++)
    for( unsigned j=0; j<l.Nx(); j++)
    for( unsigned r=0; r<l.nx(); r++)
        w[ ((i*l.ny()+k)*l.Nx() + j)*l.nx() + r] = f( absx[j*l.nx()+r], absy[i*l.ny()+k]);
    MPI_Vector<thrust::host_vector<real_type> > v( w, g.communicator());
    return v;
};
///@cond
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( real_type(f)(real_type, real_type), const aRealMPITopology2d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type)>( *f, g);
};
///@endcond

/**
 * @brief %Evaluate a 3d function on mpi distributed grid coordinates
 *
 * %Evaluate is equivalent to the following:
 *
 * -# generate the list of grid coordinates \f$ x_i\f$, \f$ y_i\f$, \f$ z_i \f$ representing the local part of the given computational space discretization (the grid) that the calling process owns
 * -# evaluate the given function or functor at these coordinates and store the result
 *   in the output vector \f$ v_i = f(x_i, y_i, z_i)\f$ for all \f$ i\f$
 *.
 * @copydoc hide_mpi_evaluate
 * @copydoc hide_code_mpi_evaluate3d
 * @copydoc hide_ternary
 * @param f The function to evaluate: f = f(x,y,z)
 * @param g The 3d grid on which to evaluate \c f
 *
 * @return The output vector \c v as an MPI host Vector
 * @note Use the elementary function \f$ f(x,y,z) = x \f$ (\c dg::cooX3d) to generate the list of grid coordinates in \c x direction (or analogous in \c y, \c dg::cooY3d or \c z, \c dg::cooZ3d)
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 */
template< class TernaryOp,class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( const TernaryOp& f, const aRealMPITopology3d<real_type>& g)
{
    //since the local grid is not binary compatible we have to use this implementation
    //abscissas
    RealGrid3d<real_type> l = g.local();
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
    thrust::host_vector<real_type> absx( create::mpi_abscissas( l.gx(), g.x0(), coords[0]));
    thrust::host_vector<real_type> absy( create::mpi_abscissas( l.gy(), g.y0(), coords[1]));
    thrust::host_vector<real_type> absz( create::mpi_abscissas( l.gz(), g.z0(), coords[2]));

    thrust::host_vector<real_type> w( l.size());
    for( unsigned s=0; s<l.Nz(); s++)
    for( unsigned m=0; m<l.nz(); m++)
    for( unsigned i=0; i<l.Ny(); i++)
    for( unsigned k=0; k<l.ny(); k++)
    for( unsigned j=0; j<l.Nx(); j++)
    for( unsigned r=0; r<l.nx(); r++)
        w[ ((((s*l.nz()+m)*l.Ny()+i)*l.ny()+k)*l.Nx() + j)*l.nx() + r] =
            f( absx[j*l.nx()+r], absy[i*l.ny()+k], absz[s*l.nz()+m]);
    MPI_Vector<thrust::host_vector<real_type> > v( w, g.communicator());
    return v;
};
///@cond
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( real_type(f)(real_type, real_type, real_type), const aRealMPITopology3d<real_type>& g)
{
    return evaluate<real_type(real_type, real_type, real_type)>( *f, g);
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
template<class real_type, class MPITopology>
MPI_Vector<thrust::host_vector<real_type> > global2local( const thrust::host_vector<real_type>& global, const MPITopology& g)
{
    assert( global.size() == g.global().size());
    auto l = g.local();
    thrust::host_vector<real_type> temp(l.size());

    int dims[g.ndim()], periods[g.ndim()], coords[g.ndim()];
    MPI_Cart_get( g.communicator(), g.ndim(), dims, periods, coords);
    std::vector<unsigned> shape = dg::shape(l);
    // an exercise in flattening and unflattening indices
    for( unsigned idx = 0; idx<l.size(); idx++)
    {
        // idx = i[0] + s[0]*( i[1] + s[1]*(i[2]+s[2]*(...)))
        // convert flattened index to indices
        size_t i[g.ndim()], rest = idx;
        for( unsigned d=0; d<g.ndim(); d++)
        {
            i[d] = rest%shape[d];
            rest = rest/shape[d];
        }
        size_t idxx = 0;
        // convert to
        for( unsigned d=0; d<g.ndim(); d++)
        {
            // we need to construct from inside
            unsigned dd = g.ndim()-1-d;
            // 2 for loops e.g.
            //for( unsigned pz=0; pz<dims[2]; pz++)
            //for( unsigned s=0; s<shape[2]; s++)
            idxx = (idxx*dims[dd] + coords[dd])*shape[dd]+i[dd];
        }
        temp[idx] = global[idxx];
    }
    return MPI_Vector<thrust::host_vector<real_type> >(temp, g.communicator());
}

}//namespace dg

