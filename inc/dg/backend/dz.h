#pragma once

#include "dz.cuh"
#include "grid.h"
#include "mpi_matrix.h"
#include "mpi_collective.h"
#include "mpi_grid.h"
#include "interpolation.cuh"
#include "typedefs.cuh"
#include "../runge_kutta.h"

namespace dg{

///@cond
namespace detail{
double oneR( double R, double Z, double phi){return R;}
double oneZ( double R, double Z, double phi){return Z;}
double zero( double R, double Z, double phi){return 0;}
double  phi( double R, double Z, double phi){return phi;}
} //namespace detail
///@endcond

/**
 * @brief Class for the evaluation of a parallel derivative
 *
 * @ingroup dz
 * @tparam Matrix The matrix class of the interpolation matrix
 * @tparam container The container-class to on which the interpolation matrix operates on (does not need to be dg::HVec)
 */
template< >
struct DZ< MPI_Matrix, MPI_Vector> 
{
    /**
     * @brief Construct from a field and a grid
     *
     * @tparam Field The Fieldlines to be integrated: Has to provide void  operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
     * @param field The field to integrate
     * @param grid The grid on which to operate
     * @param eps Desired accuracy of fieldline integration
     */
    template <class Field>
    DZ(Field field, const dg::MPI_Grid3d& grid, double eps = 1e-3): 
        hz( grid.size()), tempP( grid.size()), tempM( grid.size())
    {
        //set up grid points as start for fieldline integrations
        std::vector<dg::HVec> y( 3);
        y[0] = dg::evaluate( detail::oneR, grid.local());
        y[1] = dg::evaluate( detail::oneZ, grid.local());
        y[2] = dg::evaluate( detail::zero, grid.local());//distance (not angle)
        //integrate to next z-plane
        std::vector<dg::HVec> yp(y), ym(y); 
        dg::integrateRK4( field, y, yp,  grid.hz(), eps);
        cut( y, yp, grid.global() ); //cut points 
        //determine pid of result 
        thrust::host_vector<int> pids( grid.size());
        thrust::host_vector<double> angle = dg::evaluate( detail::phi, grid.local());
        for( unsigned i=0; i<pids.size(); i++)
        {
            angle[i] += grid.hz();
            if( angle[i] >= grid.global().z1()) angle[i] -= grid.global().lz();
            pids[i]  = grid.pidOf( yp[0][i], yp[1][i], angle[i]);
            if( pids[i]  == -1)
            {
                std::cerr << "ERROR: PID NOT FOUND!\n";
                return;
            }
        }
        //construct scatter operation from pids
        Collective cp( pids, grid.communicator());
        collP_ = cp;
        thrust::host_vector<double> pX = collP_.scatter( yp[0]),
                                    pY = collP_.scatter( yp[1]),
                                    pZ = collP_.scatter( angle);
        //construt interpolation matrix
        plus  = dg::create::interpolation( pX, pY, pZ, grid.local());
        

        //do the same for the previous z-plane
        dg::integrateRK4( field, y, ym, -grid.hz(), eps);
        cut( y, ym, grid.global() );
        for( unsigned i=0; i<pids.size(); i++)
        {
            angle[i] -= 2.*grid.hz();
            if( angle[i] <= grid.global().z0()) angle[i] += grid.global().lz();
            pids[i]  = grid.pidOf( ym[0][i], ym[1][i], angle[i]);
            if( pids[i] == -1)
            {
                std::cerr << "ERROR: PID NOT FOUND!\n";
                return;
            }
        }
        Collective cm( pids, grid.communicator());
        collM_ = cm;
        pX = collM_.scatter( ym[0]),
        pY = collM_.scatter( ym[1]),
        pZ = collM_.scatter( angle);
        minus = dg::create::interpolation( pX, pY, pZ, grid.local());
        dg::blas1::axpby( 1., yp[2], -1., ym[2], hz);

        interM.resize( collM_.recv_size());
        interP.resize( collP_.recv_size());
    }

    /**
     * @brief Apply the derivative on a 3d vector
     *
     * @param f The vector to derive
     * @param dzf contains result on output (write only)
     */
    void operator()( const MPI_Vector& f, MPI_Vector& dzf)
    {
        assert( &f != &dzf);
        cusp::array1d_view< typename thrust::host_vector<double>::const_iterator> 
            fv( f.data().cbegin(), f.data().cend());
        cusp::array1d_view< typename thrust::host_vector<double>::iterator> 
            P( interP.begin(), interP.end() );
        cusp::multiply( plus, fv, P); //interpolate input vector 

        cusp::array1d_view< typename thrust::host_vector<double>::iterator> 
            M( interM.begin(), interM.end() );
        cusp::multiply( minus, fv, M);
        //gather results from all processes
        collM_.gather( interM, tempM); 
        collP_.gather( interP, tempP);
        //compute finite difference formula
        dg::blas1::axpby( 1., tempP, -1., tempM);
        dg::blas1::pointwiseDivide( tempM, hz, dzf.data());
    }
  private:
    void cut( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp, const dg::Grid3d<double>& g) //global grid
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {            
            if      (yp[0][i] < g.x0()-g.hx()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else if (yp[0][i] > g.x1()+g.hx()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else if (yp[1][i] < g.y0()-g.hy()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else if (yp[1][i] > g.y1()+g.hy()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else                         { }
        }
    }
    thrust::host_vector<double> hz, tempP, tempM, interP, interM;
    cusp::csr_matrix<int, double, cusp::host_memory> plus, minus; //interpolation matrices
    Collective collM_, collP_;

};
}//namespace dg

