#pragma once

#include "dz.cuh"
#include "grid.h"
#include "mpi_matrix.h"
#include "mpi_collective.h"
#include "mpi_grid.h"
#include "interpolation.cuh"
#include "typedefs.cuh"
#include "functions.h"
#include "../runge_kutta.h"

namespace dg{


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
     * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there 
     is a limiter and 0 if there isn't
     * @param field The field to integrate
     * @param grid The grid on which to operate
     * @param eps Desired accuracy of fieldline integration
     * @param limit Instance of the Limiter class
     */
    template <class Field, class Limiter>
    DZ(Field field, const dg::MPI_Grid3d& grid, double eps = 1e-3, Limiter limit = DefaultLimiter()): eps_(eps),
        hz( grid.size()), hp(hz), hm(hz), tempP( grid.size()), temp0(tempP), tempM( tempP), interP(tempP), interM(tempP), ghostM( tempP), ghostP(tempP), g_(grid), bcz_(grid.bcz())
    {
        dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        limiter_ = dg::evaluate( limit, g2d);
        left_ = dg::evaluate( zero, g2d);
        right_ = left_;
        ghostM.resize( g2d.size());
        ghostP.resize( g2d.size());
        //set up grid points as start for fieldline integrations
        std::vector<dg::HVec> y( 3);
        y[0] = dg::evaluate( dg::coo1, grid.local());
        y[1] = dg::evaluate( dg::coo2, grid.local());
        y[2] = dg::evaluate( dg::zero, grid.local());//distance (not angle)
        //integrate to next z-plane
        std::vector<dg::HVec> yp(y), ym(y); 
        dg::integrateRK4( field, y, yp,  grid.hz(), eps);
        cut( y, yp, grid.global() ); //cut points 
        //determine pid of result 
        thrust::host_vector<int> pids( grid.size());
        thrust::host_vector<double> angle = dg::evaluate( dg::coo3, grid.local());
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
        dg::blas1::axpby(  1., yp[2], 0, hp);
        dg::blas1::axpby( -1., ym[2], 0, hm);
        dg::blas1::axpby(  1., hp, +1., hm, hz);

        interM.resize( collM_.recv_size());
        interP.resize( collP_.recv_size());
    }

    /**
     * @brief Apply the derivative on a 3d vector
     *
     * @param f The vector to derive
     * @param dzf contains result on output (write only)
     */
    void operator()( const MPI_Vector& f, MPI_Vector& dzf);

    /**
     * @brief Set boundary conditions
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param left left boundary value 
     * @param right right boundary value
     */
    void set_boundaries( dg::bc bcz, double left, double right)
    {
        bcz_ = bcz; 
        const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        left_  = dg::evaluate( dg::CONSTANT(left), g2d);
        right_ = dg::evaluate( dg::CONSTANT(right),g2d);
    }

    /**
     * @brief Set boundary conditions
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param left left boundary value 
     * @param right right boundary value
     */
    void set_boundaries( dg::bc bcz, const thrust::host_vector<double>& left, const thrust::host_vector<double>& right)
    {
        bcz_ = bcz; 
        left_ = left;
        right_ = right;
    }

    /**
     * @brief Compute the second derivative using finite differences
     *
     * @param f input function
     * @param dzzf output (write-only)
     */
    void dzz( const MPI_Vector& f, MPI_Vector& dzzf);

    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to 
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * @tparam BinaryOp Binary Functor 
     * @param f Functor to evaluate
     * @param plane The number of the plane to start
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp>
    container evaluate( BinaryOp f, unsigned plane=0);

  private:
    typedef cusp::array1d_view< thrust::host_vector<double>::iterator> View;
    typedef cusp::array1d_view< thrust::host_vector<double>::const_iterator> cView;
    void cut( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp, const dg::Grid3d<double>& g) //global grid
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {            
            if      (yp[0][i] < g.x0()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else if (yp[0][i] > g.x1()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else if (yp[1][i] < g.y0()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else if (yp[1][i] > g.y1()) { yp[0][i]=y[0][i]; yp[1][i]=y[1][i]; }
            else                         { }
        }
        //yp can still be outside the global grid (ghostcells!)
    }
    double eps_;
    thrust::host_vector<double> hz, hp, hm, tempP, temp0, tempM, interP, interM;
    thrust::host_vector<double> ghostM, ghostP;
    MPI_Grid3d g_;
    dg::bc bcz_;
    thrust::host_vector<double> left_, right_;
    thrust::host_vector<double> limiter_;
    cusp::csr_matrix<int, double, cusp::host_memory> plus, minus; //interpolation matrices
    Collective collM_, collP_;

};

void DZ<MPI_Matrix, MPI_Vector>::operator()( const MPI_Vector& f, MPI_Vector& dzf)
{
    assert( &f != &dzf);
    const thrust::host_vector<double>& in = f.data();
    thrust::host_vector<double>& out = dzf.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();

    cView fv( in.cbegin(), in.cend());
    View P( interP.begin(), interP.end() );
    cusp::multiply( plus, fv, P); //interpolate input vector 
    View M( interM.begin(), interM.end() );
    cusp::multiply( minus, fv, M);
    //gather results from all processes
    collM_.gather( interM, tempM); 
    collP_.gather( interP, tempP);
    //make ghostcells
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View tempPV( tempP.begin() + i0*size, tempP.begin() + (i0+1)*size);
        View tempMV( tempM.begin() + i0*size, tempM.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite tempM
        cusp::copy( f0, ghostMV);
        if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::axpby( 2., left_, -1, ghostM);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::pointwiseDot( left_, hm, ghostP);
            dg::blas1::axpby( -1, ghostP, 1., ghostM);
        }
        cusp::blas::axpby(  ghostMV,  tempMV, ghostMV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
        cusp::blas::axpby(  ghostMV,  tempMV, tempMV, 1.,1.);

    }
    else if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View tempPV( tempP.begin() + i0*size, tempP.begin() + (i0+1)*size);
        View tempMV( tempM.begin() + i0*size, tempM.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite tempP
        cusp::copy( f0, ghostPV);
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::axpby( 2., right_, -1, ghostP);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::pointwiseDot( right_, hp, ghostM);
            dg::blas1::axpby( 1., ghostM, 1., ghostP);
        }
        cusp::blas::axpby(  ghostPV,  tempPV, ghostPV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
        cusp::blas::axpby(  ghostPV,  tempPV, tempPV, 1.,1.);
    }
    //compute finite difference formula
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, hz, dzf.data());
}    

void DZ<MPI_Matrix, MPI_Vector>::dzz( const MPI_Vector& f, MPI_Vector& dzzf)
{
    assert( &f != &dzzf);

    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    const thrust::host_vector<double>& in = f.data();
    thrust::host_vector<double>& out = dzzf.data();

    cView fv( in.cbegin(), in.cend());
    View P( interP.begin(), interP.end() );
    cusp::multiply( plus, fv, P); //interpolate input vector 
    View M( interM.begin(), interM.end() );
    cusp::multiply( minus, fv, M);
    //gather results from all processes
    collM_.gather( interM, tempM); 
    collP_.gather( interP, tempP);
    //make ghostcells
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View tempPV( tempP.begin() + i0*size, tempP.begin() + (i0+1)*size);
        View tempMV( tempM.begin() + i0*size, tempM.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite tempM
        cusp::copy( f0, ghostMV);
        if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::axpby( 2., left_, -1, ghostM);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::pointwiseDot( left_, hm, ghostP);
            dg::blas1::axpby( -1, ghostP, 1., ghostM);
        }
        cusp::blas::axpby(  ghostMV,  tempMV, ghostMV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
        cusp::blas::axpby(  ghostMV,  tempMV, tempMV, 1.,1.);

    }
    else if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View tempPV( tempP.begin() + i0*size, tempP.begin() + (i0+1)*size);
        View tempMV( tempM.begin() + i0*size, tempM.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite tempP
        cusp::copy( f0, ghostPV);
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::axpby( 2., right_, -1, ghostP);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::pointwiseDot( right_, hp, ghostM);
            dg::blas1::axpby( 1., ghostM, 1., ghostP);
        }
        cusp::blas::axpby(  ghostPV,  tempPV, ghostPV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
        cusp::blas::axpby(  ghostPV,  tempPV, tempPV, 1.,1.);
    }

    {
        dg::blas1::pointwiseDivide( tempP, hp, tempP);
        dg::blas1::pointwiseDivide( tempP, hz, tempP);
        dg::blas1::pointwiseDivide( f.data(), hp, temp0);
        dg::blas1::pointwiseDivide( temp0, hm, temp0);
        dg::blas1::pointwiseDivide( tempM, hm, tempM);
        dg::blas1::pointwiseDivide( tempM, hz, tempM);
    }

    dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
    dg::blas1::axpby( -2., temp0, +1., tempM, dzzf.data()); 
    //View dzzf0( dzzf.begin() + i0*size, dzzf.begin() + (i0+1)*size);
    //cusp::copy( tempMV, dzzf0);
}

template< class BinaryOp>
MPI_Vector DZ<MPI_Matrix,MPI_Vector>::evaluate( BinaryOp f, unsigned p0)
{
    dg::DZ<cusp::csr_matrix<int, double, cusp::host_memory>, thrust::host_vector<double> > dz( g_.global(), eps_);
    thrust::host_vector<double> global_vec = dz.evaluate( f, p0);
    MPI_Vector mpi_vec( g_.n(), g_.Nx(), g_.Ny(), g_.Nz(), g_.comm());
    thrust::host_vector<double> vec = mpi_vec.cut_boundaries();
    //now take the relevant part 
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g_.comm(), 3, dims, periods, coords);
    unsigned Nx = (g_.Nx()-2)*g_.n(), Ny = (g_.Ny()-2)*g_.n(), Nz = g_.Nz();
    for( unsigned s=0; s<Nz; s++)
        for( unsigned i=0; i<Ny; i++)
            for( unsigned j=0; j<Nx; j++)
                vec[ (s*Ny+i)*Nx + j ] 
                    = global_vec[ j + Nx*(coords[0] + dims[0]*( i +Ny*(coords[1] + dims[1]*(s +Nz*coords[2])))) ];
    mpi_vec.copy_into_interior( vec);




}

}//namespace dg

