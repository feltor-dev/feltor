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
 * @brief Class for the evaluation of a parallel derivative (MPI Version)
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
    * @tparam Field The Fieldlines to be integrated: Has to provide void operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
    is a limiter and 0 if there isn't. If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply. 
    * @param field The field to integrate
    * @param grid The grid on which to operate
    * @param deltaPhi Must either equal the hz() value of the grid or a fictive deltaPhi if the grid is 2D and Nz=1
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, note that if bcz is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on box
    * @note If there is a limiter, the boundary condition is set by the bcz variable from the grid and can be changed by the set_boundaries function. If there is no limiter the boundary condition is periodic.
    */
    template <class Field, class Limiter>
    DZ(Field field, const dg::MPI_Grid3d& grid, double deltaPhi, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR );

        /**
    * @brief Apply the derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void forward(  const MPI_Vector& f, MPI_Vector& dzf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void backward( const MPI_Vector& f, MPI_Vector& dzf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void centered( const MPI_Vector& f, MPI_Vector& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void forwardTD( const MPI_Vector& f, MPI_Vector& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void backwardTD( const MPI_Vector& f, MPI_Vector& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void centeredTD( const MPI_Vector& f, MPI_Vector& dzf);
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
    MPI_Vector evaluate( BinaryOp f, unsigned plane=0);
    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to 
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * The second functor is used to scale the values along the fieldlines.
     * The fieldlines are assumed to be periodic.
     * @tparam BinaryOp Binary Functor 
     * @tparam UnaryOp Unary Functor 
     * @param f Functor to evaluate in x-y
     * @param g Functor to evaluate in z
     * @param p0 The number of the plane to start
     * @param rounds The number of rounds to follow a fieldline
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp, class UnaryOp>
    MPI_Vector evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds);

  private:
    void einsPlus( const MPI_Vector& n, thrust::host_vector<double>& npe);
    void einsMinus( const MPI_Vector& n, thrust::host_vector<double>& nme);
    typedef cusp::array1d_view< thrust::host_vector<double>::iterator> View;
    typedef cusp::array1d_view< thrust::host_vector<double>::const_iterator> cView;
    double eps_;
    thrust::host_vector<double> hz, hp, hm, tempP, temp0, tempM, interP, interM;
    thrust::host_vector<double> ghostM, ghostP;
    MPI_Vector invB;
    MPI_Grid3d g_;
    dg::bc bcz_;
    thrust::host_vector<double> left_, right_;
    thrust::host_vector<double> limiter_;
    cusp::csr_matrix<int, double, cusp::host_memory> plus, minus; //interpolation matrices
    Collective collM_, collP_;

    dg::DZ<cusp::csr_matrix<int, double, cusp::host_memory>, thrust::host_vector<double> > dz_;
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
///@cond

template <class Field, class Limiter>
DZ<MPI_Matrix, MPI_Vector>::DZ(Field field, const dg::MPI_Grid3d& grid, double deltaPhi, double eps, Limiter limit, dg::bc globalbcz ): 
    eps_(eps),
    hz( grid.size()), hp(hz), hm(hz), tempP( grid.size()), temp0(tempP), tempM( tempP), interP(tempP), interM(tempP), g_(grid), bcz_(grid.bcz()),  invB(dg::evaluate(field,grid)) ,
    dz_(field, grid.global(), deltaPhi, eps, limit, globalbcz)
{
    assert( deltaPhi == grid.hz() || grid.Nz() == 1);
    //2D local grid with ghostcells
    dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    dg::Grid3d<double> global( grid.global()); //global without ghostcells
    dg::Grid3d<double> globalWG( global.x0() - global.hx()*(1-1e-14), global.x1() + global.hx()*(1-1e-14),
                                 global.y0() - global.hy()*(1-1e-14), global.y1() + global.hy()*(1-1e-14),
                                 global.z0(), global.z1(), 
                                 global.n(), global.Nx()+2, global.Ny()+2, global.Nz()); 
    //global with ghost-boundary for boxintegrator
    limiter_ = dg::evaluate( limit, g2d);
    right_ = left_ = dg::evaluate( zero, g2d);
    ghostM.resize( g2d.size()); ghostP.resize( g2d.size());
    //set up grid points as start for fieldline integrations (but not the ghostcells)
    std::vector<dg::HVec> y( 3);
    y[0] = dg::evaluate( dg::coo1, grid.local());
    y[1] = dg::evaluate( dg::coo2, grid.local());
    y[2] = dg::evaluate( dg::zero, grid.local());//distance (not angle)
    //integrate to next z-planes
    std::vector<dg::HVec> yp(y), ym(y); 
    thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
    for( unsigned i=0; i<grid.size(); i++)
    {
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i];
        double phi1 = deltaPhi;
        boxintegrator( field, globalWG, coords, coordsP, phi1, eps, globalbcz);
        phi1 = -deltaPhi;
        boxintegrator( field, globalWG, coords, coordsM, phi1, eps, globalbcz);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }


    //determine pid of result 
    thrust::host_vector<int> pids( grid.size());
    thrust::host_vector<double> angle = dg::evaluate( dg::coo3, grid.local());
    for( unsigned i=0; i<pids.size(); i++)
    {
        angle[i] += deltaPhi;
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
    

    //do the same for the minus z-plane
    for( unsigned i=0; i<pids.size(); i++)
    {
        angle[i] -= 2.*deltaPhi;
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
void DZ<MPI_Matrix, MPI_Vector>::centered( const MPI_Vector& f, MPI_Vector& dzf)
{
    //direct discretisation
    assert( &f != &dzf);
    einsPlus( f, tempP);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, hz, dzf.data());
}
void DZ<MPI_Matrix, MPI_Vector>::centeredTD(const MPI_Vector& f, MPI_Vector& dzf)
{       
//     Direct discretisation
       assert( &f != &dzf);    
        dg::blas1::pointwiseDot( f.data(), invB.data(),  dzf.data());
        einsPlus(  dzf, tempP);
        einsMinus(  dzf , tempM);
        dg::blas1::axpby( 1., tempP, -1., tempM);
        dg::blas1::pointwiseDivide( tempM, hz,  tempM );        
        dg::blas1::pointwiseDivide(  tempM, invB.data(), dzf.data() );
}
void DZ<MPI_Matrix, MPI_Vector>::forward( const MPI_Vector& f, MPI_Vector& dzf)
{
    //direct
    assert( &f != &dzf);
    einsPlus( f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f.data(), tempP);
    dg::blas1::pointwiseDivide( tempP, hp, dzf.data() );
}
void DZ<MPI_Matrix, MPI_Vector>::forwardTD(const MPI_Vector& f, MPI_Vector& dzf)
{
    //direct discretisation
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( f.data(), invB.data(),   dzf.data());
    einsMinus(  dzf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dzf.data(),tempP);
    dg::blas1::pointwiseDivide(  tempP, hm,   tempP);        
    dg::blas1::pointwiseDivide( tempP, invB.data(),  dzf.data());
}
void DZ<MPI_Matrix, MPI_Vector>::backward( const MPI_Vector& f, MPI_Vector& dzf)
{
    //direct
    assert( &f != &dzf);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f.data(), tempM);
    dg::blas1::pointwiseDivide( tempM, hm, dzf.data());
}

void DZ<MPI_Matrix, MPI_Vector>::backwardTD( const MPI_Vector& f, MPI_Vector& dzf)
{
    //direct
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( f.data(), invB.data(), dzf.data());
    einsPlus(  dzf, tempM);
    dg::blas1::axpby( -1., tempM, 1.,  dzf.data(), tempM);
    dg::blas1::pointwiseDivide(  tempM, hp,  tempM);        
    dg::blas1::pointwiseDivide( tempM, invB.data(), dzf.data());
}
void DZ<MPI_Matrix, MPI_Vector>::operator()( const MPI_Vector& f, MPI_Vector& dzf)
{
    assert( &f != &dzf);
    einsPlus( f, tempP); 
    einsMinus( f, tempM); 
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, hz, dzf.data() );
}    

void DZ<MPI_Matrix, MPI_Vector>::dzz( const MPI_Vector& f, MPI_Vector& dzzf)
{
    assert( &f != &dzzf);
    einsPlus( f, tempP); 
    einsMinus( f, tempM); 
    dg::blas1::pointwiseDivide( tempP, hp, tempP);
    dg::blas1::pointwiseDivide( tempP, hz, tempP);
    dg::blas1::pointwiseDivide( f.data(), hp, temp0);
    dg::blas1::pointwiseDivide( temp0, hm, temp0);
    dg::blas1::pointwiseDivide( tempM, hm, tempM);
    dg::blas1::pointwiseDivide( tempM, hz, tempM);
    dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
    dg::blas1::axpby( -2., temp0, +1., tempM, dzzf.data()); 
}

template< class BinaryOp>
MPI_Vector DZ<MPI_Matrix,MPI_Vector>::evaluate( BinaryOp f, unsigned p0)
{
    //let all processes integrate the fieldlines
    thrust::host_vector<double> global_vec = dz_.evaluate( f, p0);
    MPI_Vector mpi_vec( g_.n(), g_.Nx(), g_.Ny(), g_.Nz(), g_.communicator());
    thrust::host_vector<double> vec = mpi_vec.cut_overlap();
    //now take the relevant part 
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
    unsigned Nx = (g_.Nx()-2)*g_.n(), Ny = (g_.Ny()-2)*g_.n(), Nz = g_.Nz();
    for( unsigned s=0; s<Nz; s++)
        for( unsigned i=0; i<Ny; i++)
            for( unsigned j=0; j<Nx; j++)
                vec[ (s*Ny+i)*Nx + j ] 
                    = global_vec[ j + Nx*(coords[0] + dims[0]*( i +Ny*(coords[1] + dims[1]*(s +Nz*coords[2])))) ];
    mpi_vec.copy_into_interior( vec);
    return mpi_vec;
}

template< class BinaryOp, class UnaryOp>
MPI_Vector DZ<MPI_Matrix,MPI_Vector>::evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds)
{
    //let all processes integrate the fieldlines
    thrust::host_vector<double> global_vec = dz_.evaluate( f, g,p0, rounds);
    MPI_Vector mpi_vec( g_.n(), g_.Nx(), g_.Ny(), g_.Nz(), g_.communicator());
    thrust::host_vector<double> vec = mpi_vec.cut_overlap();
    //now take the relevant part 
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
    unsigned Nx = (g_.Nx()-2)*g_.n(), Ny = (g_.Ny()-2)*g_.n(), Nz = g_.Nz();
    for( unsigned s=0; s<Nz; s++)
        for( unsigned i=0; i<Ny; i++)
            for( unsigned j=0; j<Nx; j++)
                vec[ (s*Ny+i)*Nx + j ] 
                    = global_vec[ j + Nx*(coords[0] + dims[0]*( i +Ny*(coords[1] + dims[1]*(s +Nz*coords[2])))) ];
    mpi_vec.copy_into_interior( vec);
    return mpi_vec;
}

void DZ<MPI_Matrix, MPI_Vector>::einsPlus( const MPI_Vector& f, thrust::host_vector<double>& fplus ) 
{
    const thrust::host_vector<double>& in = f.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    cView fv( in.cbegin(), in.cend());

    View P( interP.begin(), interP.end() );
    cusp::multiply( plus, fv, P); //interpolate input vector 
    //gather results from all processes
    collP_.gather( interP, fplus);
    //make ghostcells in last plane
    if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View tempPV( fplus.begin() + i0*size, fplus.begin() + (i0+1)*size);
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

}

void DZ<MPI_Matrix, MPI_Vector>::einsMinus( const MPI_Vector& f, thrust::host_vector<double>& fminus ) 
{
    const thrust::host_vector<double>& in = f.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    cView fv( in.cbegin(), in.cend());

    View M( interM.begin(), interM.end() );
    cusp::multiply( minus, fv, M);
    //gather results from all processes
    collM_.gather( interM, fminus); 
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View tempMV( fminus.begin() + i0*size, fminus.begin() + (i0+1)*size);
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
}
///@endcond
}//namespace dg

