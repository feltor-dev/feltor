#pragma once

#include "fieldaligned.h"
#include "dg/backend/grid.h"
#include "dg/backend/mpi_evaluation.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/mpi_matrix_blas.h"
#include "dg/backend/mpi_collective.h"
#include "dg/backend/mpi_grid.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/functions.h"
#include "dg/runge_kutta.h"

namespace dg{
 
///@cond
namespace detail{

///basically a copy across processes
template<class InputIterator, class OutputIterator>
void sendForward( InputIterator begin, InputIterator end, OutputIterator result, MPI_Comm comm) //send to next plane
{
    int source, dest;
    MPI_Status status;
    MPI_Cart_shift( comm_, 2, +1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize();//wait until device functions are finished before sending data
#endif //THRUST_DEVICE_SYSTEM
    unsigned size = thrust::distance( begin, end);
    MPI_Sendrecv(   thrust::raw_pointer_cast(begin), size, MPI_DOUBLE,  //sender
                    dest, 9,  //destination
                    thrust::raw_pointer_cast(result), size, MPI_DOUBLE, //receiver
                    source, 9, //source
                    comm, &status);
}
///basically a copy across processes
template<class InputIterator, class OutputIterator>
void sendBackward( InputIterator begin, InputIterator end, OutputIterator result, MPI_Comm comm) //send to next plane
{
    int source, dest;
    MPI_Status status;
    MPI_Cart_shift( comm_, 2, -1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize();//wait until device functions are finished before sending data
#endif //THRUST_DEVICE_SYSTEM
    unsigned size = thrust::distance( begin, end);
    MPI_Sendrecv(   thrust::raw_pointer_cast(begin), size, MPI_DOUBLE,  //sender
                    dest, 3,  //destination
                    thrust::raw_pointer_cast(result), size, MPI_DOUBLE, //receiver
                    source, 3, //source
                    comm, &status);
}

aGeometry2d* clone_MPI3d_to_global_perp( const aMPIGeometry3d* grid_ptr)
{
    const dg::CartesianMPIGrid3d* grid_cart = dynamic_cast<const dg::CartesianMPIGrid3d*>(grid_ptr);
    const dg::CylindricalMPIGrid3d* grid_cyl = dynamic_cast<const dg::CylindricalMPIGrid3d*>(grid_ptr);
    const dg::CurvilinearProductMPIGrid3d*  grid_curvi = dynamic_cast<const dg::CurvilinearProductMPIGrid3d*>(grid_ptr);
    aGeometry2d* g2d_ptr;
    if( grid_cart) 
    {
        dg::CartesianMPIGrid2d cart = grid_cart->perp_grid();
        dg::CartesianGrid2d global_cart( cart.global());
        g2d_ptr = global_cart.clone();
    }
    else if( grid_cyl) 
    {
        dg::CartesianMPIGrid2d cart = grid_cyl->perp_grid();
        dg::CartesianGrid2d global_cart( cart.global());
        g2d_ptr = global_cart.clone();
    }
    else if( grid_curvi) 
    {
        dg::geo::CurvilinearMPIGrid2d curv = grid_curvi->perp_grid();
        dg::geo::CurvilinearGrid2d global_curv( curv.generator(), curv.global().n(), curv.global().Nx(), curv.global().Ny(), curv.bcx(), curv.bcy());
        g2d_ptr = global_curv.clone();
    }
    else
        throw dg::Error( dg::Message(_ping_)<<"Grid class not recognized!");
    return g2d_ptr;
}
}//namespace detail

template <class Geometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
struct FieldAligned< Geometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> > 
{
    FieldAligned(){}
    template <class Limiter>
    FieldAligned(const dg::geo::BinaryVectorLvl0& vec, const Geometry& grid, unsigned multiplyX, unsigned multiplyY, double eps = 1e-4, Limiter limit = FullLimiter(), dg::bc globalbcx = dg::DIR, dg::bc globalbcy = dg::DIR, double deltaPhi = -1);

    void set_boundaries( dg::bc bcz, double left, double right)
    {
        bcz_ = bcz; 
        const dg::Grid2d g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        left_  = dg::evaluate( dg::CONSTANT(left), g2d);
        right_ = dg::evaluate( dg::CONSTANT(right),g2d);
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& left, const MPI_Vector<LocalContainer>& right)
    {
        bcz_ = bcz; 
        left_ = left.data();
        right_ = right.data();
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& global, double scal_left, double scal_right)
    {
        bcz_ = bcz;
        unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        if( g_.z0() == g_.global().z0())
        {
            cView left( global.data().cbegin(), global.data().cbegin() + size);
            View leftView( left_.begin(), left_.end());
            cusp::copy( left, leftView);
            dg::blas1::scal( left_, scal_left);
        }
        if( g_.z1() == g_.global().z1())
        {
            cView right( global.data().cbegin()+(g_.Nz()-1)*size, global.data().cbegin() + g_.Nz()*size);
            View rightView( right_.begin(), right_.end());
            cusp::copy( right, rightView);
            dg::blas1::scal( right_, scal_right);
        }
    }

    template< class BinaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, unsigned plane=0) const;

    template< class BinaryOp, class UnaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const;

    void operator()(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);

    const MPI_Vector<LocalContainer>& hz()const {return hz_;}
    const MPI_Vector<LocalContainer>& hp()const {return hp_;}
    const MPI_Vector<LocalContainer>& hm()const {return hm_;}
    const Geometry& grid() const{return g_;}
  private:
    typedef cusp::array1d_view< typename LocalContainer::iterator> View;
    typedef cusp::array1d_view< typename LocalContainer::const_iterator> cView;
    MPI_Vector<LocalContainer> hz_, hp_, hm_; 
    LocalContainer ghostM, ghostP;
    dg::Handle<Geometry> g_;
    dg::bc bcz_;
    LocalContainer left_, right_;
    LocalContainer limiter_;
    std::vector<LocalContainer> tempXYplus_, tempXYminus_, temp_; 
    CommunicatorXY commXYplus_, commXYminus_;
    LocalIMatrix plus, minus; //interpolation matrices
    LocalIMatrix plusT, minusT; //interpolation matrices
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
template <class Limiter>
FieldAligned<MPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> >::FieldAligned(
    const dg::geo::BinaryVectorLvl0& vec, const MPIGeometry& grid, unsigned mx, unsigned my, double eps, Limiter limit, dg::bc globalbcx, dg::bc globalbcy, double deltaPhi):
    hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
    g_(grid), bcz_(grid.bcz()), 
    tempXYplus_(g_.Nz()), tempXYminus_(g_.Nz()), temp_(g_.Nz())
{
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%downcast grid since we don't have a virtual function perp_grid%%%%%%%%%%%%%
    const aGeometry2d* g2d_ptr = detail::clone_MPI3d_to_global_perp(&grid);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    unsigned localsize = grid2d_ptr->size();
    limiter_ = dg::evaluate( limit, grid2d_ptr->local());
    right_ = left_ = dg::evaluate( zero, grid2d_ptr->local());
    ghostM.resize( localsize); ghostP.resize( localsize);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%%%%
    std::vector<thrust::host_vector<double> > yp_coarse( 3), ym_coarse(yp_coarse); 
    
    dg::aGeometry2d* g2dField_ptr = grid2d_ptr->clone();//INTEGRATE HIGH ORDER GRID
    g2dField_ptr->set( 7, g2dField_ptr->Nx(), g2dField_ptr->Ny());
    detail::integrate_all_fieldlines2d( vec, g2dField_ptr, yp_coarse, ym_coarse, deltaPhi, eps);
    delete g2dField_ptr;

    //determine pid of result 
    thrust::host_vector<int> pids( localsize);
    for( unsigned i=0; i<localsize; i++)
    {
        pids[i]  = grid2d_ptr.pidOf( yp[0][i], yp[1][i]);
        if( pids[i]  == -1)
        {
            std::cerr << "ERROR: PID NOT FOUND!\n";
            return;
        }
    }

    CommunicatorXY cp( pids, grid2d_ptr.communicator());
    commXYplus_ = cp;
    thrust::host_vector<double> pX, pY;
    dg::blas1::transfer( cp.global_gather( yp[0]), pX);
    dg::blas1::transfer( cp.global_gather( yp[1]), pY);

    //construt interpolation matrix
    plus = dg::create::interpolation( pX, pY, grid2d_ptr.local(), globalbcz); //inner points hopefully never lie exactly on local boundary
    cusp::transpose( plus, plusT);

    //do the same for the minus z-plane
    for( unsigned i=0; i<pids.size(); i++)
    {
        pids[i]  = grid2d_ptr.pidOf( ym[0][i], ym[1][i]);
        if( pids[i] == -1)
        {
            std::cerr << "ERROR: PID NOT FOUND!\n";
            return;
        }
    }
    CommunicatorXY cm( pids, grid2d_ptr.communicator());
    commXYminus_ = cm;
    dg::blas1::transfer( cm.global_gather( ym[0]), pX);
    dg::blas1::transfer( cm.global_gather( ym[1]), pY);
    minus = dg::create::interpolation( pX, pY, grid2d_ptr.local(), globalbcz); //inner points hopefully never lie exactly on local boundary
    cusp::transpose( minus, minusT);
    //copy to device
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        thrust::copy( yp[2].begin(), yp[2].end(), hp_.data().begin() + i*localsize);
        thrust::copy( ym[2].begin(), ym[2].end(), hm_.data().begin() + i*localsize);
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        tempXYplus_[i].resize( commXYplus_.size());
        tempXYminus_[i].resize( commXYminus_.size());
        temp_[i].resize( localsize);
    }
    delete grid2d_ptr;
}

template<class G, class M, class C, class container>
template< class BinaryOp>
MPI_Vector<container> FieldAligned<G,RowDistMat<M,C>,MPI_Vector<container> >::evaluate( BinaryOp binary, unsigned p0) const
{
    return evaluate( binary, dg::CONSTANT(1), p0, 0);
}

template<class G, class M, class C, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> FieldAligned<G,RowDistMat<M,C>, MPI_Vector<container> >::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( g_.Nz() > 1);
    assert( p0 < g_.global().Nz());
    const typename G::perpendicular_grid g2d = g_.perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, g2d); 
    container temp(init2d.data()), tempP(init2d.data()), tempM(init2d.data());
    MPI_Vector<container> vec3d = dg::evaluate( dg::zero, g_);
    std::vector<container>  plus2d( g_.global().Nz(), (container)dg::evaluate(dg::zero, g2d.local()) ), minus2d( plus2d), result( plus2d);
    container tXYplus( tempXYplus_[0]), tXYminus( tempXYminus_[0]);
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<g_.global().Nz(); i0++)
        {
            dg::blas1::copy( init2d.data(), tempP);
            dg::blas1::copy( init2d.data(), tempM);
            unsigned rep = i0 + r*g_.global().Nz(); 
            for(unsigned k=0; k<rep; k++)
            {
                if( sizeXY != 1){
                    dg::blas2::symv( plus, tempP, tXYplus);
                    commXYplus_.global_scatter_reduce( tXYplus, temp);
                }
                else
                    dg::blas2::symv( plus, tempP, temp);
                temp.swap( tempP);
                if( sizeXY != 1){
                    dg::blas2::symv( minus, tempM, tXYminus);
                    commXYminus_.global_scatter_reduce( tXYminus, temp);
                }
                else
                    dg::blas2::symv( minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*g_.hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*g_.hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<g_.Nz(); i0++)
        {
            int idx = (int)(i0+coords[2]*g_.Nz())  - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].begin(), result[i0].end(), vec3d.data().begin() + i0*g2d.size());
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<g_.global().Nz(); i0++)
        {
            //int idx = (int)(i0+coords[2]*g_.Nz());
            unsigned revi0 = (g_.global().Nz() - i0)%g_.global().Nz(); //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d.data(), 1., result[0]);
        for(unsigned i0=0; i0<g_.Nz(); i0++)
        {
            int idx = ((int)i0 + coords[2]*g_.Nz() -(int)p0 + g_.global().Nz())%g_.global().Nz(); //shift index
            thrust::copy( result[idx].begin(), result[idx].end(), vec3d.data().begin() + i0*g2d.size());
        }
    }
    return vec3d;
}

template<class G, class M, class C, class container>
void FieldAligned<G,RowDistMat<M,C>, MPI_Vector<container> >::einsPlus( const MPI_Vector<container>& f, MPI_Vector<container>& fplus ) 
{
    //dg::blas2::detail::doSymv( plus, f, fplus, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    const container& in = f.data();
    container& out = fplus.data();
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];

    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( tempXYplus_[i0].begin(), tempXYplus_[i0].end() );
            cusp::multiply( plus, inV, tempV);
            //exchange data in XY
            commXYplus_.global_scatter_reduce( tempXYplus_[i0], temp_[i0]);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( plus, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
    {
        int ip = i0 + 1;
        if( ip > (int)g_.Nz()-1) ip -= (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].begin() + size2d, out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        detail::sendBackward( temp_[0].begin(), temp_[0].end(), out.begin() + (g_.Nz()-1)*size2d);
    }

    //make ghostcells in last plane
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostPV);
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::axpby( 2., right_, -1, ghostP);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            //note that hp_ is 3d and the rest 2d
            thrust::transform( right_.begin(), right_.end(),  hp_.data().begin(), ghostM.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( 1., ghostM, 1., ghostP);
        }
        cusp::blas::axpby(  ghostPV,  outV, ghostPV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
        cusp::blas::axpby(  ghostPV,  outV, outV, 1.,1.);
    }

}

template<class G,class M, class C, class container>
void FieldAligned<G,RowDistMat<M,C>,MPI_Vector<container> >::einsMinus( const MPI_Vector<container>& f, MPI_Vector<container>& fminus ) 
{
    const container& in = f.data();
    container& out = fminus.data();
    //dg::blas2::detail::doSymv( minus, f, fminus, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];
    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minus.num_cols, in.cbegin() + (i0+1)*minus.num_cols);
            View tempV( tempXYminus_[i0].begin(), tempXYminus_[i0].end());
            cusp::multiply( minus, inV, tempV);
            //exchange data in XY
            commXYminus_.global_scatter_reduce( tempXYminus_[i0], temp_[i0]);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minus.num_cols, in.cbegin() + (i0+1)*minus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( minus, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
    {
        int ip = i0 -1;
        if( ip < 0) ip += (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].end(), out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        detail::sendForward( temp_[g_.Nz()-1].begin(), temp_[g_.Nz()-1].end(), out.begin());
    }
    //make ghostcells in first plane
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostMV);
        if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::axpby( 2., left_, -1, ghostM);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
        {
            thrust::transform( left_.begin(), left_.end(), hm_.data().begin(), ghostP.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( -1, ghostP, 1., ghostM);
        }
        cusp::blas::axpby(  ghostMV,  outV, ghostMV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
        cusp::blas::axpby(  ghostMV,  outV, outV, 1.,1.);
    }
}
template< class G, class M, class C, class container>
void FieldAligned<G,RowDistMat<M,C>,MPI_Vector<container> >::einsMinusT( const MPI_Vector<container>& f, MPI_Vector<container>& fpe)
{
    //dg::blas2::detail::doSymv( minusT, f, fpe, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    const container& in = f.data();
    container& out = fpe.data();
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];

    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        //first exchange data in XY
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            thrust::copy( in.cbegin() + i0*size2d, in.cbegin() + (i0+1)*size2d, temp_[i0].begin());
            tempXYminus_[i0] = commXYminus_.global_gather( temp_[i0] );
            cView inV( tempXYminus_[i0].cbegin(), tempXYminus_[i0].cend() );
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( minusT, inV, tempV);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minusT.num_cols, in.cbegin() + (i0+1)*minusT.num_cols);
            View tempV( temp_[i0].begin() , temp_[i0].end() );
            cusp::multiply( minusT, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
    {
        int ip = i0 + 1;
        if( ip > (int)g_.Nz()-1) ip -= (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].end(), out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        detail::sendBackward( temp_[0].begin(), temp_[0].end(), out.begin() + ( g_.Nz()-1)*size2d);
    }
    //make ghostcells in last plane
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostPV);
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::axpby( 2., right_, -1, ghostP);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            //note that hp_ is 3d and the rest 2d
            thrust::transform( right_.begin(), right_.end(),  hp_.data().begin(), ghostM.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( 1., ghostM, 1., ghostP);
        }
        cusp::blas::axpby(  ghostPV,  outV, ghostPV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
        cusp::blas::axpby(  ghostPV,  outV, outV, 1.,1.);
    }
}
template< class G,class M, class C, class container>
void FieldAligned<G,RowDistMat<M,C>,MPI_Vector<container> >::einsPlusT( const MPI_Vector<container>& f, MPI_Vector<container>& fme)
{
    //dg::blas2::detail::doSymv( plusT, f, fme, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    const container& in = f.data();
    container& out = fme.data();
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];

    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            //first exchange data in XY
            thrust::copy( in.cbegin() + i0*size2d, in.cbegin() + (i0+1)*size2d, temp_[i0].begin());
            tempXYplus_[i0] = commXYplus_.global_gather( temp_[i0]);
            cView inV( tempXYplus_[i0].cbegin(), tempXYplus_[i0].cend() );
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( plusT, inV, tempV);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end());
            cusp::multiply( plusT, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
    {
        int ip = i0 - 1;
        if( ip < 0 ) ip += (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].end(), out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        detail::sendForward( temp_[g_.Nz()-1].begin(), temp_[g_.Nz()-1].end(), out.begin());
    }
    //make ghostcells in first plane
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostMV);
        if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::axpby( 2., left_, -1, ghostM);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
        {
            thrust::transform( left_.begin(), left_.end(), hm_.data().begin(), ghostP.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( -1, ghostP, 1., ghostM);
        }
        cusp::blas::axpby(  ghostMV,  outV, ghostMV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
        cusp::blas::axpby(  ghostMV,  outV, outV, 1.,1.);
    }
}

///@endcond
}//namespace dg

