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

aMPIGeometry2d* clone_MPI3d_to_perp( const aMPIGeometry3d* grid_ptr)
{
    //%%%%%%%%%%%downcast grid since we don't have a virtual function perp_grid%%%%%%%%%%%%%
    const dg::CartesianMPIGrid3d* grid_cart = dynamic_cast<const dg::CartesianMPIGrid3d*>(grid_ptr);
    const dg::CylindricalMPIGrid3d* grid_cyl = dynamic_cast<const dg::CylindricalMPIGrid3d*>(grid_ptr);
    const dg::geo::CurvilinearProductMPIGrid3d*  grid_curvi = dynamic_cast<const dg::geo::CurvilinearProductMPIGrid3d*>(grid_ptr);
    aMPIGeometry2d* g2d_ptr;
    if( grid_cart) 
    {
        dg::CartesianMPIGrid2d cart = grid_cart->perp_grid();
        g2d_ptr = cart.clone();
    }
    else if( grid_cyl) 
    {
        dg::CartesianMPIGrid2d cart = grid_cyl->perp_grid();
        g2d_ptr = cart.clone();
    }
    else if( grid_curvi) 
    {
        dg::geo::CurvilinearMPIGrid2d curv = grid_curvi->perp_grid();
        g2d_ptr = curv.clone();
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
        m_bcz = bcz; 
        const dg::MPIGrid2d g2d( 0., 1., 0., 1., m_g.get().global().n(), m_g.get().global().Nx(), m_g.get().global().Ny(), m_g.get().perp_communicator() );
        m_left  = dg::evaluate( dg::CONSTANT(left), g2d);
        m_right = dg::evaluate( dg::CONSTANT(right),g2d);
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& left, const MPI_Vector<LocalContainer>& right)
    {
        m_bcz = bcz; 
        m_left = left;
        m_right = right;
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& global, double scal_left, double scal_right)
    {
        dg::split( global, m_temp);
        dg::blas1::axpby( scal_left, m_temp[0],               0., m_left);
        dg::blas1::axpby( scal_right, m_temp[m_g.get().Nz()], 0., m_left);
        m_bcz = bcz;
    }

    template< class BinaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, unsigned plane=0) const;

    template< class BinaryOp, class UnaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const;

    void operator()(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);

    const MPI_Vector<LocalContainer>& hz_inv()const {return m_hz_inv;}
    const MPI_Vector<LocalContainer>& hp_inv()const {return m_hp_inv;}
    const MPI_Vector<LocalContainer>& hm_inv()const {return m_hm_inv;}
    const Geometry& grid() const{return g_;}
  private:
    MPI_Vector<LocalContainer> m_hz_inv, m_hp_inv, m_hm_inv; 
    LocalContainer m_ghostM, m_ghostP;
    dg::Handle<Geometry> m_g;
    dg::bc m_bcz;
    MPI_Vector<LocalContainer> m_left, m_right;
    MPI_Vector<LocalContainer> m_limiter;
    std::vector<LocalContainer> tempXYplus_, tempXYminus_, m_temp; 
    MPIDistMat<LocalIMatrix, CommunicatorXY> m_plus, m_minus, m_plusT, m_minusT;
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
template <class Limiter>
FieldAligned<MPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> >::FieldAligned(
    const dg::geo::BinaryVectorLvl0& vec, const MPIGeometry& grid, unsigned mx, unsigned my, double eps, Limiter limit, dg::bc globalbcx, dg::bc globalbcy, double deltaPhi):
    m_hz( dg::evaluate( dg::zero, grid)), m_hp( m_hz), m_hm( m_hz), 
    m_g(grid), m_bcz(grid.bcz()), 
    tempXYplus_(g_.Nz()), tempXYminus_(g_.Nz()), temp_(g_.Nz())
{
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    const aMPIGeometry2d* grid2d_ptr = detail::clone_MPI3d_to_global_perp(&grid);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    unsigned localsize = grid2d_ptr->size();
    MPI_Vector<dg::HVec> temp = dg::pullback( limit, *grid2d_ptr);
    dg::blas1::transfer( temp.data(), limiter_);
    right_ = left_ = dg::evaluate( zero, grid2d_ptr->local());
    ghostM.resize( localsize); ghostP.resize( localsize);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%%%%
    std::vector<thrust::host_vector<double> > yp_coarse( 3), ym_coarse(yp_coarse); 
    
    dg::aMPIGeometry2d* g2dField_ptr = grid2d_ptr->clone();//INTEGRATE HIGH ORDER GRID
    g2dField_ptr->set( 7, g2dField_ptr->global().Nx(), g2dField_ptr->global().Ny());
    dg::aGeometry2d* global_g2dField_ptr = g2dField_ptr->global_grid();
    dg::aGrid2d local_g2dField = g2dField_ptr->local();
    detail::integrate_all_fieldlines2d( vec, global_g2dField_ptr, &local_g2dField, yp_coarse, ym_coarse, deltaPhi, eps);

    dg::MPIGrid2d g2dFine((dg::MPIGrid2d(*grid2d_ptr)));//FINE GRID
    g2dFine.multiplyCellNumbers((double)mx, (double)my);
    dg::IHMatrix interpolate = dg::create::interpolation( g2dFine.local(), local_g2dField);  //INTERPOLATE TO FINE GRID
    interpolate_and_clip( interpolate, global_g2dField_ptr, yp_coarse, ym_coarse, yp, ym);
    delete g2dField_ptr;
    delete global_g2dField_ptr;
    //%%%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t.tic();
    dg::IHMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], grid2d_ptr->global(), globalbcx, globalbcy), plus, plusT;
    dg::IHMatrix minusFine = dg::create::interpolation( ym[0], ym[1], grid2d_ptr->global(), globalbcx, globalbcy), minus, minusT;
    dg::IHMatrix projection = dg::create::projection( grid2d_ptr->local(), g2dFine->local());
    t.toc();
    std::cout <<"Creation of interpolation/projection took "<<t.diff()<<"s\n";
    t.tic();
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
    t.toc();
    std::cout<< "Multiplication of P*I took: "<<t.diff()<<"s\n";
    //%Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    cusp::transpose( plus, plusT);
    cusp::transpose( minus, minusT);     
    dg::blas2::transfer( plus, m_plus);
    dg::blas2::transfer( plusT, m_plusT);
    dg::blas2::transfer( minus, m_minus);
    dg::blas2::transfer( minusT, m_minusT);
    //%%%%%%%%%%%%%%%%%%%%%%%project h and copy into h vectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thrust::host_vector<double> hp( perp_size_), hm(hp);
    dg::blas2::symv( projection, yp[2], hp);
    dg::blas2::symv( projection, ym[2], hm);
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( hp.begin(), hp.end(), hp_.data().begin() + i*localsize_);
        thrust::copy( hm.begin(), hm.end(), hm_.data().begin() + i*localsize_);
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);
    delete grid2d_ptr;

    CommunicatorXY cp( pids, grid2d_ptr.communicator());
    commXYplus_ = cp;
    thrust::host_vector<double> pX, pY;
    dg::blas1::transfer( cp.global_gather( yp[0]), pX);
    dg::blas1::transfer( cp.global_gather( yp[1]), pY);

    //construt interpolation matrix
    plus = dg::create::interpolation( pX, pY, grid2d_ptr.local(), globalbcz); //inner points hopefully never lie exactly on local boundary
    cusp::transpose( plus, plusT);

    CommunicatorXY cm( pids, grid2d_ptr.communicator());
    commXYminus_ = cm;
    dg::blas1::transfer( cm.global_gather( ym[0]), pX);
    dg::blas1::transfer( cm.global_gather( ym[1]), pY);
    minus = dg::create::interpolation( pX, pY, grid2d_ptr.local(), globalbcz); //inner points hopefully never lie exactly on local boundary
    cusp::transpose( minus, minusT);
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
    assert( p0 < m_g.get().global().Nz());
    const aGeometry2d* g2d_ptr = m_g.get().perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, *g2d_perp); 

    MPI_Vector<container> temp(init2d), tempP(init2d), tempM(init2d);
    MPI_Vector<container> vec3d = dg::evaluate( dg::zero, m_g.get());
    std::vector<MPI_Vector<container> >  plus2d, minus2d, result;
    dg.:split( vec3d, plus2d);
    dg.:split( vec3d, minus2d);
    dg.:split( vec3d, result);
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<m_g.get().global().Nz(); i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = i0 + r*m_g.get().global().Nz(); 
            for(unsigned k=0; k<rep; k++)
            {
                dg::blas2::symv( plus, tempP, temp);
                temp.swap( tempP);
                dg::blas2::symv( minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*m_g.get().hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_g.get().hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<m_g.get().global().Nz(); i0++)
        {
            int idx = (int)(i0+coords[2]*m_g.get().Nz())  - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].data().begin(), result[i0].data().end(), vec3d.data().begin() + i0*g2d_ptr->size());
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<g_.global().Nz(); i0++)
        {
            unsigned revi0 = (m_g.get().global().Nz() - i0)%m_g.get().global().Nz(); //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<g_.Nz(); i0++)
        {
            int idx = ((int)i0 + coords[2]*g_.Nz() -(int)p0 + m_g.get().global().Nz())%m_g.get().global().Nz(); //shift index
            thrust::copy( result[idx].data().begin(), result[idx].data().end(), vec3d.data().begin() + i0*g2d_ptr->size());
        }
    }
    delete g2d_ptr;
    return vec3d;
}

template<class G, class M, class C, class container>
void FieldAligned<G, RowDistMatr<M,C>, MPI_Vector<container> >::operator()(enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fe)
{
    if(which == einsPlus || which == einsMinusT) ePlus( which, f, fe);
    if(which == einsMinus || which == einsPlusT) eMinus( which, f, fe);
}

template<class G, class M, class C, class container>
void FieldAligned<G,RowDistMat<M,C>, MPI_Vector<container> >::ePlus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fpe ) 
{
    dg::split( f, m_f);

    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
        if(which == einsPlus)           dg::blas2::symv( plus,   m_f[ip], m_temp[i0]);
        else if(which == einsMinusT)    dg::blas2::symv( minusT, m_f[ip], m_temp[i0]);
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = m_Nz-1;
        detail::sendBackward( m_temp[i0].begin(), m_temp[i0].end(), m_buffer.begin());
        m_temp[i0].swap( m_buffer);
    }

    //3. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER && m_g.z1() == m_g.global().z1())
    {
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            dg::blas1::axpby( 2, m_right, -1., m_f[i0], m_ghostP);
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::pointwiseDivide( m_right, m_hp_inv[i0], m_ghostP);
            dg::blas1::axpby( 1., m_ghostP, 1., m_f[i0], m_ghostP);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
    }
    dg::join( m_temp, fpe);
}

template<class G, class M, class C, class container>
void FieldAligned<G,RowDistMat<M,C>, MPI_Vector<container> >::eMinus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fme ) 
{
    dg::split( f, m_f);

    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned im = (i0==0) ? m_Nz-1:i0-1;
        if(which == einsPlusT)         dg::blas2::symv( plusT, m_f[im], m_temp[i0]);
        else if(which == einsMinus)    dg::blas2::symv( minus, m_f[im], m_temp[i0]);
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = 0;
        detail::sendForward( m_temp[i0].begin(), m_temp[i0].end(), m_buffer.begin());
        m_temp[i0].swap( m_buffer);
    }

    //3. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER)
    {
        if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( 2., m_left,  -1., m_f[i0], m_ghostM);
        if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
        {
            dg::blas1::pointwiseDivide( m_left, m_hm_inv[i0], m_ghostM);
            dg::blas1::axpby( -1., m_ghostM, 1., m_f[i0], m_ghostM);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
    }
    dg::join( m_temp, fme);
}


///@endcond
}//namespace dg

