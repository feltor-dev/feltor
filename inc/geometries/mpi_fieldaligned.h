#pragma once

#include "dg/algorithm.h"
#include "fieldaligned.h"
#include "dg/backend/timer.h"

namespace dg{
namespace geo{

///@cond
namespace detail{

///basically a copy across processes
template<class thrust_vector0, class thrust_vector1>
void sendForward( const thrust_vector0& in, thrust_vector1& out, MPI_Comm comm) //send to next plane
{
    int source, dest;
    MPI_Status status;
    MPI_Cart_shift( comm, 2, +1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    if( std::is_same< get_execution_policy<thrust_vector0>, CudaTag>::value) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaDeviceSynchronize(); //wait until device functions are finished before sending data
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    }
#endif //THRUST_DEVICE_SYSTEM
    unsigned size = in.size();
    MPI_Sendrecv(   thrust::raw_pointer_cast(in.data()), size, MPI_DOUBLE,  //sender
                    dest, 9,  //destination
                    thrust::raw_pointer_cast(out.data()), size, MPI_DOUBLE, //receiver
                    source, 9, //source
                    comm, &status);
}
///basically a copy across processes
template<class thrust_vector0, class thrust_vector1>
void sendBackward( const thrust_vector0& in, thrust_vector1& out, MPI_Comm comm) //send to next plane
{
    int source, dest;
    MPI_Status status;
    MPI_Cart_shift( comm, 2, -1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    if( std::is_same< get_execution_policy<thrust_vector0>, CudaTag>::value) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaDeviceSynchronize(); //wait until device functions are finished before sending data
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    }
#endif //THRUST_DEVICE_SYSTEM
    unsigned size = in.size();
    MPI_Sendrecv(   thrust::raw_pointer_cast(in.data()), size, MPI_DOUBLE,  //sender
                    dest, 3,  //destination
                    thrust::raw_pointer_cast(out.data()), size, MPI_DOUBLE, //receiver
                    source, 3, //source
                    comm, &status);
}
}//namespace detail

template <class ProductMPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
struct Fieldaligned< ProductMPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> >
{
    Fieldaligned(){}
    template <class Limiter>
    Fieldaligned(const dg::geo::TokamakMagneticField& vec,
        const ProductMPIGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi = -1, std::string interpolation_method = "dg",
        bool benchmark = true):
            Fieldaligned( dg::geo::createBHat(vec), grid, bcx, bcy, limit, eps,
                    mx, my, deltaPhi, interpolation_method)
    {
    }
    template <class Limiter>
    Fieldaligned(const dg::geo::CylindricalVectorLvl1& vec,
        const ProductMPIGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi = -1, std::string interpolation_method = "dg",
        bool benchmark = true);
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Fieldaligned( std::forward<Params>( ps)...);
    }

    dg::bc bcx()const{
        return m_bcx;
    }
    dg::bc bcy()const{
        return m_bcy;
    }

    void set_boundaries( dg::bc bcz, double left, double right)
    {
        m_bcz = bcz;
        dg::blas1::copy( left, m_left);
        dg::blas1::copy( right, m_right);
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& left, const MPI_Vector<LocalContainer>& right)
    {
        m_bcz = bcz;
        m_left = left;
        m_right = right;
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& global, double scal_left, double scal_right)
    {
        dg::split( global, m_f, *m_g);
        dg::blas1::axpby( scal_left,  m_f[0],                   0., m_left);
        dg::blas1::axpby( scal_right, m_f[m_g->local().Nz()-1], 0., m_right);
        m_bcz = bcz;
    }

    void operator()(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);

    double deltaPhi() const{return m_deltaPhi;}
    const MPI_Vector<LocalContainer>& hbm()const {
        return m_hbm;
    }
    const MPI_Vector<LocalContainer>& hbp()const {
        return m_hbp;
    }
    const MPI_Vector<LocalContainer>& sqrtG()const {
        return m_G;
    }
    const MPI_Vector<LocalContainer>& sqrtGm()const {
        return m_Gm;
    }
    const MPI_Vector<LocalContainer>& sqrtGp()const {
        return m_Gp;
    }
    const MPI_Vector<LocalContainer>& bphi()const {
        return m_bphi;
    }
    const MPI_Vector<LocalContainer>& bphiM()const {
        return m_bphiM;
    }
    const MPI_Vector<LocalContainer>& bphiP()const {
        return m_bphiP;
    }
    const MPI_Vector<LocalContainer>& bbm()const {
        return m_bbm;
    }
    const MPI_Vector<LocalContainer>& bbo()const {
        return m_bbo;
    }
    const MPI_Vector<LocalContainer>& bbp()const {
        return m_bbp;
    }
    const ProductMPIGeometry& grid() const{return *m_g;}

    template< class BinaryOp, class UnaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, UnaryOp g, unsigned p0,
            unsigned rounds) const;
    std::string method() const{return m_interpolation_method;}
  private:
    void ePlus( enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    void eMinus(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    void zero(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    MPIDistMat<LocalIMatrix, CommunicatorXY> m_plus, m_zero, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    MPI_Vector<LocalContainer> m_hbm, m_hbp; //3d size
    MPI_Vector<LocalContainer> m_G, m_Gm, m_Gp; //3d size
    MPI_Vector<LocalContainer> m_bphi, m_bphiM, m_bphiP; //3d size
    MPI_Vector<LocalContainer> m_bbm, m_bbp, m_bbo; //3d size

    MPI_Vector<LocalContainer> m_left, m_right; //2d size
    MPI_Vector<LocalContainer> m_limiter; //2d size
    MPI_Vector<LocalContainer> m_ghostM, m_ghostP; //2d size
    unsigned m_Nz, m_perp_size;
    dg::bc m_bcx, m_bcy, m_bcz;
    std::vector<MPI_Vector<dg::View<const LocalContainer>> > m_f;
    std::vector<MPI_Vector<dg::View<LocalContainer>> > m_temp;
    dg::ClonePtr<ProductMPIGeometry> m_g;
    double m_deltaPhi;
    std::string m_interpolation_method;
    unsigned m_coords2, m_sizeZ; //number of processes in z
#ifdef _DG_CUDA_UNAWARE_MPI
    //we need to manually send data through the host
    thrust::host_vector<double> m_send_buffer, m_recv_buffer; //2d size
#endif
    bool m_have_adjoint = false;
    void updateAdjoint( )
    {
        m_plusT = dg::transpose( m_plus);
        m_minusT = dg::transpose( m_minus);
        m_have_adjoint = true;
    }
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
template <class Limiter>
Fieldaligned<MPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> >::Fieldaligned(
    const dg::geo::CylindricalVectorLvl1& vec,
    const MPIGeometry& grid,
    dg::bc bcx, dg::bc bcy, Limiter limit, double eps,
    unsigned mx, unsigned my, double deltaPhi, std::string interpolation_method, bool benchmark
    ):
        m_g(grid),
        m_interpolation_method(interpolation_method)
{
    int rank;
    MPI_Comm_rank( grid.communicator(), &rank);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( m_g->communicator(), 3, dims, periods, coords);
    m_coords2 = coords[2], m_sizeZ = dims[2];

    std::string inter_m, project_m, fine_m;
    detail::parse_method( interpolation_method, inter_m, project_m, fine_m);
    if( benchmark && rank==0) std::cout << "# Interpolation method: \""<<inter_m << "\" projection method: \""<<project_m<<"\" fine grid \""<<fine_m<<"\"\n";
    ///Let us check boundary conditions:
    if( (grid.bcx() == PER && bcx != PER) || (grid.bcx() != PER && bcx == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting periodicity in x. The grid says "<<bc2str(grid.bcx())<<" while the parameter says "<<bc2str(bcx)));
    if( (grid.bcy() == PER && bcy != PER) || (grid.bcy() != PER && bcy == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting boundary conditions in y. The grid says "<<bc2str(grid.bcy())<<" while the parameter says "<<bc2str(bcy)));
    m_Nz=grid.local().Nz(), m_bcz=grid.bcz(), m_bcx = bcx, m_bcy = bcy;
    if( deltaPhi <=0) deltaPhi = grid.hz();
    ///%%%%%%%%%%%%%%%%%%%%%Setup grids%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //  grid_trafo -> grid_equi -> grid_fine -> grid_equi -> grid_trafo
    dg::Timer t;
    if( benchmark) t.tic();
    dg::ClonePtr<dg::aMPIGeometry2d> grid_transform( grid.perp_grid()) ;
    // We do not need metric of grid_equidist or or grid_fine
    dg::RealMPIGrid2d<double> grid_equidist( *grid_transform) ;
    dg::RealMPIGrid2d<double> grid_fine( *grid_transform);
    grid_equidist.set( 1, grid.global().gx().size(), grid.global().gy().size());
    dg::ClonePtr<dg::aMPIGeometry2d> grid_magnetic = grid_transform;//INTEGRATE HIGH ORDER GRID
    grid_magnetic->set( grid_transform->n() < 3 ? 4 : 7, grid_magnetic->Nx(), grid_magnetic->Ny());
    dg::ClonePtr<dg::aGeometry2d> global_grid_magnetic =
        grid_magnetic->global_geometry();
    // For project method "const" we round up to the nearest multiple of n
    if( project_m != "dg" && fine_m == "dg")
    {
        unsigned rx = mx % grid.nx(), ry = my % grid.ny();
        if( 0 != rx || 0 != ry)
        {
            if(rank==0)std::cerr << "#Warning: for projection method \"const\" mx and my must be multiples of nx and ny! Rounding up for you ...\n";
            mx = mx + grid.nx() - rx;
            my = my + grid.ny() - ry;
        }
    }
    if( fine_m == "equi")
        grid_fine = grid_equidist;
    grid_fine.multiplyCellNumbers((double)mx, (double)my);
    if( benchmark)
    {
        t.toc();
        if(rank==0) std::cout << "# DS: High order grid gen   took: "<<t.diff()<<"\n";
        t.tic();
    }
    ///%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%//
    std::array<thrust::host_vector<double>,3> yp_trafo, ym_trafo, yp, ym;
    thrust::host_vector<bool> in_boxp, in_boxm;
    thrust::host_vector<double> hbp, hbm;
    auto vol = dg::tensor::volume(grid.metric()), vol2d0(vol);
    auto vol2d = dg::split( vol, grid);
    dg::assign( vol2d[0], vol2d0);
    detail::integrate_all_fieldlines2d( vec, *global_grid_magnetic,
            grid_transform->local(), yp_trafo, vol2d0.data(), hbp, in_boxp,
            deltaPhi, eps);
    detail::integrate_all_fieldlines2d( vec, *global_grid_magnetic,
            grid_transform->local(), ym_trafo, vol2d0.data(), hbm, in_boxm,
            -deltaPhi, eps);
    dg::HVec Xf = dg::evaluate(  dg::cooX2d, grid_fine.local());
    dg::HVec Yf = dg::evaluate(  dg::cooY2d, grid_fine.local());
    {
    dg::IHMatrix interpolate = dg::create::interpolation( Xf, Yf,
            grid_transform->local(), dg::NEU, dg::NEU, grid_transform->n() < 3 ? "cubic" : "dg");
    yp.fill(dg::evaluate( dg::zero, grid_fine.local()));
    ym = yp;
    for( int i=0; i<2; i++)
    {
        dg::blas2::symv( interpolate, yp_trafo[i], yp[i]);
        dg::blas2::symv( interpolate, ym_trafo[i], ym[i]);
    }
    } // release memory for interpolate matrix
    if(benchmark)
    {
        t.toc();
        if(rank==0) std::cout << "# DS: Fieldline integration took: "<<t.diff()<<"\n";
        t.tic();
    }
    ///%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%//
    {
    dg::IHMatrix plusFine, minusFine, zeroFine;
    if( inter_m == "dg")
    {
        plusFine = dg::create::interpolation( yp[0], yp[1],
                grid_transform->global(), bcx, bcy, "dg");
        zeroFine = dg::create::interpolation( Xf, Yf,
                grid_transform->global(), bcx, bcy, "dg");
        minusFine = dg::create::interpolation( ym[0], ym[1],
                grid_transform->global(), bcx, bcy, "dg");
    }
    else
    {
        dg::IHMatrix plusFineTmp = dg::create::interpolation( yp[0], yp[1],
                grid_equidist.global(), bcx, bcy, inter_m);
        dg::IHMatrix zeroFineTmp = dg::create::interpolation( Xf, Yf,
                grid_equidist.global(), bcx, bcy, inter_m);
        dg::IHMatrix minusFineTmp = dg::create::interpolation( ym[0], ym[1],
                grid_equidist.global(), bcx, bcy, inter_m);
        dg::IHMatrix forw = dg::create::backproject( grid_transform->global()); // from dg to equidist
        cusp::multiply( plusFineTmp, forw, plusFine);
        cusp::multiply( zeroFineTmp, forw, zeroFine);
        cusp::multiply( minusFineTmp, forw, minusFine);
    }
    dg::IHMatrix projection;
    // Now project
    if ( project_m == "dg")
    {
        projection = dg::create::projection( grid_transform->global(), grid_fine.local());
    }
    else // const
    {
        /// ATTENTION project_m may incur communication!!
        projection = dg::create::projection( grid_equidist.global(), grid_fine.local(), project_m);
    }
    dg::IHMatrix plus, minus, zero;
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, zeroFine, zero);
    cusp::multiply( projection, minusFine, minus);
    // Now convert to row dist matrix
    auto plusL = dg::convertGlobal2LocalRows( plus, *grid_transform);
    auto zeroL = dg::convertGlobal2LocalRows( zero, *grid_transform);
    auto minusL = dg::convertGlobal2LocalRows( minus, *grid_transform);
    if( !(project_m == "dg"))
    {
        auto back = dg::create::inv_backproject( grid_transform->local());
        cusp::multiply( back, plusL, plus);
        cusp::multiply( back, zeroL, zero);
        cusp::multiply( back, minusL, minus);
    }
    else
    {
        plus = plusL;
        zero = zeroL;
        minus = minusL;
    }
    if( benchmark)
    {
        t.toc();
        if(rank==0) std::cout << "# DS: Multiplication PI     took: "<<t.diff()<<"\n";
        t.tic();
    }
    dg::MIHMatrix temp = dg::convert( plus, *grid_transform); //, tempT;
    dg::blas2::transfer( temp, m_plus);
    temp = dg::convert( zero, *grid_transform);
    dg::blas2::transfer( temp, m_zero);
    temp = dg::convert( minus, *grid_transform);
    dg::blas2::transfer( temp, m_minus);
    }
    if( benchmark)
    {
        t.toc();
        if(rank==0) std::cout << "# DS: Conversion            took: "<<t.diff()<<"\n";
    }
    ///%%%%%%%%%%%%%%%%%%%%copy into h vectors %%%%%%%%%%%%%%%%%%%//
    dg::HVec hbphi( yp_trafo[2]), hbphiP(hbphi), hbphiM(hbphi);
    auto tmp = dg::pullback( vec.z(), *grid_transform);
    hbphi = tmp.data();
    //this is a pullback bphi( R(zeta, eta), Z(zeta, eta)):
    if( dynamic_cast<const dg::CartesianMPIGrid2d*>( grid_transform.get()))
    {
        for( unsigned i=0; i<hbphiP.size(); i++)
        {
            hbphiP[i] = vec.z()(yp_trafo[0][i], yp_trafo[1][i]);
            hbphiM[i] = vec.z()(ym_trafo[0][i], ym_trafo[1][i]);
        }
    }
    else
    {
        dg::HVec Ihbphi = dg::pullback( vec.z(), *global_grid_magnetic);
        dg::HVec Lhbphi = dg::forward_transform( Ihbphi, *global_grid_magnetic);
        for( unsigned i=0; i<yp_trafo[0].size(); i++)
        {
            hbphiP[i] = dg::interpolate( dg::lspace, Lhbphi, yp_trafo[0][i],
                    yp_trafo[1][i], *global_grid_magnetic);
            hbphiM[i] = dg::interpolate( dg::lspace, Lhbphi, ym_trafo[0][i],
                    ym_trafo[1][i], *global_grid_magnetic);
        }
    }
    dg::assign3dfrom2d( dg::MHVec(hbphi,  MPI_COMM_WORLD), m_bphi,  grid);
    dg::assign3dfrom2d( dg::MHVec(hbphiM, MPI_COMM_WORLD), m_bphiM, grid);
    dg::assign3dfrom2d( dg::MHVec(hbphiP, MPI_COMM_WORLD), m_bphiP, grid);

    dg::assign3dfrom2d( dg::MHVec(yp_trafo[2], MPI_COMM_WORLD), m_Gp, grid);
    dg::assign3dfrom2d( dg::MHVec(ym_trafo[2], MPI_COMM_WORLD), m_Gm, grid);
    m_G = vol;
    MPI_Vector<LocalContainer> weights = dg::create::weights( grid);
    dg::blas1::pointwiseDot( m_G, weights, m_G);
    dg::blas1::pointwiseDot( m_Gp, weights, m_Gp);
    dg::blas1::pointwiseDot( m_Gm, weights, m_Gm);

    dg::assign( dg::evaluate( dg::zero, grid), m_hbm);
    m_temp = dg::split( m_hbm, grid); //3d vector
    m_f = dg::split( (const MPI_Vector<LocalContainer>&)m_hbm, grid);
    dg::assign3dfrom2d( dg::MHVec(hbp, MPI_COMM_WORLD), m_hbp, grid);
    dg::assign3dfrom2d( dg::MHVec(hbm, MPI_COMM_WORLD), m_hbm, grid);
    dg::blas1::scal( m_hbm, -1.);
    ///%%%%%%%%%%%%%%%%%%%%create mask vectors %%%%%%%%%%%%%%%%%%%//
    thrust::host_vector<double> bbm( in_boxp.size(),0.), bbo(bbm), bbp(bbm);
    for( unsigned i=0; i<in_boxp.size(); i++)
    {
        if( !in_boxp[i] && !in_boxm[i])
            bbo[i] = 1.;
        else if( !in_boxp[i] && in_boxm[i])
            bbp[i] = 1.;
        else if( in_boxp[i] && !in_boxm[i])
            bbm[i] = 1.;
        // else all are 0
    }
    dg::assign3dfrom2d( dg::MHVec(bbm, MPI_COMM_WORLD), m_bbm, grid);
    dg::assign3dfrom2d( dg::MHVec(bbo, MPI_COMM_WORLD), m_bbo, grid);
    dg::assign3dfrom2d( dg::MHVec(bbp, MPI_COMM_WORLD), m_bbp, grid);

    m_deltaPhi = deltaPhi; // store for evaluate

    ///%%%%%%%%%%%%%%%%%%%%%Assign Limiter%%%%%%%%%%%%%%%%%%%%%%%%%//
    m_perp_size = grid_transform->local().size();
    dg::assign( dg::pullback(limit, *grid_transform), m_limiter);
    dg::assign( dg::evaluate(dg::zero, *grid_transform), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
#ifdef _DG_CUDA_UNAWARE_MPI
    m_recv_buffer = m_send_buffer = m_ghostP.data();
#endif
}


template<class G, class M, class C, class container>
void Fieldaligned<G, MPIDistMat<M,C>, MPI_Vector<container> >::operator()(enum
        whichMatrix which, const MPI_Vector<container>& f,
        MPI_Vector<container>& fe)
{
    if(which == einsPlus || which == einsMinusT) ePlus( which, f, fe);
    if(which == einsMinus || which == einsPlusT) eMinus( which, f, fe);
    if(     which == einsPlus  || which == einsMinusT ) ePlus(  which, f, fe);
    else if(which == einsMinus || which == einsPlusT  ) eMinus( which, f, fe);
    else if(which == zeroMinus || which == zeroPlus ||
            which == zeroMinusT|| which == zeroPlusT ||
            which == zeroForw  ) zero(   which, f, fe);
}
template< class G, class M, class C, class container>
void Fieldaligned<G, MPIDistMat<M,C>, MPI_Vector<container> >::zero( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& f0)
{
    dg::split( f, m_f, *m_g);
    dg::split( f0, m_temp, *m_g);
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        if(which == zeroPlus)
            dg::blas2::symv( m_plus,   m_f[i0], m_temp[i0]);
        else if(which == zeroMinus)
            dg::blas2::symv( m_minus,  m_f[i0], m_temp[i0]);
        else if(which == zeroPlusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_plusT,  m_f[i0], m_temp[i0]);
        }
        else if(which == zeroMinusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_minusT, m_f[i0], m_temp[i0]);
        }
        else if( which == zeroForw)
        {
            if ( m_interpolation_method != "dg" )
            {
                dg::blas2::symv( m_zero, m_f[i0], m_temp[i0]);
            }
            else
                dg::blas1::copy( m_f[i0], m_temp[i0]);
        }
    }
}

template<class G, class M, class C, class container>
void Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::ePlus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fpe )
{
    dg::split( f, m_f, *m_g);
    dg::split( fpe, m_temp, *m_g);
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
        if(which == einsPlus)
            dg::blas2::symv( m_plus,   m_f[ip], m_temp[i0]);
        else if(which == einsMinusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_minusT, m_f[ip], m_temp[i0]);
        }
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = m_Nz-1;
#ifdef _DG_CUDA_UNAWARE_MPI
        thrust::copy( m_temp[i0].data().cbegin(), m_temp[i0].data().cend(), m_send_buffer.begin());
        detail::sendBackward( m_send_buffer, m_recv_buffer, m_g->communicator());
        thrust::copy( m_recv_buffer.cbegin(), m_recv_buffer.cend(), m_temp[i0].data().begin());
#else
        detail::sendBackward( m_temp[i0].data(), m_ghostM.data(), m_g->communicator());
        dg::blas1::copy( m_ghostM, m_temp[i0]);
#endif //_DG_CUDA_UNAWARE_MPI
    }

    //3. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER && m_g->local().z1() == m_g->global().z1())
    {
        if( m_bcz == dg::DIR || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( 2, m_right, -1., m_f[i0], m_ghostP);
        if( m_bcz == dg::NEU || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( m_deltaPhi, m_right, 1., m_f[i0], m_ghostP);
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
    }
}

template<class G, class M, class C, class container>
void Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::eMinus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fme )
{
    int rank;
    MPI_Comm_rank(m_g->communicator(), &rank);
    dg::split( f, m_f, *m_g);
    dg::split( fme, m_temp, *m_g);
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned im = (i0==0) ? m_Nz-1:i0-1;
        if(which == einsPlusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_plusT, m_f[im], m_temp[i0]);
        }
        else if(which == einsMinus)
            dg::blas2::symv( m_minus, m_f[im], m_temp[i0]);
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = 0;
#ifdef _DG_CUDA_UNAWARE_MPI
        thrust::copy( m_temp[i0].data().cbegin(), m_temp[i0].data().cend(), m_send_buffer.begin());
        detail::sendForward( m_send_buffer, m_recv_buffer, m_g->communicator());
        thrust::copy( m_recv_buffer.cbegin(), m_recv_buffer.cend(), m_temp[i0].data().begin());
#else
        detail::sendForward( m_temp[i0].data(), m_ghostP.data(), m_g->communicator());
        dg::blas1::copy( m_ghostP, m_temp[i0]);
#endif //_DG_CUDA_UNAWARE_MPI
    }

    //3. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER && m_g->local().z0() == m_g->global().z0())
    {
        if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( 2., m_left,  -1., m_f[i0], m_ghostM);
        if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( -m_deltaPhi, m_left, 1., m_f[i0], m_ghostM);
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
    }
}

template<class G, class M, class C, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g->global().Nz());
    const dg::ClonePtr<aMPIGeometry2d> g2d = m_g->perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, *g2d);
    MPI_Vector<container> zero2d = dg::evaluate( dg::zero, *g2d);
    unsigned globalNz = m_g->global().Nz();

    MPI_Vector<container> temp(init2d), tempP(init2d), tempM(init2d);
    MPI_Vector<container> vec3d = dg::evaluate( dg::zero, *m_g);
    std::vector<MPI_Vector<container> >  plus2d(globalNz, zero2d), minus2d(plus2d), result(plus2d);
    unsigned turns = rounds;
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<globalNz; i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = r*globalNz + i0;
            for(unsigned k=0; k<rep; k++)
            {
                //!!! The value of f at the plus plane is I^- of the current plane
                dg::blas2::symv( m_minus, tempP, temp);
                temp.swap( tempP);
                //!!! The value of f at the minus plane is I^+ of the current plane
                dg::blas2::symv( m_plus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*m_deltaPhi ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_deltaPhi ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<m_Nz; i0++)
        {
            int idx = (int)(i0+m_coords2*m_Nz)  - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].data().begin(), result[i0].data().end(),
                    vec3d.data().begin() + i0*m_perp_size);
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<globalNz; i0++)
        {
            unsigned revi0 = (globalNz - i0)%globalNz; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<m_Nz; i0++)
        {
            int idx = ((int)i0 + m_coords2*m_Nz -(int)p0 + globalNz)%globalNz; //shift index
            thrust::copy( result[idx].data().begin(), result[idx].data().end(),
                    vec3d.data().begin() + i0*m_perp_size);
        }
    }
    return vec3d;
}


///@endcond


///@brief %Evaluate a 2d functor and transform to all planes along the fieldlines (MPI Version)
///@copydetails fieldaligned_evaluate(const aProductGeometry3d&,const CylindricalVectorLvl0&,const BinaryOp&,const UnaryOp&,unsigned,unsigned,double)
///@ingroup fieldaligned
template<class BinaryOp, class UnaryOp>
MPI_Vector<thrust::host_vector<double>> fieldaligned_evaluate(
        const aProductMPIGeometry3d& grid,
        const CylindricalVectorLvl0& vec,
        const BinaryOp& binary,
        const UnaryOp& unary,
        unsigned p0,
        unsigned rounds,
        double eps = 1e-5)
{
    unsigned Nz = grid.Nz();
    const dg::ClonePtr<aMPIGeometry2d> g2d = grid.perp_grid();
    // Construct for field-aligned output
    dg::MHVec vec3d = dg::evaluate( dg::zero, grid);
    dg::MHVec tempP = dg::evaluate( dg::zero, *g2d), tempM( tempP);
    std::vector<dg::MHVec>  plus2d(Nz, tempP), minus2d(plus2d), result(plus2d);
    dg::MHVec init2d = dg::pullback( binary, *g2d);
    std::array<dg::HVec,3> yy0{
        dg::evaluate( dg::cooX2d, g2d->local()),
        dg::evaluate( dg::cooY2d, g2d->local()),
        dg::evaluate( dg::zero, g2d->local())}, yy1(yy0), xx0( yy0), xx1(yy0); //s
    dg::geo::detail::DSFieldCylindrical3 cyl_field(vec);
    double deltaPhi = grid.hz();
    double phiM0 = 0., phiP0 = 0.;
    unsigned turns = rounds;
    if( turns == 0) turns++;
    for( unsigned r=0; r<turns; r++)
        for( unsigned  i0=0; i0<Nz; i0++)
        {
            unsigned rep = r*Nz + i0;
            if( rep == 0)
                tempM = tempP = init2d;
            else
            {
                dg::Adaptive<dg::ERKStep<std::array<double,3>>> adapt(
                        "Dormand-Prince-7-4-5", std::array<double,3>{0,0,0});
                dg::AdaptiveTimeloop<std::array<double,3>> odeint( adapt,
                    cyl_field, dg::pid_control, dg::fast_l2norm, eps, 1e-10);
                for( unsigned i=0; i<g2d->local().size(); i++)
                {
                    // minus direction needs positive integration!
                    double phiM1 = phiM0 + deltaPhi;
                    std::array<double,3>
                        coords0{yy0[0][i],yy0[1][i],yy0[2][i]}, coords1;
                    odeint.integrate_in_domain( phiM0, coords0, phiM1,
                            coords1, deltaPhi, g2d->global(), eps);
                    yy1[0][i] = coords1[0], yy1[1][i] = coords1[1], yy1[2][i] =
                        coords1[2];
                    tempM.data()[i] = binary( yy1[0][i], yy1[1][i]);

                    // plus direction needs negative integration!
                    double phiP1 = phiP0 - deltaPhi;
                    coords0 = std::array<double,3>{xx0[0][i],xx0[1][i],xx0[2][i]};
                    odeint.integrate_in_domain( phiP0, coords0, phiP1,
                            coords1, -deltaPhi, g2d->global(), eps);
                    xx1[0][i] = coords1[0], xx1[1][i] = coords1[1], xx1[2][i] =
                        coords1[2];
                    tempP.data()[i] = binary( xx1[0][i], xx1[1][i]);
                }
                std::swap( yy0, yy1);
                std::swap( xx0, xx1);
                phiM0 += deltaPhi;
                phiP0 -= deltaPhi;
            }
            dg::blas1::scal( tempM, unary( -(double)rep*deltaPhi ) );
            dg::blas1::scal( tempP, unary(  (double)rep*deltaPhi ) );
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
        }
    //now we have the plus and the minus filaments
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( grid.communicator(), 3, dims, periods, coords);
    unsigned coords2 = coords[2];
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<grid.local().Nz(); i0++)
        {
            int idx = (int)(i0+coords2*grid.local().Nz()) - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].data().begin(), result[i0].data().end(),
                    vec3d.data().begin() + i0*g2d->local().size());
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<Nz; i0++)
        {
            unsigned revi0 = (Nz - i0)%Nz; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for( unsigned i0=0; i0<grid.local().Nz(); i0++)
        {
            int idx = ((int)i0 +coords2*grid.local().Nz()-(int)p0 + Nz)%Nz;
            //shift index
            thrust::copy( result[idx].data().begin(), result[idx].data().end(),
                    vec3d.data().begin() + i0*g2d->local().size());
        }
    }
    return vec3d;
}

}//namespace geo
}//namespace dg
