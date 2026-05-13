#pragma once

#include "dg/algorithm.h"
#include "fieldaligned.h"
#include "dg/backend/timer.h"

namespace dg{
namespace geo{

///@cond

template <class ProductMPIGeometry, class MIMatrix, class LocalContainer>
struct Fieldaligned< ProductMPIGeometry, MIMatrix, MPI_Vector<LocalContainer> >
{
    Fieldaligned(){}
    template <class Limiter>
    Fieldaligned(const dg::geo::TokamakMagneticField& vec,
        const ProductMPIGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=12, unsigned my=12,
        double deltaPhi = -1, std::string interpolation_method = "linear-nearest",
        bool benchmark = true):
            Fieldaligned( dg::geo::createBHat(vec), grid, bcx, bcy, limit, eps,
                    mx, my, deltaPhi, interpolation_method, benchmark)
    {
    }
    template <class Limiter>
    Fieldaligned(const dg::geo::CylindricalVectorLvl1& vec,
        const ProductMPIGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=12, unsigned my=12,
        double deltaPhi = -1, std::string interpolation_method = "linear-nearest",
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
        dg::split( global, m_split_in, *m_g);
        dg::blas1::axpby( scal_left,  m_split_in[0],                   0., m_left);
        dg::blas1::axpby( scal_right, m_split_in[m_g->local().Nz()-1], 0., m_right);
        m_bcz = bcz;
    }

    void operator()(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out) const;

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
    void ePlus( enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out) const;
    void eMinus(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out) const;
    void zero(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out) const;
    MIMatrix m_plus, m_zero, m_minus; //2d interpolation matrices
    mutable MIMatrix m_plusT, m_minusT; // only allocated if necessary
    // backproject
    typename detail::Container2EllSparseBlockMat<LocalContainer>::type m_bx, m_by, m_inv_bx, m_inv_by;
    bool apply_backproject = false, apply_inv_backproject = false;
    mutable MPI_Vector<LocalContainer> m_temp0, m_temp1; // local 3d size
    //
    MPI_Vector<LocalContainer> m_hbm, m_hbp; //3d size
    MPI_Vector<LocalContainer> m_G, m_Gm, m_Gp; //3d size
    MPI_Vector<LocalContainer> m_bphi, m_bphiM, m_bphiP; //3d size
    MPI_Vector<LocalContainer> m_bbm, m_bbp, m_bbo; //3d size

    MPI_Vector<LocalContainer> m_left, m_right; //2d size
    MPI_Vector<LocalContainer> m_limiter; //2d size
    mutable MPI_Vector<LocalContainer> m_ghostM, m_ghostP; //2d size
    mutable std::vector<MPI_Vector<dg::View<const LocalContainer>> > m_split_in;
    mutable std::vector<MPI_Vector<dg::View<LocalContainer>> > m_split_out;
    dg::ClonePtr<ProductMPIGeometry> m_g;
    dg::bc m_bcx, m_bcy, m_bcz;
    unsigned m_Nz, m_perp_size, m_mx, m_my;
    double m_deltaPhi, m_eps;
    std::string m_interpolation_method;
    dg::geo::CylindricalVectorLvl1 m_vec; // to reconstruct adjoint
    unsigned m_coords2, m_sizeZ; //number of processes in z
    //we need to manually send data through the host for cuda-unaware-mpi
    mutable thrust::host_vector<double> m_buffer; //2d size
    dg::detail::MPIContiguousGather m_from_minus, m_from_plus;
    mutable bool m_have_adjoint = false;
    void updateAdjoint( ) const // only changes mutable m_have_adjoint and m_minusT, m_plusT
    {
        auto vol = dg::tensor::volume(m_g->metric()), vol2d0(vol);
        auto vol2d = dg::split( vol, *m_g);
        dg::assign( vol2d[0], vol2d0);
        dg::ClonePtr<dg::aMPIGeometry2d> grid_transform( m_g->perp_grid()) ;
        dg::ClonePtr<dg::aGeometry2d> global_grid_magnetic;
        std::array<thrust::host_vector<double>,3> yp_trafo, ym_trafo;
        thrust::host_vector<double> hbp, hbm;
        thrust::host_vector<bool> in_boxp, in_boxm;

        dg::MIHMatrix minus, zero, plus, minusT, plusT;
        make_matrices( m_vec, grid_transform, global_grid_magnetic,
            m_bcx, m_bcy, m_eps, m_mx, m_my, m_deltaPhi,
            m_interpolation_method,
            false, vol2d0, hbp, hbm,
            in_boxp, in_boxm,
            yp_trafo, ym_trafo,
            minus, zero, plus, true, minusT, plusT
            );
        dg::blas2::transfer( minusT, m_minusT);
        dg::blas2::transfer( plusT, m_plusT);
        m_have_adjoint = true;
    }

    void make_matrices(
        const dg::geo::CylindricalVectorLvl1& vec,
        const dg::ClonePtr<dg::aMPIGeometry2d>& grid_transform,
        dg::ClonePtr<dg::aGeometry2d>& global_grid_magnetic,
        dg::bc bcx, dg::bc bcy, double eps,
        unsigned mx, unsigned my,
        double deltaPhi, std::string interpolation_method,
        bool benchmark,
        const MPI_Vector<thrust::host_vector<double>>& vol2d0,
        thrust::host_vector<double>& hbp,
        thrust::host_vector<double>& hbm,
        thrust::host_vector<bool>& in_boxp,
        thrust::host_vector<bool>& in_boxm,
        std::array<thrust::host_vector<double>,3>& yp_trafo,
        std::array<thrust::host_vector<double>,3>& ym_trafo,
        dg::MIHMatrix& minus, dg::MIHMatrix& zero, dg::MIHMatrix& plus,
        bool make_adjoint,
        dg::MIHMatrix& minusT, dg::MIHMatrix& plusT
        ) const
    {
        int rank;
        MPI_Comm_rank( m_g->communicator(), &rank);
        ////////////////////////////////////////////////////////////////
        std::string inter_m, project_m, fine_m;
        detail::parse_method( interpolation_method, inter_m, project_m, fine_m);
        // For project method "const" we round up to the nearest multiple of n
        if( project_m != "dg" && fine_m == "dg")
        {
            unsigned rx = mx % grid_transform->nx(), ry = my % grid_transform->ny();
            if( 0 != rx || 0 != ry)
            {
                if(rank==0)std::cerr << "#Warning: for projection method \"const\" mx and my must be multiples of nx and ny! Rounding up for you ...\n";
                mx = mx + grid_transform->nx() - rx;
                my = my + grid_transform->ny() - ry;
            }
        }
        if( benchmark && rank==0)
            std::cout << "# Interpolation method: \""<<inter_m
                << "\" projection method: \""<<project_m
                <<"\" fine grid \""<<fine_m<<"\"\n";
        ///%%%%%%%%%%%%%%%%%%%%%Setup grids%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //  grid_trafo -> grid_equi -> grid_fine -> grid_equi -> grid_trafo
        dg::Timer t;
        if( benchmark) t.tic();
        // We do not need metric of grid_equidist or or grid_fine
        dg::ClonePtr<dg::aMPIGeometry2d> grid_magnetic = grid_transform;//INTEGRATE HIGH ORDER GRID
        // grid_magnetic is only used for integrating in curvilinear coords
        grid_magnetic->set( grid_transform->n() < 3 ? 4 : 7, grid_magnetic->Nx(), grid_magnetic->Ny());
        global_grid_magnetic = grid_magnetic->global_geometry();
        ///%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%//
        detail::integrate_all_fieldlines2d( vec, *global_grid_magnetic,
                grid_transform->local(), yp_trafo, vol2d0.data(), hbp, in_boxp,
                deltaPhi, eps);
        detail::integrate_all_fieldlines2d( vec, *global_grid_magnetic,
                grid_transform->local(), ym_trafo, vol2d0.data(), hbm, in_boxm,
                -deltaPhi, eps);
        if( benchmark)
        {
            t.toc();
            if(rank==0) std::cout << "# DS: Fieldline integration took: "<<t.diff()<<"\n";
            t.tic();
        }
        // Assemble minus, zero and plus through sub-grids
        // The idea for the sub-grids is that the fine interpolation matrix takes
        // up a large chunk of memory which can be avoided by sub-dividing the
        // fine grid along its rows
        dg::IHMatrix local_plus, local_zero, local_minus;
        dg::IHMatrix interpolate, zero_interpolate, projection;
        for( unsigned sub = 0; sub < grid_transform->local().Ny(); sub++)
        {
            dg::RealGrid2d<double> grid_fine_sub( grid_transform->local());
            grid_fine_sub = dg::RealGrid2d<double>(
                grid_fine_sub.x0(),
                grid_fine_sub.x1(),
                grid_fine_sub.y0() + sub*grid_fine_sub.hy(),
                // Fix: construction bug The local right boundary should be the same as the global right boundary
                sub == grid_transform->local().Ny()-1 ? grid_fine_sub.y1() :
                    grid_fine_sub.y0() + (sub+1)*grid_fine_sub.hy(),
                grid_fine_sub.n(), grid_fine_sub.Nx(), 1,
                grid_fine_sub.bcx(), grid_fine_sub.bcy());

            // We only need grid_fine_local and grid_equidist_global:
            // multiplying cell numbers on an MPI grid would redistribute
            // points for non-equipartition. So we make them RealGrid
            dg::RealGrid2d<double> grid_equidist_global( grid_transform->global()) ;
            grid_equidist_global.set( 1, grid_equidist_global.shape(0), grid_equidist_global.shape(1));
            if( fine_m == "equi")
                grid_fine_sub.set( 1, grid_fine_sub.shape(0), grid_fine_sub.shape(1));
            grid_fine_sub.multiplyCellNumbers((double)mx, (double)my);
            std::array<thrust::host_vector<double>,3> yp, ym;
            dg::HVec Xf = dg::evaluate(  dg::cooX2d, grid_fine_sub);
            dg::HVec Yf = dg::evaluate(  dg::cooY2d, grid_fine_sub);
            // interpolate matrix is same in all sub the rows just shift ...
            unsigned shift = grid_transform->local().shape(0) * grid_transform->n();
            unsigned global_shift = grid_transform->global().shape(0) * grid_transform->n();
            if( sub <= 1 || sub >= grid_transform->local().Ny() - 2 || grid_transform->n() < 3) // in latter case boundary conditions could destroy invariance
            {
                interpolate = dg::create::interpolation( Xf, Yf,
                    grid_transform->local(), dg::NEU, dg::NEU, grid_transform->n() < 3 ? "cubic" : "dg");
                zero_interpolate = dg::create::interpolation( Xf, Yf,
                    inter_m == "dg" ? grid_transform->global() :
                    grid_equidist_global, bcx, bcy, inter_m);
            }
            else
            {
                dg::blas1::plus( interpolate.column_indices(), shift);
                dg::blas1::plus( zero_interpolate.column_indices(), global_shift);
            }

            yp.fill(dg::evaluate( dg::zero, grid_fine_sub));
            ym = yp;
            for( int i=0; i<2; i++)
            {
                dg::blas2::symv( interpolate, yp_trafo[i], yp[i]);
                dg::blas2::symv( interpolate, ym_trafo[i], ym[i]);
            }
            ///%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%//
            if( sub <= 1 || sub >= grid_transform->local().Ny() -2 )
            {
                if( project_m == "dg")
                {
                    // Note that it is possible to project onto a bigger grid, the corresponding rows are just empty
                    projection = dg::create::projection( grid_transform->global(), grid_fine_sub);
                }
                else
                {
                    projection =
                        dg::create::projection( grid_equidist_global, grid_fine_sub, project_m);
                }
            }
            else
            {
                // how to add to rows in csr formatted matrix:
                projection.row_offsets().insert( projection.row_offsets().begin(),
                    global_shift, 0);
                projection.row_offsets().erase( projection.row_offsets().end() -
                    global_shift, projection.row_offsets().end());
            }

            std::array<dg::HVec*,3> xcomp{ &yp[0], &Xf, &ym[0]};
            std::array<dg::HVec*,3> ycomp{ &yp[1], &Yf, &ym[1]};
            std::array<dg::IHMatrix*,3> result{ &local_plus, &local_zero, &local_minus};
            dg::IHMatrix subresult;

            for( unsigned u=0; u<3; u++)
            {
                if( u == 1)
                    subresult = projection*zero_interpolate;
                else
                    subresult = projection*dg::create::interpolation( *xcomp[u], *ycomp[u],
                        inter_m == "dg" ? grid_transform->global() :
                        grid_equidist_global, bcx, bcy, inter_m);
                detail::add_from_sub( *result[u], subresult, project_m);
            }
        }
        if( benchmark)
        {
            t.toc();
            if(rank==0) std::cout << "# DS: Assembly of matrices  took: "<<t.diff()<<"\n";
            t.tic();
        }
        // Now convert to MPI matrices
        std::array<dg::IHMatrix*,3> input{ &local_plus, &local_zero, &local_minus};
        std::array<MIHMatrix*,3> result{ &plus, &zero, &minus};
        std::array<MIHMatrix*,3> resultT{ &plusT, &zero, &minusT};
        for( unsigned u=0; u<3; u++)
        {
            // important! only convert after addition (else the addition is wrong for linear projection)
            *input[u] = dg::convertGlobal2LocalRows( *input[u], *grid_transform);
            *result[u] = dg::make_mpi_matrix( *input[u], *grid_transform);
            if( make_adjoint and  u != 1)
            {
                dg::IHMatrix inputT = input[u]->transpose();
                // inputT is column distributed
                // inputT has global rows and local column indices
                dg::convertLocal2GlobalCols( inputT, *grid_transform);
                // now inputT has global rows and global column indices
                auto mat = dg::convertGlobal2LocalRows( inputT, *grid_transform);
                // now mat is row distributed with global column indices
                *resultT[u] = dg::make_mpi_matrix(  mat, *grid_transform);
            }
        }
        if( benchmark)
        {
            t.toc();
            if(rank==0)
            {
                std::cout << "# DS: MPI conversion        took: "<<t.diff()<<"\n";
                std::streamsize ss = std::cout.precision();
                std::cout << std::setprecision(1) << std::fixed;
                std::cout << "# DS: Average nnz per row inner (plus | zero | minus): "
                          <<(double)plus.inner_matrix().num_nnz()/(double)plus.inner_matrix().num_rows()<<" | "
                          <<(double)zero.inner_matrix().num_nnz()/(double)zero.inner_matrix().num_rows()<<" | "
                          <<(double)minus.inner_matrix().num_nnz()/(double)minus.inner_matrix().num_rows()<<"\n";
                if( plus.outer_matrix().num_nnz() != 0)
                    std::cout << "# DS: Average nnz per row outer (plus | zero | minus): "
                          <<(double)plus.outer_matrix().num_nnz()/(double)plus.outer_matrix().num_rows()<<" | "
                          <<(double)zero.outer_matrix().num_nnz()/(double)zero.outer_matrix().num_rows()<<" | "
                          <<(double)minus.outer_matrix().num_nnz()/(double)minus.outer_matrix().num_rows()<<"\n";
                std::cout << std::defaultfloat << std::setprecision(ss);
            }
        }
    } // make_matrices
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class MIMatrix, class LocalContainer>
template <class Limiter>
Fieldaligned<MPIGeometry, MIMatrix, MPI_Vector<LocalContainer> >::Fieldaligned(
    const dg::geo::CylindricalVectorLvl1& vec,
    const MPIGeometry& grid,
    dg::bc bcx, dg::bc bcy, Limiter limit, double eps,
    unsigned mx, unsigned my,
    double deltaPhi, std::string interpolation_method, bool benchmark
    ):
        m_g(grid), m_bcx(bcx), m_bcy(bcy), m_bcz(grid.bcz()),
        m_Nz( grid.local().Nz()), m_mx(mx), m_my(my), m_eps(eps),
        m_interpolation_method(interpolation_method),
        m_vec(vec)
{
    int rank;
    MPI_Comm_rank( grid.communicator(), &rank);
    int dims[3], periods[3], coords[3];
    dg::mpi_cart_get( m_g->communicator(), 3, dims, periods, coords);
    m_coords2 = coords[2], m_sizeZ = dims[2];

    ///Let us check boundary conditions:
    if( (grid.bcx() == PER && bcx != PER) || (grid.bcx() != PER && bcx == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting periodicity in x. The grid says "<<bc2str(grid.bcx())<<" while the parameter says "<<bc2str(bcx)));
    if( (grid.bcy() == PER && bcy != PER) || (grid.bcy() != PER && bcy == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting boundary conditions in y. The grid says "<<bc2str(grid.bcy())<<" while the parameter says "<<bc2str(bcy)));
    if( deltaPhi <=0) deltaPhi = grid.hz();
    m_deltaPhi = deltaPhi; // store for evaluate

    auto vol = dg::tensor::volume(grid.metric()), vol2d0(vol);
    auto vol2d = dg::split( vol, grid);
    dg::assign( vol2d[0], vol2d0);
    dg::ClonePtr<dg::aMPIGeometry2d> grid_transform( grid.perp_grid()) ;
    dg::ClonePtr<dg::aGeometry2d> global_grid_magnetic;
    std::array<thrust::host_vector<double>,3> yp_trafo, ym_trafo;
    thrust::host_vector<double> hbp, hbm;
    thrust::host_vector<bool> in_boxp, in_boxm;

    dg::MIHMatrix minus, zero, plus, minusT, plusT;
    make_matrices( vec, grid_transform, global_grid_magnetic,
            bcx, bcy, eps, mx, my, m_deltaPhi, interpolation_method,
            benchmark, vol2d0, hbp, hbm,
            in_boxp, in_boxm,
            yp_trafo, ym_trafo,
            minus, zero, plus, false, minusT, plusT
            );
    dg::blas2::transfer( minus, m_minus);
    dg::blas2::transfer( zero, m_zero);
    dg::blas2::transfer( plus, m_plus);
    m_have_adjoint = false;

    // 26.9.25: See performance comments on backproject on fieldaligned.h
    std::string inter_m, project_m, fine_m;
    detail::parse_method( interpolation_method, inter_m, project_m, fine_m);
    if( project_m != "dg")
    {
        apply_inv_backproject = true; // from equidist to dg
        m_inv_bx = dg::create::fast_transform( 0,
            dg::invert(dg::create::detail::square_backproject( grid.local().grid(0))),
            grid.local());
        m_inv_by = dg::create::fast_transform( 1,
            dg::invert(dg::create::detail::square_backproject( grid.local().grid(1))),
            grid.local());
    }
    if( inter_m != "dg")
    {
        apply_backproject = true; // from dg to equidist
        m_bx = dg::create::fast_transform( 0,
            dg::create::detail::square_backproject( grid.local().grid(0)),
            grid.local());
        m_by = dg::create::fast_transform( 1,
            dg::create::detail::square_backproject( grid.local().grid(1)),
            grid.local());
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
    dg::assign3dfrom2d( dg::MHVec(hbphi,  grid_transform->communicator()), m_bphi,  grid);
    dg::assign3dfrom2d( dg::MHVec(hbphiM, grid_transform->communicator()), m_bphiM, grid);
    dg::assign3dfrom2d( dg::MHVec(hbphiP, grid_transform->communicator()), m_bphiP, grid);

    dg::assign3dfrom2d( dg::MHVec(yp_trafo[2], grid_transform->communicator()), m_Gp, grid);
    dg::assign3dfrom2d( dg::MHVec(ym_trafo[2], grid_transform->communicator()), m_Gm, grid);
    MPI_Vector<LocalContainer> weights = dg::create::weights( grid);
    m_G = vol;
    dg::blas1::pointwiseDot( m_G, weights, m_G);
    dg::blas1::pointwiseDot( m_Gp, weights, m_Gp);
    dg::blas1::pointwiseDot( m_Gm, weights, m_Gm);

    dg::assign( dg::evaluate( dg::zero, grid), m_hbm);
    m_split_out = dg::split( m_hbm, grid); //3d vector
    m_split_in = dg::split( (const MPI_Vector<LocalContainer>&)m_hbm, grid);
    dg::assign3dfrom2d( dg::MHVec(hbp, grid_transform->communicator()), m_hbp, grid);
    dg::assign3dfrom2d( dg::MHVec(hbm, grid_transform->communicator()), m_hbm, grid);
    dg::blas1::scal( m_hbm, -1.);

    m_temp0 = m_temp1 = m_hbm; // 3d size
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
    dg::assign3dfrom2d( dg::MHVec(bbm, grid_transform->communicator()), m_bbm, grid);
    dg::assign3dfrom2d( dg::MHVec(bbo, grid_transform->communicator()), m_bbo, grid);
    dg::assign3dfrom2d( dg::MHVec(bbp, grid_transform->communicator()), m_bbp, grid);


    ///%%%%%%%%%%%%%%%%%%%%%Assign Limiter%%%%%%%%%%%%%%%%%%%%%%%%%//
    m_perp_size = grid_transform->local().size();
    dg::assign( dg::pullback(limit, *grid_transform), m_limiter);
    dg::assign( dg::evaluate(dg::zero, *grid_transform), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
    /// %%%%%%%%%%%%%%%%%%%%%%SETUP MPI in Z%%%%%%%%%%%%%%%%%%%%%%%%//
    int source, dest;
    dg::detail::MsgChunk chunk { 0, (int)m_perp_size};

    dg::mpi_cart_shift( grid.comm(2), 0, +1, &source, &dest);
    std::map<int, thrust::host_vector<dg::detail::MsgChunk>> recvMsgP =
        {{ dest, thrust::host_vector<dg::detail::MsgChunk>( 1, chunk)}};
    m_from_plus = dg::detail::MPIContiguousGather( recvMsgP, grid.comm(2));

    dg::mpi_cart_shift( grid.comm(2), 0, -1, &source, &dest);
    std::map<int, thrust::host_vector<dg::detail::MsgChunk>> recvMsgM =
        {{ dest, thrust::host_vector<dg::detail::MsgChunk>( 1, chunk)}};
    m_from_minus = dg::detail::MPIContiguousGather( recvMsgM, grid.comm(2));
}


template<class G, class M, class container>
void Fieldaligned<G, M, MPI_Vector<container> >::operator()(enum
        whichMatrix which, const MPI_Vector<container>& f,
        MPI_Vector<container>& fe) const
{
    if(     which == einsPlus  || which == einsMinusT ) ePlus(  which, f, fe);
    else if(which == einsMinus || which == einsPlusT  ) eMinus( which, f, fe);
    else if(which == zeroMinus || which == zeroPlus ||
            which == zeroMinusT|| which == zeroPlusT ||
            which == zeroForw  ) zero(   which, f, fe);
}
template< class G, class M, class container>
void Fieldaligned<G, M, MPI_Vector<container> >::zero( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& f0) const
{
    dg::split( f, m_split_in, *m_g);
    dg::split( f0, m_split_out, *m_g);
    if( apply_backproject)
    {
        dg::blas2::symv( m_bx, f.data(), m_temp0.data());
        dg::blas2::symv( m_by, m_temp0.data(), m_temp1.data());
        dg::split( (const MPI_Vector<container>&)m_temp1, m_split_in, *m_g);
    }
    //1. compute 2d interpolation in every plane and store in m_split_out
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        if(which == zeroPlus)
            dg::blas2::symv( m_plus,   m_split_in[i0], m_split_out[i0]);
        else if(which == zeroMinus)
            dg::blas2::symv( m_minus,  m_split_in[i0], m_split_out[i0]);
        else if(which == zeroPlusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_plusT,  m_split_in[i0], m_split_out[i0]);
        }
        else if(which == zeroMinusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_minusT, m_split_in[i0], m_split_out[i0]);
        }
        else if( which == zeroForw)
        {
            if ( m_interpolation_method != "dg" )
            {
                dg::blas2::symv( m_zero, m_split_in[i0], m_split_out[i0]);
            }
            else
                dg::blas1::copy( m_split_in[i0], m_split_out[i0]);
        }
    }
    if( apply_inv_backproject)
    {
        dg::blas2::symv( m_inv_bx, f0.data(), m_temp0.data());
        dg::blas2::symv( m_inv_by, m_temp0.data(), f0.data());
    }
}

template<class G, class M, class container>
void Fieldaligned<G,M, MPI_Vector<container> >::ePlus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fpe ) const
{
    dg::split( f, m_split_in, *m_g);
    dg::split( fpe, m_split_out, *m_g);
    if( apply_backproject)
    {
        dg::blas2::symv( m_bx, f.data(), m_temp0.data());
        dg::blas2::symv( m_by, m_temp0.data(), m_temp1.data());
        dg::split( (const MPI_Vector<container>&)m_temp1, m_split_in, *m_g);
    }
    MPI_Vector<dg::View<container>> send_buf(
            {m_ghostP.data().data(), m_ghostP.size()}, m_g->get_perp_comm());
    //1. compute 2d interpolation in every plane and store in m_split_out
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        // If communication necessary we write the symv result into send buffer
        bool comm_plane = (m_sizeZ != 1 and i0 == m_Nz -1);
        unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
        if(which == einsPlus)
            dg::blas2::symv( m_plus,   m_split_in[ip], comm_plane ? send_buf : m_split_out[i0]);
        else if(which == einsMinusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_minusT, m_split_in[ip], comm_plane ? send_buf : m_split_out[i0]);
        }
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = m_Nz-1;
        m_from_plus.global_gather_init( send_buf.data(), m_split_out[i0].data());
        m_from_plus.global_gather_wait( m_split_out[i0].data());
    }
    if( apply_inv_backproject)
    {
        dg::blas2::symv( m_inv_bx, fpe.data(), m_temp0.data());
        dg::blas2::symv( m_inv_by, m_temp0.data(), fpe.data());
    }

    //3. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER && m_g->local().z1() == m_g->global().z1())
    {
        if( m_bcz == dg::DIR || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( 2, m_right, -1., m_split_in[i0], m_ghostP);
        if( m_bcz == dg::NEU || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( m_deltaPhi, m_right, 1., m_split_in[i0], m_ghostP);
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_split_out[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_split_out[i0]);
    }
}

template<class G, class M, class container>
void Fieldaligned<G, M, MPI_Vector<container> >::eMinus( enum
    whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fme ) const
{
    int rank;
    MPI_Comm_rank(m_g->communicator(), &rank);
    dg::split( f, m_split_in, *m_g);
    dg::split( fme, m_split_out, *m_g);
    if( apply_backproject)
    {
        dg::blas2::symv( m_bx, f.data(), m_temp0.data());
        dg::blas2::symv( m_by, m_temp0.data(), m_temp1.data());
        dg::split( (const MPI_Vector<container>&)m_temp1, m_split_in, *m_g);
    }
    MPI_Vector<dg::View<container>> send_buf(
            {m_ghostM.data().data(), m_ghostM.size()}, m_g->get_perp_comm());
    //1. compute 2d interpolation in every plane and store in m_split_out
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        // If communication necessary we write the symv result into send buffer
        bool comm_plane = (m_sizeZ != 1 and i0 == 0);
        unsigned im = (i0==0) ? m_Nz-1:i0-1;
        if(which == einsPlusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_plusT, m_split_in[im], comm_plane ? send_buf : m_split_out[i0]);
        }
        else if(which == einsMinus)
            dg::blas2::symv( m_minus, m_split_in[im], comm_plane ? send_buf : m_split_out[i0]);
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = 0;
        m_from_minus.global_gather_init( send_buf.data(), m_split_out[i0].data());
        m_from_minus.global_gather_wait( m_split_out[i0].data());
    }
    if( apply_inv_backproject)
    {
        dg::blas2::symv( m_inv_bx, fme.data(), m_temp0.data());
        dg::blas2::symv( m_inv_by, m_temp0.data(), fme.data());
    }

    //3. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER && m_g->local().z0() == m_g->global().z0())
    {
        if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( 2., m_left,  -1., m_split_in[i0], m_ghostM);
        if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( -m_deltaPhi, m_left, 1., m_split_in[i0], m_ghostM);
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_split_out[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_split_out[i0]);
    }
}

template<class G, class M, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> Fieldaligned<G,M, MPI_Vector<container> >::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g->global().Nz());
    const dg::ClonePtr<aMPIGeometry2d> g2d = m_g->perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, *g2d);
    MPI_Vector<container> zero2d = dg::evaluate( dg::zero, *g2d);
    unsigned globalNz = m_g->global().Nz();
    typename detail::Container2EllSparseBlockMat<container>::type bx, by, inv_bx, inv_by;
    if( apply_backproject)
    {
        bx = dg::create::fast_transform( 0,
            dg::create::detail::square_backproject( g2d->local().grid(0)),
            g2d->local());
        by = dg::create::fast_transform( 1,
            dg::create::detail::square_backproject( g2d->local().grid(1)),
            g2d->local());
    }
    if( apply_inv_backproject)
    {
        inv_bx = dg::create::fast_transform( 0,
            dg::invert(dg::create::detail::square_backproject( g2d->local().grid(0))),
            g2d->local());
        inv_by = dg::create::fast_transform( 1,
            dg::invert(dg::create::detail::square_backproject( g2d->local().grid(1))),
            g2d->local());
    }

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
                if( apply_backproject)
                {
                    dg::blas2::symv( bx, tempP.data(), temp.data());
                    dg::blas2::symv( by, temp.data(), tempP.data());
                    dg::blas2::symv( bx, tempM.data(), temp.data());
                    dg::blas2::symv( by, temp.data(), tempM.data());
                }
                //!!! The value of f at the plus plane is I^- of the current plane
                dg::blas2::symv( m_minus, tempP, temp);
                if( apply_inv_backproject)
                {
                    dg::blas2::symv( inv_bx, temp.data(), tempP.data());
                    dg::blas2::symv( inv_by, tempP.data(), temp.data());
                }
                temp.swap( tempP);
                //!!! The value of f at the minus plane is I^+ of the current plane
                dg::blas2::symv( m_plus, tempM, temp);
                if( apply_inv_backproject)
                {
                    dg::blas2::symv( inv_bx, temp.data(), tempM.data());
                    dg::blas2::symv( inv_by, tempM.data(), temp.data());
                }
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
    mpi_cart_get( grid.communicator(), 3, dims, periods, coords);
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
