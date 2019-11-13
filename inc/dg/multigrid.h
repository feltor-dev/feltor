#pragma once

#include "backend/exceptions.h"
#include "backend/memory.h"
#include "topology/fast_interpolation.h"
#include "topology/interpolation.h"
#include "blas.h"
#include "cg.h"
#include "chebyshev.h"
#include "eve.h"
#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK
#ifdef MPI_VERSION
#include "topology/mpi_projection.h"
#endif

#include "dg/file/nc_utilities.h"
size_t start=0;
int ncid =0;
int xID=0, bID=0, rID=0, tvarID=0;
file::NC_Error_Handle err;

namespace dg
{

template<class Container, class Matrix, class Geometry>
void file_output( const std::vector<Container>& x,const std::vector<Container>& b, const std::vector<Container>& r, std::vector<Container>& out, unsigned p, std::vector<Matrix>& inter, Geometry& grid )
{
    size_t count = 1;
    double time = start;
    err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
    dg::HVec data = out[0];

    dg::blas1::copy( x[p], out[p]);
    for( int i=(int)p; i>0; i--)
        dg::blas2::gemv( inter[i-1], out[i], out[i-1]);
    data = out[0];
    file::put_vara_double( ncid, xID, start, grid, data);

    dg::blas1::copy( b[p], out[p]);
    for( int i=(int)p; i>0; i--)
        dg::blas2::gemv( inter[i-1], out[i], out[i-1]);
    data = out[0];
    file::put_vara_double( ncid, bID, start, grid, data);

    dg::blas1::copy( r[p], out[p]);
    for( int i=(int)p; i>0; i--)
        dg::blas2::gemv( inter[i-1], out[i], out[i-1]);
    data = out[0];
    file::put_vara_double( ncid, rID, start, grid, data);

    start++;
}

/**
* @brief Solves the Equation \f[ \frac{1}{W} \hat O \phi = \rho \f]
*
 * using a multigrid algorithm for any operator \f$\hat O\f$ that is symmetric
 * and appropriate weights \f$W\f$ (s. comment below).
*
* @snippet elliptic2d_b.cu multigrid
* We use conjugate gradient (CG) at each stage and refine the grids in the first two dimensions (2d / x and y)
 * @note A note on weights, inverse weights and preconditioning.
 * A normalized DG-discretized derivative or operator is normally not symmetric.
 * The diagonal coefficient matrix that is used to make the operator
 * symmetric is called weights W, i.e. \f$ \hat O = W\cdot O\f$ is symmetric.
 * Now, to compute the correct scalar product of the right hand side the
 * inverse weights have to be used i.e. \f$ W\rho\cdot W \rho /W\f$.
 * Independent from this, a preconditioner should be used to solve the
 * symmetric matrix equation.
* @note The preconditioner for the CG solver is taken from the \c precond() method in the \c SymmetricOp class
* @copydoc hide_geometry_matrix_container
* @ingroup multigrid
* @sa \c Extrapolation  to generate an initial guess
*
*/
template< class Geometry, class Matrix, class Container>
struct MultigridCG2d
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    MultigridCG2d(){}
    /**
     * @brief Construct the grids and the interpolation/projection operators
     *
     * @param grid the original grid (Nx() and Ny() must be evenly divisable by pow(2, stages-1)
     * @param stages number of grids in total (The second grid contains half the points of the original grids,
     *   The third grid contains half of the second grid ...). Must be > 1
     *   @param ps parameters necessary for \c dg::construct to construct a \c Container from a \c dg::HVec
    */
    template<class ...Params>
    MultigridCG2d( const Geometry& grid, const unsigned stages, Params&& ... ps)
    {
        construct( grid, stages, std::forward<Params>(ps)...);
    }
    template<class ...Params>
    void construct( const Geometry& grid, const unsigned stages, Params&& ... ps)
    {
        m_stages = stages;
        if(stages < 2 ) throw Error( Message(_ping_)<<" There must be minimum 2 stages in a multigrid solver! You gave " << stages);

		m_grids.resize(stages);
        m_cg.resize(stages);
        m_cheby.resize(stages);

        m_grids[0].reset( grid);
        //m_grids[0].get().display();

		for(unsigned u=1; u<stages; u++)
        {
            m_grids[u] = m_grids[u-1]; // deep copy
            m_grids[u]->multiplyCellNumbers(0.5, 0.5);
            //m_grids[u]->display();
        }

		m_inter.resize(stages-1);
		m_interT.resize(stages-1);
        m_project.resize( stages-1);

		for(unsigned u=0; u<stages-1; u++)
        {
            // Projecting from one grid to the next is the same as
            // projecting from the original grid to the coarse grids
            m_project[u].construct( dg::create::fast_projection(*m_grids[u], 2, 2, dg::normed), std::forward<Params>(ps)...);
            m_inter[u].construct( dg::create::fast_interpolation(*m_grids[u+1], 2, 2), std::forward<Params>(ps)...);
            m_interT[u].construct( dg::create::fast_projection(*m_grids[u], 2, 2, dg::not_normed), std::forward<Params>(ps)...);
        }

        m_x.resize( m_stages);
        for( unsigned u=0; u<m_stages; u++)
            m_x[u] = dg::construct<Container>( dg::evaluate( dg::zero, *m_grids[u]), std::forward<Params>(ps)...);
        m_r = m_b = m_x;
        m_p = m_cgr = m_r[0];
        for (unsigned u = 0; u < m_stages; u++)
        {
            m_cg[u].construct(m_x[u], 1);
            m_cg[u].set_max(m_grids[u]->size());
            m_cheby[u].construct(m_x[u]);
        }
    }

    /**
     * @brief Nested iterations
     *
     * - Compute residual with given initial guess.
     * - Project residual down to the coarsest grid.
     * - Solve equation on the coarse grid
     * - interpolate solution up to next finer grid and repeat until the original grid is reached.
     * @note The preconditioner for the CG solver is taken from the \c precond() method in the \c SymmetricOp class
     * @copydoc hide_symmetric_op
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     * @param op Index 0 is the \c SymmetricOp on the original grid, 1 on the half grid, 2 on the quarter grid, ...
     * @param x (read/write) contains initial guess on input and the solution on output
     * @param b The right hand side (will be multiplied by \c weights)
     * @param eps the accuracy: iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + 1) \f$
     * @return the number of iterations in each of the stages beginning with the finest grid
     * @note If the Macro \c DG_BENCHMARK is defined this function will write timings to \c std::cout
    */
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    std::vector<unsigned> direct_solve( std::vector<SymmetricOp>& op, ContainerType0&  x, const ContainerType1& b, double eps)
    {
        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        // compute residual r = Wb - A x
        dg::blas2::symv(op[0], x, m_r[0]);
        dg::blas1::axpby(-1.0, m_r[0], 1.0, m_b[0], m_r[0]);
        // project residual down to coarse grid
        for( unsigned u=0; u<m_stages-1; u++)
            dg::blas2::gemv( m_interT[u], m_r[u], m_r[u+1]);
        std::vector<unsigned> number(m_stages);
#ifdef DG_BENCHMARK
        Timer t;
#endif //DG_BENCHMARK

        dg::blas1::scal( m_x[m_stages-1], 0.0);
        //now solve residual equations
		for( unsigned u=m_stages-1; u>0; u--)
        {
#ifdef DG_BENCHMARK
            t.tic();
#endif //DG_BENCHMARK
            number[u] = m_cg[u]( op[u], m_x[u], m_r[u], op[u].precond(),
                op[u].inv_weights(), eps/2, 1.);
            dg::blas2::symv( m_inter[u-1], m_x[u], m_x[u-1]);
#ifdef DG_BENCHMARK
            t.toc();
#ifdef MPI_VERSION
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(rank==0)
#endif //MPI
            std::cout << "# Nested iterations stage: " << u << ", iter: " << number[u] << ", took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK

        }
#ifdef DG_BENCHMARK
        t.tic();
#endif //DG_BENCHMARK

        //update initial guess
        dg::blas1::axpby( 1., m_x[0], 1., x);
        number[0] = m_cg[0]( op[0], x, m_b[0], op[0].precond(),
            op[0].inv_weights(), eps);
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        std::cout << "# Nested iterations stage: " << 0 << ", iter: " << number[0] << ", took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK

        return number;
    }

    /**
    * @brief Project vector to all involved grids
    * @param src the input vector (may alias first element of out)
    * @param out the input vector projected to all grids ( index 0 contains a copy of src, 1 is the projetion to the first coarse grid, 2 is the next coarser grid, ...)
    * @note \c out is not resized
    */
    template<class ContainerType0>
    void project( const ContainerType0& src, std::vector<ContainerType0>& out)
    {
        dg::blas1::copy( src, out[0]);
        for( unsigned u=0; u<m_stages-1; u++)
            dg::blas2::gemv( m_project[u], out[u], out[u+1]);
    }

    /**
    * @brief Project vector to all involved grids (allocate memory version)
    * @param src the input vector
    * @return the input vector projected to all grids ( index 0 contains a copy of src, 1 is the projetion to the first coarse grid, 2 is the next coarser grid, ...)
    */
    template<class ContainerType0>
    std::vector<ContainerType0> project( const ContainerType0& src)
    {
        //use the fact that m_x has the correct sizes from the constructor
        std::vector<Container> out( m_x);
        project( src, out);
        return out;

    }
    ///@return number of stages (same as \c num_stages)
    unsigned stages()const{return m_stages;}
    ///@return number of stages (same as \c stages)
    unsigned num_stages()const{return m_stages;}

    ///observe the grids at all stages
    ///@param stage must fulfill \c 0<stage<stages()
    const Geometry& grid( unsigned stage) const {
        return *(m_grids[stage]);
    }


    ///After a call to a solution method returns the maximum number of iterations allowed at stage  0
    ///(if the solution method returns this number, failure is indicated)
    unsigned max_iter() const{return m_cg[0].get_max();}

    ///@brief Return an object of same size as the object used for construction on the finest grid
    ///@return A copyable object; what it contains is undefined, its size is important
    const Container& copyable() const {return m_x[0];}


	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    void solve( std::vector<SymmetricOp>& op,
    ContainerType0& x, const ContainerType1& b, std::vector<double> ev, unsigned nu_pre, unsigned
    nu_post, unsigned gamma, double eps)
    {
        //FULL MULTIGRID
        //solve for residuum ( if not we always get the same solution)
        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        value_type nrmb = sqrt( blas1::dot( m_b[0], b));

        dg::blas2::symv( op[0], x, m_r[0]);
        dg::blas1::axpby( -1., m_r[0], 1., m_b[0]);
        dg::blas1::copy( 0., m_x[0]);
        full_multigrid( op, m_x, m_b, ev, nu_pre, nu_post, gamma, 1, eps);
        dg::blas1::axpby( 1., m_x[0], 1., x);

        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        blas2::symv( op[0],x,m_r[0]);
        dg::blas1::axpby( -1., m_r[0], 1., m_b[0]);
        dg::blas1::copy( 0., m_x[0]);
        value_type error = sqrt( blas2::dot(op[0].inv_weights(),m_b[0]) );
        std::cout<< "# Relative Residual error is  "<<error/(nrmb+1)<<"\n";

        while ( error >  eps*(nrmb + 1))
        {
            //MULTIGRID CYCLES
            //dg::blas1::copy( x, m_x[0]);
            //multigrid_cycle( op, m_x, m_b, ev, nu_pre, nu_post, gamma, 0, eps);
            //dg::blas1::copy( m_x[0], x);
            //FMG cycles
            full_multigrid( op, m_x, m_b, ev, nu_pre, nu_post, gamma, 1, eps);
            dg::blas1::axpby( 1., m_x[0], 1., x);

            dg::blas2::symv(op[0].weights(), b, m_b[0]);
            blas2::symv( op[0],x,m_r[0]);
            dg::blas1::axpby( -1., m_r[0], 1., m_b[0]);
            dg::blas1::copy( 0., m_x[0]);
            error = sqrt( blas2::dot(op[0].inv_weights(),m_b[0]) );
            std::cout<< "# Relative Residual error is  "<<error/(nrmb+1)<<"\n";
        }
    }
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    void pcg_solve( std::vector<SymmetricOp>& op,
    ContainerType0& x, const ContainerType1& b, std::vector<double> ev, unsigned nu_pre, unsigned
    nu_post, unsigned gamma, double eps)
    {

        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        //PCG WITH MULTIGRID CYCLE AS PRECONDITIONER
        unsigned max_iter_ = m_grids[0]->size();
        value_type nrmb = sqrt( blas2::dot( op[0].inv_weights(), m_b[0]));
        if( nrmb == 0)
        {
            blas1::copy( m_b[0], x);
            return;
        }
        blas2::symv( op[0],x,m_cgr);
        blas1::axpby( 1., m_b[0], -1., m_cgr);
        //if x happens to be the solution
        if( sqrt( blas2::dot(op[0].inv_weights(),m_cgr) )
                < eps*(nrmb + 1))
            return;

        dg::blas1::copy( 0, m_x[0]);
        dg::blas1::copy( m_cgr, m_b[0]);
        //multigrid_cycle( op, m_x, m_b, ev, nu_pre, nu_post, gamma, 0, eps);
        full_multigrid( op,m_x, m_b, ev, nu_pre, nu_post, gamma, 1, eps);
        dg::blas1::copy( m_x[0], m_p);

        //and store the scalar product
        value_type nrmzr_old = blas1::dot( m_p,m_cgr);
        value_type alpha, nrmzr_new;
        for( unsigned i=2; i<max_iter_; i++)
        {
            blas2::symv( op[0], m_p, m_x[0]);
            alpha =  nrmzr_old/blas1::dot( m_p, m_x[0]);
            blas1::axpby( alpha, m_p, 1., x);
            blas1::axpby( -alpha, m_x[0], 1., m_cgr);
            value_type error = sqrt( blas2::dot(op[0].inv_weights(), m_cgr))/(nrmb+1);
            std::cout << "\t\t\tError at "<<i<<" is "<<error<<"\n";
            if( error < eps)
                return;
        dg::blas1::copy( 0, m_x[0]);
        dg::blas1::copy( m_cgr, m_b[0]);
        //multigrid_cycle( op, m_x, m_b, ev, nu_pre, nu_post, gamma, 0, eps);
        full_multigrid( op,m_x, m_b, ev, nu_pre, nu_post, gamma, 1, eps);

            nrmzr_new = blas1::dot( m_x[0], m_cgr);
            blas1::axpby(1., m_x[0], nrmzr_new/nrmzr_old, m_p );
            nrmzr_old=nrmzr_new;
        }

    }

    template<class SymmetricOp>
    void multigrid_cycle( std::vector<SymmetricOp>& op,
    std::vector<Container>& x, std::vector<Container>& b,
    std::vector<double> ev,
        unsigned nu1, unsigned nu2, unsigned gamma, unsigned p, double eps)
    {
        // 1 multigrid cycle beginning on grid p
        // p < m_stages-1
        // x[p]    initial condition on input, solution on output
        // x[p+1]  write only (solution of lower stage)
        // b[p]    read only,
        // b[p+1]  write only (residual after Pre-smooting at lower stage)
        // m_r[p]  write only
        //
        // gamma:  typically 1 (V-cycle) or 2 (W-cycle)
        // nu1, nu2: typically in {0,1,2,3}

        // 1. Pre-Smooth nu1 times
        //std::cout << "STAGE "<<p<<"\n";
        //dg::blas2::symv( op[p], x[p], m_r[p]);
        //dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //double norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //std::cout<< " Norm residuum befor "<<norm_res<<"\n";
#ifdef DG_BENCHMARK
        Timer t;
#endif //DG_BENCHMARK

        //std::vector<Container> out( x);
        //file_output( x, b, m_r, out, p, m_inter, *m_grids[0] );

        m_cheby[p].solve( op[p], x[p], b[p], 1e-2*ev[p], 1.1*ev[p], nu1);
        //m_cheby[p].solve( op[p], x[p], b[p], 0.1*ev[p], 1.1*ev[p], nu1, op[p].inv_weights());
        // 2. Residuum
        dg::blas2::symv( op[p], x[p], m_r[p]);
        dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //file_output( x, b, m_r, out, p, m_inter, *m_grids[0] );
        //norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //std::cout<< " Norm residuum after  "<<norm_res<<"\n";
        // 3. Coarsen
        dg::blas2::symv( m_interT[p], m_r[p], b[p+1]);
        // 4. Solve or recursive call to get x[p+1] with initial guess 0
        dg::blas1::scal( x[p+1], 0.);
        if( p+1 == m_stages-1)
        {
            //file_output( x, b, m_r, out, p+1, m_inter, *m_grids[0] );
#ifdef DG_BENCHMARK
            t.tic();
#endif //DG_BENCHMARK
            int number = m_cg[p+1]( op[p+1], x[p+1], b[p+1], op[p+1].precond(),
                op[p+1].inv_weights(), eps/2.);
#ifdef DG_BENCHMARK
            t.toc();
#ifdef MPI_VERSION
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(rank==0)
#endif //MPI
            std::cout << "# Multigrid stage: " << p+1 << ", iter: " << number << ", took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
            //file_output( x, b, m_r, out, p+1, m_inter, *m_grids[0] );
            //dg::blas2::symv( op[p+1], x[p+1], m_r[p+1]);
            //dg::blas1::axpby( 1., b[p+1], -1., m_r[p+1]);
            //double norm_res = sqrt(dg::blas1::dot( m_r[p+1], m_r[p+1]));
            //std::cout<< " Exact solution "<<norm_res<<"\n";
        }
        else
        {
            //update x[p+1] gamma times
            for( unsigned u=0; u<gamma; u++)
                multigrid_cycle( op, x, b, ev, nu1, nu2, gamma, p+1, eps);
        }

        // 5. Correct
        dg::blas2::symv( 1., m_inter[p], x[p+1], 1., x[p]);
        //dg::blas2::symv( op[p], x[p], m_r[p]);
        //dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //std::cout<< " Norm residuum befor "<<norm_res<<"\n";
        // 6. Post-Smooth nu2 times
        //file_output( x, b, m_r, out, p, m_inter, *m_grids[0] );
        m_cheby[p].solve( op[p], x[p], b[p], 1e-2*ev[p], 1.1*ev[p], nu2);
        //m_cheby[p].solve( op[p], x[p], b[p], 0.1*ev[p], 1.1*ev[p], nu2, op[p].inv_weights());
        //file_output( x, b, m_r, out, p, m_inter, *m_grids[0] );
        //dg::blas2::symv( op[p], x[p], m_r[p]);
        //dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //double norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //std::cout<< " Norm residuum after "<<norm_res<<"\n";
    }


	template<class SymmetricOp>
    void full_multigrid( std::vector<SymmetricOp>& op,
        std::vector<Container>& x, std::vector<Container>& b, std::vector<double> ev,
        unsigned nu1, unsigned nu2, unsigned gamma, unsigned mu, double eps)
    {
        for( unsigned u=0; u<m_stages-1; u++)
        {
            dg::blas2::gemv( m_interT[u], x[u], x[u+1]);
            dg::blas2::gemv( m_interT[u], b[u], b[u+1]);
        }
        //std::vector<Container> out( x);
        //begins on coarsest level and cycles through to highest
        unsigned s = m_stages-1;
        //file_output( x, b, m_r, out, s, m_inter, *m_grids[0] );
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        int number = m_cg[s]( op[s], x[s], b[s], op[s].precond(),
            op[s].inv_weights(), eps/2.);
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        std::cout << "# Multigrid stage: " << s << ", iter: " << number << ", took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK

        //file_output( x, b, m_r, out, s, m_inter, *m_grids[0] );
		for( int p=m_stages-2; p>=0; p--)
        {
            dg::blas2::gemv( m_inter[p], x[p+1],  x[p]);
            for( unsigned u=0; u<mu; u++)
                multigrid_cycle( op, x, b, ev, nu1, nu2, gamma, p, eps);
        }
    }

  private:
    unsigned m_stages;
    std::vector< dg::ClonePtr< Geometry> > m_grids;
    std::vector< MultiMatrix<Matrix, Container> >  m_inter;
    std::vector< MultiMatrix<Matrix, Container> >  m_interT;
    std::vector< MultiMatrix<Matrix, Container> >  m_project;
    std::vector< CG<Container> > m_cg;
    std::vector< Chebyshev<Container>> m_cheby;
    std::vector< Container> m_x, m_r, m_b;
    Container  m_p, m_cgr;

};

}//namespace dg
