#pragma once

#include "backend/exceptions.h"
#include "backend/memory.h"
#include "topology/fast_interpolation.h"
#include "topology/interpolation.h"
#include "blas.h"
#include "cg.h"
#include "chebyshev.h"
#include "eve.h"
#include "backend/timer.h"
#ifdef MPI_VERSION
#include "topology/mpi_projection.h"
#endif

namespace dg
{

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
    ///@brief Allocate nothing, Call \c construct method before usage
    MultigridCG2d(){}
    /**
     * @brief Construct the grids and the interpolation/projection operators
     *
     * @param grid the original grid (Nx() and Ny() must be evenly divisable by pow(2, stages-1)
     * @param stages number of grids in total (The second grid contains half the points of the original grids,
     *   The third grid contains half of the second grid ...). Must be > 1
     * @param ps parameters necessary for \c dg::construct to construct a \c Container from a \c dg::HVec
    */
    template<class ...Params>
    MultigridCG2d( const Geometry& grid, const unsigned stages, Params&& ... ps):
        m_stages(stages),
        m_grids( stages),
        m_inter(    stages-1),
        m_interT(   stages-1),
        m_project(  stages-1),
        m_cg(    stages),
        m_cheby( stages),
        m_x( stages)
    {
        if(stages < 2 )
            throw Error( Message(_ping_)<<" There must be minimum 2 stages in a multigrid solver! You gave " << stages);

        m_grids[0].reset( grid);
        //m_grids[0].get().display();

		for(unsigned u=1; u<stages; u++)
        {
            m_grids[u] = m_grids[u-1]; // deep copy
            m_grids[u]->multiplyCellNumbers(0.5, 0.5);
            //m_grids[u]->display();
        }

		for(unsigned u=0; u<stages-1; u++)
        {
            // Projecting from one grid to the next is the same as
            // projecting from the original grid to the coarse grids
            m_project[u].construct( dg::create::fast_projection(*m_grids[u], 1, 2, 2, dg::normed), std::forward<Params>(ps)...);
            m_inter[u].construct( dg::create::fast_interpolation(*m_grids[u+1], 1, 2, 2), std::forward<Params>(ps)...);
            m_interT[u].construct( dg::create::fast_projection(*m_grids[u], 1, 2, 2, dg::not_normed), std::forward<Params>(ps)...);
        }

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
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MultigridCG2d( std::forward<Params>( ps)...);
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

    ///@brief return the grid at given stage
    ///@param stage must fulfill \c 0 <= stage < stages()
    const Geometry& grid( unsigned stage) const {
        return *(m_grids[stage]);
    }


    ///The maximum number of iterations allowed at stage 0
    ///(if the solution method returns this number, failure is indicated)
    unsigned max_iter() const{return m_cg[0].get_max();}
    /**
     * @brief Set the maximum number of iterations allowed at stage 0
     *
     * By default this number is the grid size. However, for large
     * simulations you may want to prevent the solver from iterating to that number
     * in case of failure.
     * @param new_max new maximum number of iterations allowed at stage 0
    */
    void set_max_iter(unsigned new_max){ m_cg[0].set_max(new_max);}
    ///@brief Set or unset performance timings during iterations
    ///@param benchmark If true, additional output will be written to \c std::cout during solution
    void set_benchmark( bool benchmark){ m_benchmark = benchmark;}

    ///@brief Return an object of same size as the object used for construction on the finest grid
    ///@return A copyable object; what it contains is undefined, its size is important
    const Container& copyable() const {return m_x[0];}
    /**
     * @brief USE THIS %ONE Nested iterations
     *
     * Equivalent to the following
     * -# Compute residual with given initial guess.
     * -# Project residual down to the coarsest grid.
     * -# Solve equation on the coarse grid.
     * -# interpolate solution up to next finer grid and repeat 3 and 4 until the original grid is reached.
     * @note The preconditioner for the CG solver is taken from the \c precond() method in the \c SymmetricOp class
     * @copydoc hide_symmetric_op
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     * @param op Index 0 is the \c SymmetricOp on the original grid, 1 on the half grid, 2 on the quarter grid, ...
     * @param x (read/write) contains initial guess on input and the solution on output (if the initial guess is good enough the solve may return immediately)
     * @param b The right hand side (will be multiplied by \c weights)
     * @param eps the accuracy: iteration stops if \f$ ||b - Ax|| < \epsilon(
     * ||b|| + 1) \f$. If needed (and it is recommended to tune these values)
     * the accuracy can be set for each stage separately. Per default the same
     * accuracy is used at all stages.
     * @return the number of iterations in each of the stages beginning with the finest grid
     * @note the convergence test on the coarse grids is only evaluated every
     * 10th iteration. This effectively saves one dot product per iteration.
     * The dot product is the main performance bottleneck on the coarse grids.
    */
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    std::vector<unsigned> direct_solve( std::vector<SymmetricOp>& op, ContainerType0&  x, const ContainerType1& b, value_type eps)
    {
        std::vector<value_type> v_eps( m_stages, eps);
		for( unsigned u=m_stages-1; u>0; u--)
            v_eps[u] = 1.5*eps;
        return direct_solve( op, x, b, v_eps);
    }
    ///@copydoc direct_solve()
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    std::vector<unsigned> direct_solve( std::vector<SymmetricOp>& op, ContainerType0&  x, const ContainerType1& b, std::vector<value_type> eps)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        std::vector<unsigned> number(m_stages, 0);
        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        value_type nrmb = sqrt( blas2::dot( op[0].inv_weights(), m_b[0]));
        if( nrmb == 0)
        {
            blas1::copy( 0., x);
            return number;
        }
        // compute residual r = Wb - A x
        dg::blas2::symv(op[0], x, m_r[0]);
        dg::blas1::axpby(-1.0, m_r[0], 1.0, m_b[0], m_r[0]);
        if( sqrt( blas2::dot(op[0].inv_weights(),m_r[0]) ) < eps[0]*(nrmb+1.)) //if x happens to be the solution
            return number;
        // project residual down to coarse grid
        for( unsigned u=0; u<m_stages-1; u++)
            dg::blas2::gemv( m_interT[u], m_r[u], m_r[u+1]);

        dg::blas1::copy( 0., m_x[m_stages-1]);
        //now solve residual equations
		for( unsigned u=m_stages-1; u>0; u--)
        {
            if(m_benchmark)m_timer.tic();
            number[u] = m_cg[u].solve( op[u], m_x[u], m_r[u], op[u].precond(),
                op[u].inv_weights(), eps[u], 1., 10);
            dg::blas2::symv( m_inter[u-1], m_x[u], m_x[u-1]);
            if( m_benchmark)
            {
                m_timer.toc();
                DG_RANK0 std::cout << "# Nested iterations stage: " << u << ", iter: " << number[u] << ", took "<<m_timer.diff()<<"s\n";
            }

        }
        if( m_benchmark) m_timer.tic();

        //update initial guess
        dg::blas1::axpby( 1., m_x[0], 1., x);
        number[0] = m_cg[0].solve( op[0], x, m_b[0], op[0].precond(),
            op[0].inv_weights(), eps[0]);
        if( m_benchmark)
        {
            m_timer.toc();
            DG_RANK0 std::cout << "# Nested iterations stage: " << 0 << ", iter: " << number[0] << ", took "<<m_timer.diff()<<"s\n";
        }

        return number;
    }

    /**
     * @brief EXPERIMENTAL Nested iterations with Chebyshev as preconditioner for CG
     *
     * @note This function does the same as direct_solve but uses a
     * ChebyshevPreconditioner (with EVE to estimate the largest EV) at the coarse
     * grid levels (but not the fine level).
     * @copydetails direct_solve()
     * @param num_cheby Number of chebyshev iterations. If needed can be set
     * for each stage separately. Per default it is the same for all stages.
     * The 0 stage does not use the Chebyshev preconditioner, therefore
     * num_cheby[0] will be ignored.
    */
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    std::vector<unsigned> direct_solve_with_chebyshev( std::vector<SymmetricOp>& op, ContainerType0&  x, const ContainerType1& b, value_type eps, unsigned num_cheby)
    {
        std::vector<value_type> v_eps( m_stages, eps);
        std::vector<unsigned> v_num_cheby( m_stages, num_cheby);
        v_num_cheby[0] = 0;
		for( unsigned u=m_stages-1; u>0; u--)
            v_eps[u] = 1.5*eps;
        return direct_solve_with_chebyshev( op, x, b, v_eps, v_num_cheby);
    }
    ///@copydoc direct_solve_with_chebyshev()
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    std::vector<unsigned> direct_solve_with_chebyshev( std::vector<SymmetricOp>& op, ContainerType0&  x, const ContainerType1& b, std::vector<value_type> eps, std::vector<unsigned> num_cheby)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        // compute residual r = Wb - A x
        dg::blas2::symv(op[0], x, m_r[0]);
        dg::blas1::axpby(-1.0, m_r[0], 1.0, m_b[0], m_r[0]);
        // project residual down to coarse grid
        for( unsigned u=0; u<m_stages-1; u++)
            dg::blas2::gemv( m_interT[u], m_r[u], m_r[u+1]);
        std::vector<unsigned> number(m_stages);

        dg::blas1::scal( m_x[m_stages-1], 0.0);
        //now solve residual equations
		for( unsigned u=m_stages-1; u>0; u--)
        {
            if(m_benchmark) m_timer.tic();
            unsigned lowest = u;
            dg::EVE<Container> eve( m_x[lowest]);
            double evu_max;
            Container tmp = m_x[lowest];
            dg::blas1::scal( tmp, 0.);
            //unsigned counter = eve( op[lowest], tmp, m_r[lowest], op[u].precond(), evu_max, 1e-10);
            unsigned counter = eve.solve( op[lowest], tmp, m_r[lowest], evu_max, 1e-10);
            counter++;
            //DG_RANK0 std::cout << "# MAX EV is "<<evu_max<<" in "<<counter<<" iterations\t";
            //    m_timer.toc();
            //    DG_RANK0 std::cout << " took "<<m_timer.diff()<<"s\n";
            //    m_timer.tic();

            //double evu_min;
            //dg::detail::WrapperSpectralShift<SymmetricOp, Container> shift(
            //        op[u], evu_max);
            //counter = eve.solve( shift, m_x[u], m_r[u], evu_min, eps);
            //evu_min = evu_max - evu_min;
            //DG_RANK0 std::cout << "# MIN EV is "<<evu_min<<" in "<<counter<<"iterations\n";
            dg::ChebyshevPreconditioner<SymmetricOp&, Container> precond(
                    op[u], m_x[u], 0.01*evu_max, 1.1*evu_max, num_cheby[u] );
            //dg::ModifiedChebyshevPreconditioner<SymmetricOp&, Container> precond(
            //        op[u], m_x[u], evu_max/5./(num_cheby[u]), evu_max, num_cheby[u] );
            //dg::LeastSquaresPreconditioner<SymmetricOp&, const Container&, Container> precond(
            //        op[u], op[u].precond(), m_x[u], evu_max, num_cheby );
            number[u] = m_cg[u].solve( op[u], m_x[u], m_r[u], precond,
                op[u].inv_weights(), eps[u], 1., 10);
            dg::blas2::symv( m_inter[u-1], m_x[u], m_x[u-1]);
            if( m_benchmark)
            {
                m_timer.toc();
                DG_RANK0 std::cout << "# Nested iterations stage: " << u << ", iter: " << number[u] << ", took "<<m_timer.diff()<<"s\n";
            }

        }
        if(m_benchmark)m_timer.tic();
        //unsigned lowest = 0;
        //dg::EVE<Container> eve.solve( m_x[lowest]);
        //double evu_max;
        //Container tmp = m_x[lowest];
        //dg::blas1::scal( tmp, 0.);
        ////unsigned counter = eve.solve( op[lowest], tmp, m_r[lowest], op[u].precond(), evu_max, 1e-10);
        //unsigned counter = eve.solve( op[lowest], tmp, m_r[lowest], evu_max, 1e-10);
        //counter++;

        //dg::ChebyshevPreconditioner<SymmetricOp&, Container> precond(
        //        op[0], m_x[0], 0.01*evu_max, 1.1*evu_max, num_cheby[0] );
        //update initial guess
        dg::blas1::axpby( 1., m_x[0], 1., x);
        number[0] = m_cg[0].solve( op[0], x, m_b[0], op[0].precond(),
                //precond,
            op[0].inv_weights(), eps[0]);
        if( m_benchmark)
        {
            m_timer.toc();
            DG_RANK0 std::cout << "# Nested iterations stage: " << 0 << ", iter: " << number[0] << ", took "<<m_timer.diff()<<"s\n";
        }

        return number;
    }

    /**
     * @brief EXPERIMENTAL Full multigrid cycles (use at own risk)
     *
     * - Compute residual with given initial guess.
     * - If error larger than tolerance, do a full multigrid cycle with Chebeyshev iterations as smoother
     * - repeat
     * @note The preconditioner for the CG solver is taken from the \c precond() method in the \c SymmetricOp class
     * @copydoc hide_symmetric_op
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     * @param op Index 0 is the \c SymmetricOp on the original grid, 1 on the half grid, 2 on the quarter grid, ...
     * @param x (read/write) contains initial guess on input and the solution on output
     * @param b The right hand side (will be multiplied by \c weights)
     * @param ev The estimate of the largest Eivenvalue for each stage
     * @param nu_pre number of pre-smoothing steps (make it >10)
     * @param nu_post number of post-smoothing steps (make it >10)
     * @param gamma The shape of the multigrid ( 1 is usually ok)
     * @param eps the accuracy: iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + 1) \f$
     * @attention This method is rather unreliable, it only converges if the
     * parameters are chosen correctly ( there need to be enough smooting steps
     * for instance, and a large jump  factor in the Elliptic class also seems
     * to help) and otherwise just iterates to infinity. This behaviour is probably related to the use of the Chebyshev solver as a smoother
    */
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    void fmg_solve( std::vector<SymmetricOp>& op,
    ContainerType0& x, const ContainerType1& b, std::vector<value_type> ev, unsigned nu_pre, unsigned
    nu_post, unsigned gamma, value_type eps)
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
        //DG_RANK0 std::cout<< "# Relative Residual error is  "<<error/(nrmb+1)<<"\n";

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
            //DG_RANK0 std::cout<< "# Relative Residual error is  "<<error/(nrmb+1)<<"\n";
        }
    }

    /**
     * @brief EXPERIMENTAL A conjugate gradient with a full multigrid cycle as preconditioner (use at own risk)
     *
     * @copydoc hide_symmetric_op
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     * @param op Index 0 is the \c SymmetricOp on the original grid, 1 on the half grid, 2 on the quarter grid, ...
     * @param x (read/write) contains initial guess on input and the solution on output
     * @param b The right hand side (will be multiplied by \c weights)
     * @param ev The estimate of the largest Eivenvalue for each stage
     * @param nu_pre number of pre-smoothing steps (make it >10)
     * @param nu_post number of post-smoothing steps (make it >10)
     * @param gamma The shape of the multigrid ( 1 is usually ok)
     * @param eps the accuracy: iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + 1) \f$
     * @attention This method is rather unreliable, it only converges if the
     * parameters are chosen correctly ( there need to be enough smooting steps
     * for instance, and a large jump  factor in the Elliptic class also seems
     * to help) and otherwise just iterates to infinity
    */
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    void pcg_solve( std::vector<SymmetricOp>& op,
    ContainerType0& x, const ContainerType1& b, std::vector<value_type> ev, unsigned nu_pre, unsigned
    nu_post, unsigned gamma, value_type eps)
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
            //DG_RANK0 std::cout << "\t\t\tError at "<<i<<" is "<<error<<"\n";
            if( error < eps)
                return;
            dg::blas1::copy( 0, m_x[0]);
            dg::blas1::copy( m_cgr, m_b[0]);
            full_multigrid( op,m_x, m_b, ev, nu_pre, nu_post, gamma, 1, eps);

            nrmzr_new = blas1::dot( m_x[0], m_cgr);
            blas1::axpby(1., m_x[0], nrmzr_new/nrmzr_old, m_p );
            nrmzr_old=nrmzr_new;
        }

    }
  private:
    template<class SymmetricOp>
    void multigrid_cycle( std::vector<SymmetricOp>& op,
    std::vector<Container>& x, std::vector<Container>& b,
    std::vector<value_type> ev,
        unsigned nu1, unsigned nu2, unsigned gamma, unsigned p, value_type eps)
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
        //DG_RANK0 std::cout << "STAGE "<<p<<"\n";
        //dg::blas2::symv( op[p], x[p], m_r[p]);
        //dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //value_type norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //DG_RANK0 std::cout<< " Norm residuum befor "<<norm_res<<"\n";

        //std::vector<Container> out( x);

        m_cheby[p].solve( op[p], x[p], b[p], 1e-2*ev[p], 1.1*ev[p], nu1);
        //m_cheby[p].solve( op[p], x[p], b[p], 0.1*ev[p], 1.1*ev[p], nu1, op[p].inv_weights());
        // 2. Residuum
        dg::blas2::symv( op[p], x[p], m_r[p]);
        dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //DG_RANK0 std::cout<< " Norm residuum after  "<<norm_res<<"\n";
        // 3. Coarsen
        dg::blas2::symv( m_interT[p], m_r[p], b[p+1]);
        // 4. Solve or recursive call to get x[p+1] with initial guess 0
        dg::blas1::scal( x[p+1], 0.);
        if( p+1 == m_stages-1)
        {
//if( m_benchmark) m_timer.tic();
            int number = m_cg[p+1].solve( op[p+1], x[p+1], b[p+1], op[p+1].precond(),
                op[p+1].inv_weights(), eps/2.);
            number++;//avoid compiler warning
//if( m_benchmark){
//            m_timer.toc();
            //DG_RANK0 std::cout << "# Multigrid stage: " << p+1 << ", iter: " << number << ", took "<<m_timer.diff()<<"s\n";
            //}
            //dg::blas2::symv( op[p+1], x[p+1], m_r[p+1]);
            //dg::blas1::axpby( 1., b[p+1], -1., m_r[p+1]);
            //value_type norm_res = sqrt(dg::blas1::dot( m_r[p+1], m_r[p+1]));
            //DG_RANK0 std::cout<< " Exact solution "<<norm_res<<"\n";
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
        //DG_RANK0 std::cout<< " Norm residuum befor "<<norm_res<<"\n";
        // 6. Post-Smooth nu2 times
        m_cheby[p].solve( op[p], x[p], b[p], 1e-2*ev[p], 1.1*ev[p], nu2);
        //m_cheby[p].solve( op[p], x[p], b[p], 0.1*ev[p], 1.1*ev[p], nu2, op[p].inv_weights());
        //dg::blas2::symv( op[p], x[p], m_r[p]);
        //dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        //value_type norm_res = sqrt(dg::blas1::dot( m_r[p], m_r[p]));
        //DG_RANK0 std::cout<< " Norm residuum after "<<norm_res<<"\n";
    }


	template<class SymmetricOp>
    void full_multigrid( std::vector<SymmetricOp>& op,
        std::vector<Container>& x, std::vector<Container>& b, std::vector<value_type> ev,
        unsigned nu1, unsigned nu2, unsigned gamma, unsigned mu, value_type eps)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        for( unsigned u=0; u<m_stages-1; u++)
        {
            dg::blas2::gemv( m_interT[u], x[u], x[u+1]);
            dg::blas2::gemv( m_interT[u], b[u], b[u+1]);
        }
        //std::vector<Container> out( x);
        //begins on coarsest level and cycles through to highest
        unsigned s = m_stages-1;
        if( m_benchmark) m_timer.tic();
        int number = m_cg[s].solve( op[s], x[s], b[s], op[s].precond(),
            op[s].inv_weights(), eps/2.);
        number++;//avoid compiler warning
        if( m_benchmark)
        {
            m_timer.toc();
            DG_RANK0 std::cout << "# Multigrid stage: " << s << ", iter: " << number << ", took "<<m_timer.diff()<<"s\n";
        }

		for( int p=m_stages-2; p>=0; p--)
        {
            dg::blas2::gemv( m_inter[p], x[p+1],  x[p]);
            for( unsigned u=0; u<mu; u++)
                multigrid_cycle( op, x, b, ev, nu1, nu2, gamma, p, eps);
        }
    }
    unsigned m_stages;
    std::vector< dg::ClonePtr< Geometry> > m_grids;
    std::vector< MultiMatrix<Matrix, Container> >  m_inter;
    std::vector< MultiMatrix<Matrix, Container> >  m_interT;
    std::vector< MultiMatrix<Matrix, Container> >  m_project;
    std::vector< CG<Container> > m_cg;
    std::vector< ChebyshevIteration<Container>> m_cheby;
    std::vector< Container> m_x, m_r, m_b;
    Container  m_p, m_cgr;
    Timer m_timer;
    bool m_benchmark = true;

};

}//namespace dg
