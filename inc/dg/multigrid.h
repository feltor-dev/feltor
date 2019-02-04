#pragma once

#include "backend/exceptions.h"
#include "backend/memory.h"
#include "topology/fast_interpolation.h"
#include "topology/interpolation.h"
#include "blas.h"
#include "cg.h"
#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK
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
        for (unsigned u = 0; u < m_stages; u++)
            m_cg[u].construct(m_x[u], 1);
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
            m_cg[u].set_max(m_grids[u]->size());
            number[u] = m_cg[u]( op[u], m_x[u], m_r[u], op[u].precond(), op[u].inv_weights(), eps/2, 1.);
            dg::blas2::symv( m_inter[u-1], m_x[u], m_x[u-1]);
#ifdef DG_BENCHMARK
            t.toc();
#ifdef MPI_VERSION
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(rank==0)
#endif //MPI
            std::cout << "stage: " << u << ", iter: " << number[u] << ", took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK

        }
#ifdef DG_BENCHMARK
        t.tic();
#endif //DG_BENCHMARK

        //update initial guess
        dg::blas1::axpby( 1., m_x[0], 1., x);
        m_cg[0].set_max(m_grids[0]->size());
        number[0] = m_cg[0]( op[0], x, m_b[0], op[0].precond(), op[0].inv_weights(), eps);
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        std::cout << "stage: " << 0 << ", iter: " << number[0] << ", took "<<t.diff()<<"s\n";
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

  private:

    /*
	template<class SymmetricOp, class ContainerType0, class ContainerType1>
    void full_multigrid( std::vector<SymmetricOp>& op,
        ContainerType0& x, ContainerType1& b,
        unsigned nu1, unsigned nu2, unsigned gamma, unsigned mu, double eps)
    {
        dg::blas2::symv(op[0].weights(), b, m_b[0]);
        // project x down to coarse grid
        dg::blas1::copy( x, m_x[0]);
        dg::blas1::copy( b, m_b[0]);
        for( unsigned u=0; u<m_stages-1; u++)
        {
            dg::blas2::gemv( m_interT[u], m_x[u], m_x[u+1]);
            dg::blas2::gemv( m_interT[u], m_b[u], m_b[u+1]);
        }
        solve( op[m_stages-1], m_x[m_stages-1], m_b[m_stages-1], eps);

		for( int p=m_stages-2; p>=0; p--)
        {
            dg::blas2::gemv( m_inter[p], m_x[p+1],  m_x[p]);
            for( unsigned u=0; u<mu; u++)
                multigrid_cycle( op, m_x, m_b, nu1, nu2, gamma, p, eps);
        }
    }

    template<class SymmetricOp>
    void multigrid_cycle( std::vector<SymmetricOp>& op,
        std::vector<Container>& x, std::vector<Container>& b,
        unsigned nu1, unsigned nu2, unsigned gamma, unsigned p, double eps)
    {
        // p < m_stages-1
        // x[p]   initial condition on input, solution on output
        // x[p+1] write only (solution of lower stage)
        // b[p]   read only,
        // b[p+1] write only (residual after Pre-smooting at lower stage)
        // m_r[p] write only
        // gamma: typically 1 (V-cycle) or 2 (W-cycle)
        // nu1, nu2: typically in {0,1,2,3}

        // 1. Pre-Smooth
        smooth( op[p], x[p], b[p]); //nu1 times
        // 2. Residual
        dg::blas2:symv( op[p], x[p], m_r[p]);
        dg::blas1::axpby( 1., b[p], -1., m_r[p]);
        // 3. Coarsen
        dg::blas2::symv( m_interT[p], m_r[p], b[p+1]);
        // 4. Solve or recursive call to get x[p+1] with initial guess 0
        dg::blas1::scal( x[p+1], 0.);
        if( p+1 == m_stages-1)
            solve ( op[p+1], x[p+1], b[p+1], eps);
        else
        {
            //update x[p+1] gamma times
            for( unsigned u=0; u<gamma; u++)
                multigrid_cycle( op, x, b, nu1, nu2, gamma, p+1, eps);
        }

        // 5. Correct
        dg::blas2::symv( 1., m_inter[p], x[p+1], 1., x[p]);
        // 6. Post-Smooth
        smooth( op[p], x[p], b[p]); //nu2 times
    }
    */

private:
    unsigned m_stages;
    std::vector< dg::ClonePtr< Geometry> > m_grids;
    std::vector< MultiMatrix<Matrix, Container> >  m_inter;
    std::vector< MultiMatrix<Matrix, Container> >  m_interT;
    std::vector< MultiMatrix<Matrix, Container> >  m_project;
    std::vector< CG<Container> > m_cg;
    std::vector< Container> m_x, m_r, m_b;

};

}//namespace dg
