#pragma once

#include "backend/exceptions.h"
#include "backend/memory.h"
#include "topology/fast_interpolation.h"
#include "topology/interpolation.h"
#include "blas.h"
#include "pcg.h"
#include "chebyshev.h"
#include "eve.h"
#include "backend/timer.h"
#ifdef MPI_VERSION
#include "topology/mpi_projection.h"
#endif

namespace dg
{

///@addtogroup multigrid
///@{

/**
 * @brief Hold nested grids and provide dg fast interpolation and projection matrices
 *
 * @copydoc hide_geometry_matrix_container
 */
template<class Geometry, class Matrix, class Container>
struct NestedGrids
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief Allocate nothing, Call \c construct method before usage
    NestedGrids(): m_stages(0), m_grids(0), m_inter(0), m_project(0){}
    /**
     * @brief Construct the grids and the interpolation/projection operators
     *
     * @param grid the original grid (Nx() and Ny() must be evenly divisable by pow(2, stages-1)
     * @param stages number of grids in total (The second grid contains half the points of the original grids,
     *   The third grid contains half of the second grid ...). Must be >= 1
     * @param ps parameters necessary for \c dg::construct to construct a \c Container from a \c dg::HVec
    */
    template<class ...ContainerParams>
    NestedGrids( const Geometry& grid, const unsigned stages, ContainerParams&& ...ps):
        m_stages(stages),
        m_grids( stages),
        m_x( stages)
    {
        if(stages < 1 )
            throw Error( Message(_ping_)<<" There must be minimum 1 stage in nested Grids construction! You gave " << stages);
        m_grids[0].reset( grid);
        //m_grids[0].get().display();

		for(unsigned u=1; u<stages; u++)
        {
            m_grids[u] = m_grids[u-1]; // deep copy
            m_grids[u]->multiplyCellNumbers(0.5, 0.5);
            //m_grids[u]->display();
        }

        m_inter.resize(    stages-1);
        m_project.resize(  stages-1);
		for(unsigned u=0; u<stages-1; u++)
        {
            // Projecting from one grid to the next is the same as
            // projecting from the original grid to the coarse grids
            m_project[u].construct( dg::create::fast_projection(*m_grids[u], 1,
                        2, 2), std::forward<ContainerParams>(ps)...);
            m_inter[u].construct( dg::create::fast_interpolation(*m_grids[u+1],
                        1, 2, 2), std::forward<ContainerParams>(ps)...);
        }
        for( unsigned u=0; u<m_stages; u++)
            m_x[u] = dg::construct<Container>( dg::evaluate( dg::zero,
                        *m_grids[u]), std::forward<ContainerParams>(ps)...);
        m_w = m_r = m_b = m_x;

    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = NestedGrids( std::forward<Params>( ps)...);
    }
    /**
    * @brief Project vector to all involved grids
    * @param src the input vector (may alias first element of out)
    * @param out the input vector projected to all grids ( index 0 contains a copy of src, 1 is the projetion to the first coarse grid, 2 is the next coarser grid, ...)
    * @note \c out is not resized
    */
    template<class ContainerType0>
    void project( const ContainerType0& src, std::vector<ContainerType0>& out) const
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
    std::vector<ContainerType0> project( const ContainerType0& src) const
    {
        //use the fact that m_x has the correct sizes from the constructor
        std::vector<Container> out( m_x);
        project( src, out);
        return out;

    }
    ///@brief Return an object of same size as the object used for construction on the finest grid
    ///@return A copyable object; what it contains is undefined, its size is important
    const Container& copyable() const {return m_x[0];}

    ///@return number of stages (same as \c num_stages)
    unsigned stages()const{return m_stages;}
    ///@return number of stages (same as \c stages)
    unsigned num_stages()const{return m_stages;}

    ///@brief return the grid at given stage
    ///@param stage must fulfill \c 0 <= stage < stages()
    const Geometry& grid( unsigned stage) const {
        return *(m_grids[stage]);
    }

    ///@brief return the interpolation matrix at given stage
    ///@param stage must fulfill \c 0 <= stage < stages()-1
    const MultiMatrix<Matrix, Container>& interpolation( unsigned stage) const
    {
        return m_inter[stage];
    }
    ///@brief return the projection matrix at given stage
    ///@param stage must fulfill \c 0 <= stage < stages()-1
    const MultiMatrix<Matrix, Container>& projection( unsigned stage) const
    {
        return m_project[stage];
    }
    Container& x(unsigned stage){ return m_x[stage];}
    const Container& x(unsigned stage) const{ return m_x[stage];}
    Container& r(unsigned stage){ return m_r[stage];}
    const Container& r(unsigned stage) const{ return m_r[stage];}
    Container& b(unsigned stage){ return m_b[stage];}
    const Container& b(unsigned stage) const{ return m_b[stage];}
    Container& w(unsigned stage){ return m_w[stage];}
    const Container& w(unsigned stage) const{ return m_w[stage];}

    private:
    unsigned m_stages;
    std::vector< dg::ClonePtr< Geometry> > m_grids;
    std::vector< MultiMatrix<Matrix, Container> >  m_inter;
    std::vector< MultiMatrix<Matrix, Container> >  m_project;
    std::vector< Container> m_x, m_r, m_b, m_w;
};

/*!@brief Full approximation nested iterations
 *
 * Solve \f$ f(x_0^{h})  = b^{h}\f$ for given \f$ b^h\f$ and initial guess \f$ x_0^h\f$:
 * - Compute \f$ r_0^h = b^h - f(x_0^h)\f$
 * - Project \f$ r_0^{2h} = P r_0^h\f$ and \f$ x_0^{2h} = Px_0^h\f$
 * - Compute \f$ b^{2h} = f(x_0^{2h}) + r_0^{2h}\f$
 * - Cycle: \f$ f(x^{2h})  = b^{2h}\f$ with initial guess \f$ x_0^{2h}\f$:
 *      - (residuum \f$ r_0^{2h}\f$ remains unchanged)
 *      - \f$ r_0^{4h} = Pr_0^{2h}\f$ and \f$ x_0^{4h} = Px_0^{2h}\f$
 *      - Compute \f$ b^{4h} = f(x_0^{4h}) + r_0^{4h}\f$
 *      - Cycle (on coarsest grid solve): \f$ f(x^{4h}) = b^{4h}\f$ with initial guess \f$x_0^{4h}\f$:
 *          - ...
 *      - \f$ \delta^{4h} = x^{4h} - x_0^{4h}\f$
 *      - \f$ \tilde x_0^{2h} = x_0^{2h} + I \delta^{4h}\f$
 *      - Solve \f$ f(x^{2h}) = b^{2h}\f$ with initial guess \f$ \tilde x_0^{2h}\f$
 *      .
 * - \f$ \delta^{2h} = x^{2h} - x_0^{2h}\f$
 * - \f$ \tilde x_0^h = x_0^h + I \delta^{2h}\f$
 * - Solve \f$ f(x^h) = b^h\f$ with initial guess \f$ \tilde x_0^h\f$
 * .
 * This algorithm is equivalent to a multigrid V-cycle with zero down-grid smoothing
 * and infinite (i.e. solving) upgrid smoothing.
 * @param ops Operators \c f on the various grids, i.e. \c dg::apply( ops[0], x, b)
 *  computes b = f(x). Index 0 is on the original grid, 1 on the half
 *  grid, 2 on the quarter grid, ...
 * @param x (read/write) contains initial guess on input and the solution on
 *  output (if the initial guess is good enough the solve may return
 *  immediately)
 * @param b The right hand side
 * @param inverse_ops a vector of inverse operators \c f^{-1},
 *  i.e. \c dg::apply( inverse_ops[0], b, x) computes \f$ x = f^{-1}(b)\f$
 *  (usually lambda functions combining operators and solvers). On call \c x
 *  contains the initial guess and should contain the solution on return.
 * @param nested_grids provides projection and interapolation operations and workspace
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 */
template<class MatrixType0, class ContainerType0, class ContainerType1, class MatrixType1, class NestedGrids>
void nested_iterations(
    std::vector<MatrixType0>& ops, ContainerType0& x, const ContainerType1& b,
    std::vector<MatrixType1>& inverse_ops, NestedGrids& nested_grids)
{
    NestedGrids& nested = nested_grids;
    // compute residual r = b - A x
    dg::apply(ops[0], x, nested.r(0));
    dg::blas1::axpby(1., b, -1., nested.r(0));
    // project residual down to coarse grid
    dg::blas1::copy( x, nested.x(0));
    for( unsigned u=0; u<nested.stages()-1; u++)
    {
        dg::blas2::gemv( nested.projection(u), nested.r(u), nested.r(u+1));
        dg::blas2::gemv( nested.projection(u), nested.x(u), nested.x(u+1));
        // compute FAS right hand side
        dg::blas2::symv( ops[u+1], nested.x(u+1), nested.b(u+1));
        dg::blas1::axpby( 1., nested.b(u+1), 1., nested.r(u+1), nested.b(u+1));
        dg::blas1::copy( nested.x(u+1), nested.w(u+1)); // remember x0
    }

    //now solve residual equations
    for( unsigned u=nested.stages()-1; u>0; u--)
    {
        try{
            dg::apply( inverse_ops[u],  nested.b(u), nested.x(u));
        }catch( dg::Error& err){
            err.append_line( dg::Message(_ping_)<<"ERROR on stage "<<u<<" of nested iterations");
            throw;
        }
        // delta
        dg::blas1::axpby( 1., nested.x(u), -1., nested.w(u), nested.x(u) );
        // update x
        dg::blas2::symv( 1., nested.interpolation(u-1), nested.x(u), 1.,
                nested.x(u-1));
    }
    //update initial guess
    dg::blas1::copy( nested.x(0), x);
    try{
        dg::apply(inverse_ops[0], b, x);
    }catch( dg::Error& err){
        err.append_line( dg::Message(_ping_)<<"ERROR on stage 0 of nested iterations");
        throw;
    }
}

/*!@brief EXPERIMENTAL Full approximation multigrid cycle
 *
 * @sa https://www.osti.gov/servlets/purl/15002749
 *
 * Compute \f$ x_0^h \leftarrow C_{\nu_1}^{\nu_2}(x_0^h, b^h)\f$ for given \f$ b^h\f$ and initial guess \f$ x_0^h\f$: \n
 * Note that we need to save the initial guess \f$ w_0^h := x_0^h\f$
 * in a multigrid cycle because we need it to compute the error for the next upper level.
 * - Smooth \f$ f(x^h) = b^h\f$ with initial guess \f$x_0^{h}\f$, overwrite \f$ x_0^h\f$
 * - Compute \f$ r_0^h = b^h - f(x_0^h)\f$
 * - Project \f$ r_0^{2h} = P r_0^h\f$ and \f$ x_0^{2h} = Px_0^h\f$
 * - Compute \f$ b^{2h} = f(x_0^{2h}) + r_0^{2h}\f$
 * - \f$ w_0^{2h} = x_0^{2h}\f$
 * - recursively call itself \f$\gamma\f$ times or solve \f$ f(x^{2h})  = b^{2h}\f$ with initial guess \f$ x_0^{2h}\f$, overwrite \f$ x_0^{2h}\f$
 *      - ...
 * - \f$ \delta^{2h} = x_0^{2h} - w_0^{2h}\f$ (Here we need the saved initial guess \f$ w_0^{2h}\f$)
 * - \f$ x_0^h = x_0^h + I \delta^{2h}\f$
 * - Smooth \f$ f(x^h) = b^h\f$ with initial guess \f$ x_0^h\f$, overwrite \f$ x_0^h\f$
 * .
 * This algorithm forms the core of multigrid algorithms.
 * @param ops
     Index 0 is the Operator on the original grid, 1 on the half grid, 2 on the
     quarter grid, ...
 * @param inverse_ops_down a vector of inverse, smoothing operators (usually
 *  lambda functions combining operators and solvers) of size \c stages-1
 * @param inverse_ops_up a vector of inverse, smoothing operators (usually
 *  lambda functions combining operators and solvers) of size \c stages
 * @param nested_grids provides projection and interapolation operations and workspace
 * @param gamma The shape of the multigrid cycle:
    typically 1 (V-cycle) or 2 (W-cycle)
 * @param p The current stage \c h
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 */
template<class NestedGrids, class MatrixType0, class MatrixType1, class MatrixType2>
void multigrid_cycle(
    std::vector<MatrixType0>& ops,
    std::vector<MatrixType1>& inverse_ops_down, //stages -1
    std::vector<MatrixType2>& inverse_ops_up, //stages
    NestedGrids& nested_grids, unsigned gamma, unsigned p)
{
    NestedGrids& nested = nested_grids;
    // 1 multigrid cycle beginning on grid p
    // p < m_stages-1
    // x[p]    READ-write, initial guess on input, solution on output
    // x[p+1]  write only, solution on next stage on output
    // w[p]    untouched, copy of initial guess on current stage
    // w[p+1]  write only, contains delta solution on next stage on output
    // b[p]    READ only, right hand side on current stage
    // b[p+1]  write only new right hand side on next stage
    // r[p]    write only residuum at current stage
    // r[p+1]  write only, residuum on next stage

    // 1. Pre-Smooth times
    try{
        dg::apply( inverse_ops_down[p], nested.b(p), nested.x(p));
    }catch( dg::Error& err){
        err.append_line( dg::Message(_ping_)<<"ERROR on pre-smoothing stage "<<p<<" of multigrid cycle");
        throw;
    }
    // 2. Residuum
    dg::apply( ops[p], nested.x(p), nested.r(p));
    dg::blas1::axpby( 1., nested.b(p), -1., nested.r(p));
    // 3. Coarsen
    dg::blas2::symv( nested.projection(p), nested.r(p), nested.r(p+1));
    dg::blas2::symv( nested.projection(p), nested.x(p), nested.x(p+1));
    dg::blas2::symv( ops[p+1], nested.x(p+1), nested.b(p+1));
    dg::blas1::axpby( 1., nested.r(p+1), 1., nested.b(p+1));
    // 4. Solve or recursive call to get x[p+1] with initial guess 0
    dg::blas1::copy( nested.x(p+1), nested.w(p+1));
    if( p+1 == nested.stages()-1)
    {
        try{
            dg::apply( inverse_ops_up[p+1], nested.b(p+1), nested.x(p+1));
        }catch( dg::Error& err){
            err.append_line( dg::Message(_ping_)<<"ERROR on stage "<<p+1<<" of multigrid cycle");
            throw;
        }
    }
    else
    {
        //update x[p+1] gamma times
        for( unsigned u=0; u<gamma; u++)
        {
            multigrid_cycle( ops, inverse_ops_down, inverse_ops_up,
                nested, gamma, p+1);
        }
    }

    // 5. Correct
    dg::blas1::axpby( 1., nested.x(p+1), -1., nested.w(p+1));
    dg::blas2::symv( 1., nested.interpolation(p), nested.w(p+1), 1., nested.x(p));
    // 6. Post-Smooth nu2 times
    try{
        dg::apply(inverse_ops_up[p], nested.b(p), nested.x(p));
    }catch( dg::Error& err){
        err.append_line( dg::Message(_ping_)<<"ERROR on post-smoothing stage "<<p<<" of multigrid cycle");
        throw;
    }
}

/**
 * @brief EXPERIMENTAL One Full multigrid cycle
 *
 * @param ops Index 0 is the \c f on the original grid, 1 on the half
 *  grid, 2 on the quarter grid, ...
 * @param x (read/write) contains initial guess on input and the solution on output
 * @param b The right hand side
 * @param inverse_ops_down a vector of inverse, smoothing operators (usually
 *  lambda functions combining operators and solvers) of size \c stages-1
 * @param inverse_ops_up a vector of inverse, smoothing operators (usually
 *  lambda functions combining operators and solvers) of size \c stages
 * @param nested_grids provides projection and interapolation operations and workspace
 * @param gamma The shape of the multigrid cycle:
    typically 1 (V-cycle) or 2 (W-cycle)
 * @param mu The repetition of the multigrid cycle (1 is typically ok)
 * @attention This method is rather unreliable, it only converges if the
 * parameters are chosen correctly ( there need to be enough smooting steps
 * for instance, and a large jump  factor in the Elliptic class also seems
 * to help) and otherwise just iterates to infinity. This behaviour is probably related to the use of the Chebyshev solver as a smoother
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
*/
template<class MatrixType0, class MatrixType1, class MatrixType2, class NestedGrids, class ContainerType0, class ContainerType1>
void full_multigrid(
    std::vector<MatrixType0>& ops, ContainerType0& x, const ContainerType1& b,
    std::vector<MatrixType1>& inverse_ops_down, //stages -1
    std::vector<MatrixType2>& inverse_ops_up, //stages
    NestedGrids& nested_grids, unsigned gamma, unsigned mu)
{
    NestedGrids& nested = nested_grids;
    // Like nested iterations, just uses multigrid-cycles instead of solves
    // compute residual r = b - A x
    dg::apply(ops[0], x, nested.r(0));
    dg::blas1::axpby(1., b, -1., nested.r(0));
    // project residual down to coarse grid
    dg::blas1::copy( x, nested.x(0));
    for( unsigned u=0; u<nested.stages()-1; u++)
    {
        dg::blas2::gemv( nested.projection(u), nested.r(u), nested.r(u+1));
        dg::blas2::gemv( nested.projection(u), nested.x(u), nested.x(u+1));
        // compute FAS right hand side
        dg::blas2::symv( ops[u+1], nested.x(u+1), nested.b(u+1));
        dg::blas1::axpby( 1., nested.b(u+1), 1., nested.r(u+1), nested.b(u+1));
        dg::blas1::copy( nested.x(u+1), nested.w(u+1)); // remember x0
    }

    //begin on coarsest level and cycle through to highest
    unsigned s = nested.stages()-1;
    try{
        dg::apply( inverse_ops_up[s], nested.b(s), nested.x(s));
    }catch( dg::Error& err){
        err.append_line( dg::Message(_ping_)<<"ERROR on stage "<<s<<" of full multigrid");
        throw;
    }
    dg::blas1::axpby( 1., nested.x(s), -1., nested.w(s), nested.x(s) );
    dg::blas2::symv( 1., nested.interpolation(s-1), nested.x(s), 1.,
            nested.x(s-1));

    for( int p=nested.stages()-2; p>=1; p--)
    {
        for( unsigned u=0; u<mu; u++)
            multigrid_cycle( ops, inverse_ops_down, inverse_ops_up, nested, gamma, p);
        dg::blas1::axpby( 1., nested.x(p), -1., nested.w(p), nested.x(p) );
        dg::blas2::symv( 1., nested.interpolation(p-1), nested.x(p), 1.,
                nested.x(p-1));
    }
    dg::blas1::copy( b, nested.b(0));
    for( unsigned u=0; u<mu; u++)
        multigrid_cycle( ops, inverse_ops_down, inverse_ops_up, nested, gamma, 0);
    dg::blas1::copy( nested.x(0), x);
}

/**
 * @brief EXPERIMENTAL Full multigrid cycles
 *
 * - Compute residual with given initial guess.
 * - If error larger than tolerance, do a full multigrid cycle with Chebeyshev iterations as smoother
 * - repeat
 * @param ops Index 0 is the \c MatrixType on the original grid, 1 on the half grid, 2 on the quarter grid, ...
 * @param x (read/write) contains initial guess on input and the solution on output
 * @param b The right hand side
 * @param inverse_ops_down a vector of inverse, smoothing operators (usually
 *  lambda functions combining operators and solvers) of size \c stages-1
 * @param inverse_ops_up a vector of inverse, smoothing operators (usually
 *  lambda functions combining operators and solvers) of size \c stages
 * @param nested_grids provides projection and interapolation operations and workspace
 * @param weights Defines the error norm
 * @param eps relative and absolute error tolerance
 * @param gamma The shape of the multigrid cycle:
    typically 1 (V-cycle) or 2 (W-cycle)
 * @attention This method is rather unreliable, it only converges if the
 * parameters are chosen correctly ( there need to be enough smooting steps
 * for instance, and a large jump  factor in the Elliptic class also seems
 * to help) and otherwise just iterates to infinity. This behaviour is probably
 * related to the use of the Chebyshev solver as a smoother
*/
template<class NestedGrids, class MatrixType0, class MatrixType1, class MatrixType2,
    class ContainerType0, class ContainerType1, class ContainerType2>
void fmg_solve(
    std::vector<MatrixType0>& ops,
    ContainerType0& x, const ContainerType1& b,
    std::vector<MatrixType1>& inverse_ops_down, //stages -1
    std::vector<MatrixType2>& inverse_ops_up, //stages
    NestedGrids& nested_grids,
    const ContainerType2& weights, double eps, unsigned gamma)
{
    //FULL MULTIGRID
    //full approximation scheme
    double nrmb = sqrt( blas2::dot( weights, b));

    try{
        full_multigrid( ops, x, b, inverse_ops_down, inverse_ops_up, nested_grids, gamma, 1);
    }catch( dg::Error& err){
        err.append_line( dg::Message(_ping_)<<"ERROR in fmg_solve");
        throw;
    }

    dg::apply( ops[0], x, nested_grids.r(0));
    dg::blas1::axpby( 1., b, -1., nested_grids.r(0));
    double error = sqrt( blas2::dot(weights,nested_grids.r(0)) );

    while ( error >  eps*(nrmb + 1))
    {
        //MULTIGRID CYCLES
        //multigrid_cycle( ops, inverse_ops_down, inverse_ops_up, nested_grids, gamma, 0);
        //FMG cycles
        try{
            full_multigrid( ops, x, b, inverse_ops_down, inverse_ops_up, nested_grids, gamma, 1);
        }catch( dg::Error& err){
            err.append_line( dg::Message(_ping_)<<"ERROR in fmg_solve");
            throw;
        }

        blas2::symv( ops[0], x, nested_grids.r(0));
        dg::blas1::axpby( 1., b, -1., nested_grids.r(0));
        error = sqrt( blas2::dot(weights,nested_grids.r(0)) );
        //DG_RANK0 std::cout<< "# Relative Residual error is  "<<error/(nrmb+1)<<"\n";
    }
}



/**
* @brief Solve \f[ \hat O \phi = \rho \f] for self-adjoint \f$\hat O\f$
*
* using \c dg::nested_iterations with \c dg::PCG solvers for any operator
* \f$\hat O\f$ that is self-adjoint in appropriate weights \f$W\f$
* We refine the grids in the first two dimensions (2d / x and y)
* @note The \c dg::Elliptic and \c dg::Helmholtz classes are self-adjoint so
*  these are the intended target operators.
* @note The preconditioner and weights for the \c dg::PCG solver are taken from the
* \c precond() and \c weights() method in the \c MatrixType class
*
* @snippet elliptic2d_b.cpp multigrid
* @copydoc hide_geometry_matrix_container
* @sa \c Extrapolation to generate an initial guess
*/
template< class Geometry, class Matrix, class Container>
struct MultigridCG2d
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief Allocate nothing, Call \c construct method before usage
    MultigridCG2d() = default;
    /**
     * @brief Construct the grids and the interpolation/projection operators
     *
     * @param grid the original grid (Nx() and Ny() must be evenly divisable by pow(2, stages-1)
     * @param stages number of grids in total (The second grid contains half the points of the original grids,
     *   The third grid contains half of the second grid ...). Must be >= 1. A good number to start is 3.
     * @param ps parameters necessary for \c dg::construct to construct a \c Container from a \c dg::HVec
    */
    template<class ...ContainerParams>
    MultigridCG2d( const Geometry& grid, const unsigned stages,
            ContainerParams&& ... ps):
        m_nested( grid, stages, std::forward<ContainerParams>(ps)...),
        m_pcg(    stages), m_stages(stages)
    {
        for (unsigned u = 0; u < stages; u++)
            m_pcg[u].construct(m_nested.x(u), m_nested.grid(u).size());
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

    ///@copydoc dg::NestedGrids::project(const ContainerType0&,std::vector<ContainerType0>&)const
    template<class ContainerType0>
    void project( const ContainerType0& src, std::vector<ContainerType0>& out) const
    {
        m_nested.project( src, out);
    }

    ///@copydoc dg::NestedGrids::project(const ContainerType0&)const
    template<class ContainerType0>
    std::vector<ContainerType0> project( const ContainerType0& src) const
    {
        return m_nested.project( src);
    }
    ///@return number of stages (same as \c num_stages)
    unsigned stages()const{return m_nested.stages();}
    ///@return number of stages (same as \c stages)
    unsigned num_stages()const{return m_nested.num_stages();}

    ///@brief return the grid at given stage
    ///@param stage must fulfill \c 0 <= stage < stages()
    const Geometry& grid( unsigned stage) const {
        return m_nested.grid(stage);
    }


    ///The maximum number of iterations allowed at stage 0
    ///(if the solution method returns this number, failure is indicated)
    unsigned max_iter() const{return m_pcg[0].get_max();}
    /**
     * @brief Set the maximum number of iterations allowed at stage 0
     *
     * By default this number is the grid size. However, for large
     * simulations you may want to prevent the solver from iterating to that number
     * in case of failure.
     * @param new_max new maximum number of iterations allowed at stage 0
    */
    void set_max_iter(unsigned new_max){ m_pcg[0].set_max(new_max);}
    /**
     *@brief Set or unset performance timings during iterations
     *@param benchmark If true, additional output will be written to \c std::cout during solution
     *@param message An optional identifier that is printed together with the
     * benchmark (intended use is to distinguish different messages
     * in the output)
    */
    void set_benchmark( bool benchmark, std::string message = "Nested Iterations"){
        m_benchmark = benchmark;
        m_message = message;
    }

    ///@brief Return an object of same size as the object used for construction on the finest grid
    ///@return A copyable object; what it contains is undefined, its size is important
    const Container& copyable() const {return m_nested.copyable();}
    /**
     * @brief Nested iterations
     *
     * Equivalent to the following
     * -# Compute residual with given initial guess.
     * -# Project residual down to the coarsest grid.
     * -# Solve equation on the coarse grid.
     * -# interpolate solution up to next finer grid and repeat 3 and 4 until the original grid is reached.
     * @sa \c dg::nested_iterations
     * @note The weights and preconditioner for the \c dg::PCG solver is taken
     *  from the \c weights() and \c precond() method in the \c MatrixType class
     * @param ops Index 0 is the \c MatrixType on the original grid, 1 on the half grid, 2 on the quarter grid, ...
     *  \c ops[u].precond() and \c ops[u].weights() need to be callable!
     * @param x (read/write) contains initial guess on input and the solution on output (if the initial guess is good enough the solve may return immediately)
     * @param b The right hand side
     * @param eps the accuracy: iteration stops if \f$ ||b - Ax|| < \epsilon(
     * ||b|| + 1) \f$. If needed (and it is recommended to tune these values)
     * the accuracy can be set for each stage separately. Per default the same
     * accuracy is used at all stages but \f$ \epsilon_i = 0.5\epsilon_0\f$ for i > 0 may be a good value as well.
     * @return the number of iterations in each of the stages beginning with the finest grid
     * @note the convergence test on the coarse grids is only evaluated every
     * 10th iteration. This effectively saves one dot product per iteration.
     * The dot product is the main performance bottleneck on the coarse grids.
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
    */
    template<class MatrixType, class ContainerType0, class ContainerType1>
    std::vector<unsigned> solve( std::vector<MatrixType>& ops, ContainerType0&  x, const ContainerType1& b, value_type eps)
    {
        std::vector<value_type> v_eps( m_stages, eps);
        for( unsigned u=m_stages-1; u>0; u--)
            v_eps[u] = eps;
        return solve( ops, x, b, v_eps);
    }
    ///@copydoc solve()
    template<class MatrixType, class ContainerType0, class ContainerType1>
    std::vector<unsigned> solve( std::vector<MatrixType>& ops, ContainerType0&  x, const ContainerType1& b, std::vector<value_type> eps)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        std::vector<unsigned> number(m_stages);
        std::vector<std::function<void( const ContainerType1&, ContainerType0&)> >
            multi_inv_pol(m_stages);
        for(unsigned u=0; u<m_stages; u++)
        {
            multi_inv_pol[u] = [&, u, &pcg = m_pcg[u], &pol = ops[u]](
            const auto& y, auto& x)
            {
                dg::Timer t;
                t.tic();
                if ( u == 0)
                    number[u] = pcg.solve( pol, x, y, pol.precond(),
                            pol.weights(), eps[u], 1, 1);
                else
                    number[u] = pcg.solve( pol, x, y, pol.precond(),
                            pol.weights(), eps[u], 1, 10);
                t.toc();
                if( m_benchmark)
                    DG_RANK0 std::cout << "# `"<<m_message<<"` stage: " << u << ", iter: " << number[u] << ", took "<<t.diff()<<"s\n";
            };
        }
        nested_iterations( ops, x, b, multi_inv_pol, m_nested);

        return number;
    }

  private:
    dg::NestedGrids<Geometry, Matrix, Container> m_nested;
    std::vector< PCG<Container> > m_pcg;
    unsigned m_stages;
    bool m_benchmark = true;
    std::string m_message = "Nested Iterations";

};
///@}

}//namespace dg
