#pragma once
#include "pcg.h"

namespace dg{
///@cond
namespace detail{
template<class Implicit, class Solver>
struct Adaptor
{
    Adaptor( Implicit& im, Solver& solver) : m_im(im), m_solver(solver){}
    template<class ContainerType, class value_type>
    void operator()( value_type t, const ContainerType& x, ContainerType& y)
    {
        m_im( t,x,y);
    }
    template<class ContainerType, class value_type>
    void operator()( value_type alpha, value_type t, ContainerType& y, const ContainerType& yp)
    {
        m_solver.solve( alpha, m_im, t, y, yp);
    }
    private:
    Implicit& m_im;
    Solver& m_solver;
};
}
///@endcond

/*!@brief PCG Solver class for solving \f$ (y-\alpha\hat I(t,y)) = \rho\f$
 *
 * for given t, alpha and rho.
 * \f$ \hat I\f$ must be linear self-adjoint positive definite as it uses a conjugate
 * gradient solver to invert the equation.
 * @note This struct is a simple wrapper. It exists because self-adjoint
 * operators appear quite often in practice and is not actually the recommended
 * default way of writing a solver for the implicit time part. It is better to
 * start with the following code and adapt from there
 * @code{.cpp}
 *  auto solver = [&eps = eps, &im = im, pcg = dg::PCG<ContainerType>( y0, 1000)]
 *      ( value_type alpha, value_type time, ContainerType& y, const
 *          ContainerType& ys) mutable
 *  {
 *     auto wrapper = [a = alpha, t = time, &i = im]( const auto& x, auto& y){
 *         i( t, x, y);
 *         dg::blas1::axpby( 1., x, -a, y);
 *     };
 *     dg::blas1::copy( ys, y); // take rhs as initial guess
 *     Timer t;
 *     t.tic();
 *     unsigned number = pcg.solve( wrapper, y, ys, im.precond(), im.weights(), eps);
 *     t.toc();
 *     DG_RANK0 std::cout << "# of pcg iterations time solver: "
 *           <<number<<"/"<<pcg.get_max()<<" took "<<t.diff()<<"s\n";
 *  };
 * @endcode
 * @sa In general it is recommended to write your own solver using a wrapper
 * lambda like the above and one of the existing solvers like \c dg::PCG,
 * \c dg::LGMRES or \c dg::AndersonAcceleration
 *
 * @copydoc hide_ContainerType
 * @sa ImExMultistep ImplicitMultistep ARKStep DIRKStep
 * @ingroup invert
 */
template<class ContainerType>
struct DefaultSolver
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>;//!< value type of vectors
    ///No memory allocation
    DefaultSolver(){}
    /*!
    * it does not matter what values \c copyable contains, but its size is important;
    * the \c solve method can only be called with vectors of the same size)
    * @tparam Implicit The self-adjoint, positive definite
    * implicit part of the right hand side.
    * Has signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
    * The first argument is the time, the second is the input vector, which the
    * functor may \b not override, and the third is the output,
    * i.e. y' = I(t, y) translates to I(t, y, y').  The two ContainerType
    * arguments never alias each other in calls to the functor.
    * Also needs the \c WeightType weights() and
    * \c PreconditionerType precond() member functions.
    * @param im The implicit part of the differential equation.
    * Stored as a \c std::function.
    * @attention make sure that im lives throughout the lifetime of this
    * object, else we'll get a dangling reference
    * @param copyable forwarded to constructor of \c dg::PCG
    * @param max_iter maximum iteration number in cg, forwarded to constructor of \c dg::PCG
    * @param eps relative and absolute accuracy parameter, used in the solve method of \c dg::PCG
    */
    template<class Implicit>
    DefaultSolver( Implicit& im, const ContainerType& copyable,
            unsigned max_iter, value_type eps): m_max_iter(max_iter)
    {
        m_im = [&im = im]( value_type t, const ContainerType& y, ContainerType&
                yp) mutable
        {
            im( t, y, yp);
        };
        // We'll have to do some gymnastics to store precond and weights
        // We do so by wrapping pcg's solve method
        // This should not be an example of how to write custom Solvers!
        m_solve = [ &weights = im.weights(), &precond = im.precond(), pcg =
            dg::PCG<ContainerType>( copyable, max_iter), eps = eps ]
            ( const std::function<void( const ContainerType&,ContainerType&)>&
              wrapper, ContainerType& y, const ContainerType& ys) mutable
        {
            return pcg.solve( wrapper, y, ys, precond, weights, eps);
        };
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = DefaultSolver( std::forward<Params>( ps)...);
    }
    ///@brief Set or unset performance timings during iterations
    ///@param benchmark If true, additional output will be written to \c std::cout during solution
    void set_benchmark( bool benchmark){ m_benchmark = benchmark;}

    void operator()( value_type alpha, value_type time, ContainerType& y, const
            ContainerType& ys)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif//MPI
        auto wrapper = [a = alpha, t = time, &i = m_im]( const auto& x, auto& y){
            i( t, x, y);
            dg::blas1::axpby( 1., x, -a, y);
        };
        Timer ti;
        if(m_benchmark) ti.tic();
        dg::blas1::copy( ys, y); // take rhs as initial guess
        unsigned number = m_solve( wrapper, y, ys);
        if( m_benchmark)
        {
            ti.toc();
            DG_RANK0 std::cout << "# of pcg iterations time solver: "
                <<number<<"/"<<m_max_iter<<" took "<<ti.diff()<<"s\n";
        }
    }
    private:
    std::function<void( value_type, const ContainerType&, ContainerType&)>
        m_im;
    std::function< unsigned ( const std::function<void( const
                ContainerType&,ContainerType&)>&, ContainerType&,
            const ContainerType&)> m_solve;
    unsigned m_max_iter;
    bool m_benchmark = true;
};

}//namespace dg
