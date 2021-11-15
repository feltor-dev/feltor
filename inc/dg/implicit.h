#pragma once
#include "cg.h"
#include "andersonacc.h"

namespace dg{
///@cond
namespace detail{

//compute: y + alpha f(y,t)
template< class LinearOp, class ContainerType>
struct ImplicitWithWeights
{
    using value_type = get_value_type<ContainerType>;
    ImplicitWithWeights(){}
    ImplicitWithWeights( value_type alpha, value_type t, LinearOp& f):
        f_(f), alpha_(alpha), t_(t){}
    void construct( value_type alpha, value_type t, LinearOp& f){
        f_ = f; alpha_=alpha; t_=t;
    }
    void symv( const ContainerType& x, ContainerType& y)
    {
        if( alpha_ != 0)
            f_(t_,x,y);
        blas1::axpby( 1., x, alpha_, y, y);
        blas2::symv( f_.weights(), y, y);
    }
    void operator()( const ContainerType& x, ContainerType& y){ symv(x,y);}

    value_type& alpha( ){  return alpha_;}
    value_type alpha( ) const  {return alpha_;}
    value_type& time( ){  return t_;}
    value_type time( ) const  {return t_;}
  private:
    LinearOp& f_;
    value_type alpha_;
    value_type t_;
};
template< class LinearOp, class ContainerType>
struct ImplicitWithoutWeights
{
    using value_type = get_value_type<ContainerType>;
    ImplicitWithoutWeights(){}
    ImplicitWithoutWeights( value_type alpha, value_type t, LinearOp& f):
        f_(f), alpha_(alpha), t_(t){}
    void construct( value_type alpha, value_type t, LinearOp& f){
        f_ = f; alpha_=alpha; t_=t;
    }
    void symv( const ContainerType& x, ContainerType& y)
    {
        if( alpha_ != 0)
            f_(t_,x,y);
        blas1::axpby( 1., x, alpha_, y, y);
    }
    void operator()( const ContainerType& x, ContainerType& y){ symv(x,y);}
    value_type& alpha( ){  return alpha_;}
    value_type alpha( ) const  {return alpha_;}
    value_type& time( ){  return t_;}
    value_type time( ) const  {return t_;}
  private:
    LinearOp& f_;
    value_type alpha_;
    value_type t_;
};

}//namespace detail
template< class M, class V>
struct TensorTraits< detail::ImplicitWithWeights<M, V> >
{
    using value_type = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
template< class M, class V>
struct TensorTraits< detail::ImplicitWithoutWeights<M, V> >
{
    using value_type = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond

/*! @class hide_SolverType
 *
 * @tparam SolverType
    The task of this class is to solve the equation \f$ (y+\alpha\hat I(t,y)) = \rho\f$
    for the given implicit part I, parameter alpha, time t and
    right hand side rho. For example \c dg::DefaultSolver or \c dg::FixedPointSolver
    If you write your own class:
 * it must have a solve method of type:
    \c void \c solve( value_type alpha, Implicit im, value_type t, ContainerType& y, const ContainerType& rhs);
  The <tt> const ContainerType& copyable() const; </tt> member must return a container of the size that is later used in \c solve
  (it does not matter what values \c copyable contains, but its size is important;
  the \c solve method will be called with vectors of this size)
 */

/*!@brief Default Solver class for solving \f[ (y+\alpha\hat I(t,y)) = \rho\f]
 *
 * for given t, alpha and rho.
 * works only for linear positive definite operators as it uses a conjugate
 * gradient solver to invert the equation
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
    * @param copyable vector of the size that is later used in \c solve (
     it does not matter what values \c copyable contains, but its size is important;
     the \c solve method can only be called with vectors of the same size)
    * @param max_iter maimum iteration number in cg
    * @param eps accuracy parameter for cg
    */
    DefaultSolver( const ContainerType& copyable, unsigned max_iter, value_type eps):
        m_pcg(copyable, max_iter), m_rhs( copyable), m_eps(eps)
        {}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_rhs;}
    ///@brief Set or unset performance timings during iterations
    ///@param benchmark If true, additional output will be written to \c std::cout during solution
    void set_benchmark( bool benchmark){ m_benchmark = benchmark;}

    template< class Implicit>
    void solve( value_type alpha, Implicit& im, value_type t, ContainerType& y, const ContainerType& rhs)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif//MPI
        detail::ImplicitWithWeights<Implicit, ContainerType> implicit( alpha,
                t, im);
        blas2::symv( im.weights(), rhs, m_rhs);
        Timer ti;
        if(m_benchmark) ti.tic();
        unsigned number = m_pcg( implicit, y, m_rhs, im.precond(), im.inv_weights(), m_eps);
        if( m_benchmark)
        {
            ti.toc();
            DG_RANK0 std::cout << "# of pcg iterations time solver: "<<number<<"/"<<m_pcg.get_max()<<" took "<<ti.diff()<<"s\n";
        }
    }
    private:
    CG< ContainerType> m_pcg;
    ContainerType m_rhs;
    value_type m_eps;
    bool m_benchmark = true;
};

/*!@brief Fixed point iterator for solving \f[ (y+\alpha\hat I(t,y)) = \rho\f]
 *
 * for given t, alpha and rho.
 * The fixed point iteration is given by
 * \f[
 *  y_{k+1}  = \rho - \alpha\hat I(t,y_k)
 *  \f]
 * @copydoc hide_ContainerType
 * @sa ImExMultistep ImplicitMultistep ARKStep DIRKStep
 * @ingroup invert
 */
template<class ContainerType>
struct FixedPointSolver
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>;//!< value type of vectors
    ///No memory allocation
    FixedPointSolver(){}
    /*!
    * @param copyable vector of the size that is later used in \c solve (
     it does not matter what values \c copyable contains, but its size is important;
     the \c solve method can only be called with vectors of the same size)
    * @param max_iter maimum iteration number
    * @param eps accuracy parameter. Convergence is in the l2 norm
    */
    FixedPointSolver( const ContainerType& copyable, unsigned max_iter, value_type eps):
        m_current( copyable), m_eps(eps), m_max_iter(max_iter)
        {}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_current;}

    ///@brief Set or unset performance timings during iterations
    ///@param benchmark If true, additional output will be written to \c std::cout during solution
    void set_benchmark( bool benchmark){ m_benchmark = benchmark;}

    template< class Implicit>
    void solve( value_type alpha, Implicit& im, value_type t, ContainerType& y, const ContainerType& rhs)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif//MPI
        Timer ti;
        if( m_benchmark)
            ti.tic();
        unsigned number = 0;
        value_type error = 0;
        do
        {
            dg::blas1::copy( y, m_current);
            im( t, m_current, y);
            dg::blas1::axpby( 1., rhs, -alpha, y);
            dg::blas1::axpby( 1., y, -1., m_current); //the difference
            number++;
            error = sqrt( dg::blas1::dot( m_current, m_current));
        }while ( error > m_eps && number < m_max_iter);
        if( m_benchmark)
        {
            ti.toc();
            DG_RANK0 std::cout << "# of iterations Fixed Point time solver: "<<number<<"/"<<m_max_iter<<" took "<<ti.diff()<<"s\n";
        }
    }
    private:
    ContainerType m_current;
    value_type m_eps;
    unsigned m_max_iter;
    bool m_benchmark = true;
};

/*!@brief Fixed Point iterator with Anderson Acceleration for solving \f[ (y+\alpha\hat I(t,y)) = \rho\f]
 *
 * for given t, alpha and rho.
 * @copydoc hide_ContainerType
 * @sa AndersonAcceleration ImExMultistep ImplicitMultistep ARKStep DIRKStep
 * @ingroup invert
 */
template<class ContainerType>
struct AndersonSolver
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>;//!< value type of vectors
    ///No memory allocation
    AndersonSolver(){}
    /*!
     * @param copyable vector of the size that is later used in \c solve (
      it does not matter what values \c copyable contains, but its size is important;
      the \c solve method can only be called with vectors of the same size)
     * @param mMax The maximum number of vectors to include in the optimization
     * procedure. \c mMax+1 is the number of solutions involved in computing
     * the new solution.  Something between 3 and 10 are good values but higher
     * values mean more storage space that needs to be reserved.  If \c mMax==0
     * then the algorithm is equivalent to Fixed Point (or Richardson if the
     * damping parameter is used in the \c solve() method) iteration i.e. no
     * optimization and only 1 solution needed to compute a new solution.
     * @param eps accuracy parameter for (rtol=atol=eps)
     * @param max_iter Maxmimum number of iterations
     * @param damping Paramter to prevent too large jumps around the actual
     * solution. Hard to determine in general but values between 1e-2 and 1e-4
     * are good values to begin with. This is the parameter that appears in
     * Richardson iteration. It is beter to have it too small than too large
     * (where it can lead to divergence of the solver)
     * @param restart Number >= mMax that indicates after how many iterations
     * to restart the acceleration. Periodic restarts are important for this
     * method.  Per default it should be the same value as \c mMax but \c mMax+1
     * or higher could be valuable to consider (but usually are worse
     * than \c mMax). Lower values \c restart<mMax are equivalent to setting
     * \c mMax=restart.
     */
    AndersonSolver( const ContainerType& copyable, unsigned mMax, value_type
            eps, unsigned max_iter, value_type damping, unsigned restart):
        m_acc(copyable, mMax), m_eps(eps), m_damp(damping), m_max(max_iter),
        m_restart(restart)
        {}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_acc.copyable();}
    ///@brief Set or unset performance timings during iterations
    ///@param benchmark If true, additional output will be written to \c std::cout during solution
    void set_benchmark( bool benchmark){ m_benchmark = benchmark;}

    template< class Implicit>
    void solve( value_type alpha, Implicit& im, value_type t, ContainerType& y, const ContainerType& rhs)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif//MPI
        //dg::WhichType<Implicit> {};
        detail::ImplicitWithoutWeights<Implicit, ContainerType> implicit(
                alpha, t, im);
        Timer ti;
        if( m_benchmark) ti.tic();
        unsigned number = m_acc.solve( implicit, y, rhs, im.weights(), m_eps,
                m_eps, m_max, m_damp, m_restart, false);
        if( m_benchmark)
        {
            ti.toc();
            DG_RANK0 std::cout << "# of Anderson iterations time solver: "<<number<<"/"<<m_max<<" took "<<ti.diff()<<"s\n";
        }
    }
    private:
    AndersonAcceleration< ContainerType> m_acc;
    value_type m_eps, m_damp;
    unsigned m_max, m_restart;
    bool m_benchmark = true;
};
}//namespace dg
