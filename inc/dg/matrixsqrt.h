#include <cmath>

#include "blas.h"
#include "functors.h"

#include "adaptive.h"
#include "lanczos.h"
#include "sqrt_cauchy.h"
#include "sqrt_ode.h"

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

///@brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve directly via sqrt ODE solve with adaptive ERK class as timestepper
template< class Geometry, class Matrix, class Container>
struct DirectSqrtODESolve
{
   public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    DirectSqrtODESolve() {}
    /**
     * @brief Construct DirectSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     */
    DirectSqrtODESolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, value_type epsTimerel, value_type epsTimeabs):   
        m_A(A),
        m_rhs(A, g, epsCG),
        m_epsTimerel(epsTimerel),
        m_epsTimeabs(epsTimeabs) 
    { }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x  \f] via sqrt ODE solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \f]
     * @return number of timesteps of sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        return dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., x, 1., b, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs);
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    Rhs<Geometry, Matrix, Container> m_rhs;  
    value_type m_epsTimerel, m_epsTimeabs;
};

/*! 
 * @brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[b \approx \sqrt{A} x \approx b \approx ||x||_M V \sqrt{T} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the Lanczos solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T} e_1\f] is computed via the sqrt ODE solve.
 */
template< class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container>
struct KrylovSqrtODESolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtODESolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     */
    KrylovSqrtODESolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned iter):   
        m_A(A),
        m_epsTimerel(epsTimerel),
        m_epsTimeabs(epsTimeabs),
        m_xnorm(0.),
        m_e1(iter, 0.),
        m_y(iter, 1.),
        m_rhs(m_T, m_e1, epsCG),
        m_lanczos(copyable, iter)
    { 
        m_e1[0]=1.;
    }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f] via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f]
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        //Lanczos solve first         
        m_xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        m_TVpair = m_lanczos(m_A, x, b, m_A.weights(), m_A.inv_weights()); 
        m_T = m_TVpair.first; 
        m_V = m_TVpair.second;   
        //update T
        m_rhs.set_T(m_T);
        
        unsigned counter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs); // y = T^(1/2) e_1
        dg::blas2::gemv(m_V, m_y, b);
        dg::blas1::scal(b, m_xnorm);             // b = ||x|| V T^(1/2) e_1     
        return counter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    value_type m_epsTimerel, m_epsTimeabs, m_xnorm;
    Container m_e1, m_y;
    RhsT<DiaMatrix, Container> m_rhs;  
    dg::Lanczos< Container > m_lanczos;
    DiaMatrix m_T; 
    CooMatrix m_V;
    std::pair<DiaMatrix, CooMatrix> m_TVpair; 
};

/*! 
 * @brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[b \approx \sqrt{A} x \approx b \approx ||x||_M V \sqrt{T} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the Lanczos solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T} e_1\f] is computed via the sqrt ODE solve.
 */
template< class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container>
struct KrylovSqrtCauchySolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtCauchySolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param iter Iterations of Lanczos method
     */
    KrylovSqrtCauchySolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable, value_type epsCG, unsigned iter):   
        m_A(A),
        m_xnorm(0.),
        m_e1(iter, 0.),
        m_y(iter, 1.),
        m_cauchysqrt(m_T, m_e1, epsCG),
        m_lanczos(copyable, iter)
    { 
        m_e1[0]=1.;
    }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f] via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f]
     * @param iterCauchy iterations of sum of cauchy integral
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    void operator()(const Container& x, Container& b, const unsigned& iterCauchy)
    {
        //Lanczos solve first         
        m_xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        m_TVpair = m_lanczos(m_A, x, b, m_A.weights(), m_A.inv_weights()); 
        m_T = m_TVpair.first; 
        m_V = m_TVpair.second;   
        //update T
        m_cauchysqrt.set_T(m_T);
        m_cauchysqrt(m_e1, m_y, 1. ,10., iterCauchy); //(minEV, maxEV) estimated to (1, 10) - seems to work
        
        dg::blas2::gemv(m_V, m_y, b);
        dg::blas1::scal(b, m_xnorm);             // b = ||x|| V T^(1/2) e_1     
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    value_type m_xnorm;
    Container m_e1, m_y;
    CauchySqrtIntT<DiaMatrix, Container> m_cauchysqrt;  
    dg::Lanczos< Container > m_lanczos;
    DiaMatrix m_T; 
    CooMatrix m_V;
    std::pair<DiaMatrix, CooMatrix> m_TVpair; 
};

    
    
/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix 
*/
template< class ContainerType>
class InvTridiag
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::host_memory>;
    ///@brief Allocate nothing, Call \c construct method before usage
    InvTridiag(){}
    //Constructor
    InvTridiag(const std::vector<value_type>& copyable) 
    {
        phi.assign(copyable.size()+1,0.);
        theta.assign(copyable.size()+1,0.);
        Tinv.resize(copyable.size(), copyable.size(),  copyable.size()* copyable.size());
        temp = 0.;
    }

    void resize(unsigned new_size) {
        phi.resize(new_size+1);
        theta.resize(new_size+1);
        Tinv.resize(new_size, new_size, new_size*new_size);
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     * 
     * @param a  "0" diagonal vector
     * @param b "+1" diagonal vector (starts with index 0 to (size of b)-1)
     * @param c "-1" diagonal vector (starts with index 0 to (size of c)-1)
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     */
    template<class ContainerType0>
    coo_type operator()(const ContainerType0& a, const ContainerType0& b,  const ContainerType0& c)
    {

        //Compute theta and phi
        unsigned is=0;
        for( unsigned i = 0; i<theta.size(); i++)
        {   
            is = (theta.size() - 1) - i;
            if (i==0) 
            {   
                theta[0] = 1.; 
                phi[is]  = 1.;
            }
            else if (i==1) 
            {
                theta[1] = a[0]; 
                phi[is]  = a[is];
            }
            else
            {
                theta[i] = a[i-1] * theta[i-1] - b[i-2] * c[i-2] * theta[i-2];
                phi[is]  = a[is]  * phi[is+1]  - b[is]  * c[is]  * phi[is+2];
            }
        }

        //Compute inverse tridiagonal matrix elements
        unsigned counter = 0;
        for( unsigned i=0; i<a.size(); i++)
        {   
            for( unsigned j=0; j<a.size(); j++)
            {   
                Tinv.row_indices[counter]    = j;
                Tinv.column_indices[counter] = i; 
                temp=1.;
                if (i<j) {
                    for (unsigned k=i; k<j; k++) temp*=b[k];
                    Tinv.values[counter] =temp*pow(-1,i+j) * theta[i] * phi[j+1]/theta[a.size()];
                    
                }
                else if (i==j)
                {
                    Tinv.values[counter] =theta[i] * phi[j+1]/theta[a.size()];
                }   
                else // if (i>j)
                {
                    for (unsigned k=j; k<i; k++) temp*=c[k];           
                    Tinv.values[counter] =temp*pow(-1,i+j) * theta[j] * phi[i+1]/theta[a.size()];

                }
                counter++;
            }
        }
        return Tinv;
    }
  private:
    std::vector<value_type> phi, theta;
    coo_type Tinv;    
    value_type temp;
};


/*! 
 * @brief Shortcut for \f[x \approx \sqrt{A}^{-1} b  \f] solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the PCG solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T^{-1}} e_1\f] is computed via the sqrt ODE solve.
 */
template< class ContainerType>
class CGsqrt
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::host_memory>;
    ///@brief Allocate nothing, Call \c construct method before usage
    CGsqrt(){}
    ///@copydoc construct()
    CGsqrt( const ContainerType& copyable, unsigned max_iterations) : 
        max_iter(max_iterations)
    {
          construct(copyable, max_iterations);
    }
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return max_iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return p;}

    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable, unsigned max_iterations)
    {
        v0.assign(max_iterations,0.);
        vp.assign(max_iterations,0.);
        vm.assign(max_iterations,0.);
        rh.assign(max_iterations, copyable);
        m_e1.assign(max_iterations,0.);
        m_y.assign(max_iterations,0.);
        ap = p = r =  copyable;
        max_iter = max_iterations;
        R.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
    }
    /**
     * @brief Solve the system \f[\sqrt{A}*x = b \f] for x using PCG method
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
      */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
    unsigned operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1)
    {
        //Do CG iteration do get R and T matrix
        value_type nrmb = sqrt( dg::blas2::dot( S, b));
    #ifdef DG_DEBUG
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif //MPI
        {
        std::cout << "# Norm of b "<<nrmb <<"\n";
        std::cout << "# Residual errors: \n";
        }
    #endif //DG_DEBUG


        if( nrmb == 0)
        {
            dg::blas1::copy( b, x);
            return 0;
        }
        dg::blas2::symv( A, x, r);
        dg::blas1::axpby( 1., b, -1., r);
        dg::blas2::symv( P, r, p );//<-- compute p_0
        dg::blas1::copy( p, rh[0]);

        //note that dot does automatically synchronize
        value_type nrm2r_old = dg::blas1::dot( rh[0],r); //and store the norm of it
        value_type nrm2r_new;
        
        for( unsigned i=0; i<max_iter; i++)
        {
            dg::blas2::symv( A, p, ap);
            alpha = nrm2r_old /dg::blas1::dot( p, ap);
            dg::blas1::axpby( -alpha, ap, 1., r);
    #ifdef DG_DEBUG
    #ifdef MPI_VERSION
            if(rank==0)
    #endif //MPI
            {
                std::cout << "# Absolute r*S*r "<<sqrt( dg::blas2::dot(S,r)) <<"\t ";
                std::cout << "#  < Critical "<<eps*nrmb + eps <<"\t ";
                std::cout << "# (Relative "<<sqrt( dg::blas2::dot(S,r) )/nrmb << ")\n";
            }
    #endif //DG_DEBUG
            if( sqrt( dg::blas2::dot( S, r)) < eps*(nrmb + nrmb_correction)) //TODO change this criterium  for square root matrix
            {
                dg::blas2::symv(P, r, rh[i+1]);
                nrm2r_new = dg::blas1::dot( rh[i+1], r);
                    
                vp[i] = -nrm2r_new/nrm2r_old/alpha;
                vm[i] = -1./alpha;
                v0[i+1] = -vp[i];
                v0[i] -= vm[i];
                
                max_iter=i+1;
                break;
            }
            dg::blas2::symv(P, r, rh[i+1]);
            nrm2r_new = dg::blas1::dot( rh[i+1], r);
            dg::blas1::axpby(1., rh[i+1], nrm2r_new/nrm2r_old, p );
                 
            vp[i] = -nrm2r_new/nrm2r_old/alpha;
            vm[i] = -1./alpha;
            v0[i+1] = -vp[i];
            v0[i] -= vm[i];
            nrm2r_old=nrm2r_new;
        }
        //Resize vectors and matrix first
        v0.resize(max_iter);
        vp.resize(max_iter);
        vm.resize(max_iter);
        rh.resize(max_iter);
        m_e1.resize(max_iter,0.);
        m_e1[0]=1.;
        m_y.resize(max_iter,0.);

        R.resize(rh[0].size(), max_iter, max_iter*rh[0].size());
        Tinv.resize(max_iter, max_iter, max_iter*max_iter);
        invtridiag.resize(max_iter);

        //Compute inverse of tridiagonal matrix
        Tinv = invtridiag(v0,vp,vm);
        // fill R matrix
        unsigned counter = 0;
        for( unsigned i=0; i<max_iter; i++)
        {           
            for( unsigned j=0; j<rh[0].size(); j++)
            {            
                R.row_indices[counter]    = j;
                R.column_indices[counter] = i; 
                R.values[counter]         = rh[i][j];
                counter++;
            }
        }     
        //Compute x (with initODE with gemres replacing cg invert)
        RhsTasym<coo_type, ContainerType> m_rhs(Tinv, m_e1, eps);        
        m_rhs.set_T(Tinv);

        //could be replaced by Cauchy sqrt solve
        unsigned time_iter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, 1e-8, 1e-10);
        std::cout << "Time iterations  " << time_iter  << "\n";
        dg::blas2::gemv(R, m_y, x);  // x =  R T^(-1/2) e_1   

        return max_iter;
    }
  private:
    value_type alpha;
    std::vector<value_type> v0, vp, vm;
    std::vector<ContainerType> rh;
    ContainerType r, ap, p, m_e1, m_y;
    unsigned max_iter;
    coo_type R, Tinv;      
    InvTridiag<ContainerType> invtridiag;
};
