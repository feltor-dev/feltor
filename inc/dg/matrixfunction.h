#include <cmath>
#include "blas.h"
#include "functors.h"

#include "lanczos.h"

#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

#include <cusp/lapack/lapack.h>
#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK

namespace dg
{
/*! 
 * @brief Shortcut for \f$b \approx f(A) x  \f$ solve via exploiting first a Krylov projection achieved by the M-lanczos method and a matrix function computation via eigen-decomposition
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$b \approx f(A) x \approx  ||x||_M V f(T) e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the Lanczos solve and \f$e_1\f$ is the normalized unit vector. The vector \f$f(T) e_1\f$ is computed via eigen decomposition
 */
template< class Container >
struct KrylovFuncEigenSolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HArray2d = cusp::array2d< value_type, cusp::host_memory>;
    using HArray1d = cusp::array1d< value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief empty object ( no memory allocation)
    KrylovFuncEigenSolve() {}
    /**
     * @brief Construct KrylovFuncEigenSolve
     *
     * @param copyable a copyable container
     * @param max_iterations Max iterations of Lanczos method
     */
    KrylovFuncEigenSolve( const Container& copyable, unsigned max_iterations)
    { 
        construct(copyable,  max_iterations);
    }
    void construct(const Container& copyable, unsigned max_iterations)
    {      
        m_max_iter = max_iterations;
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_lanczos.construct(copyable, max_iterations);
    }
    /**
     * @brief Compute \f$b \approx f(A) x \approx  ||x||_M V f(T) e_1\f$ via M-Lanczos and eigendecomposition
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx f(A) x  \approx  ||x||_M V f(T) e_1\f$
     * @param f the matrix function (e.g. dg::SQRT<double>)
     * @param A symmetric and semi-positive definit matrix (e.g. not normed Helmholtz operator)
     * @param Minv inverse weights
     * @param M weights
     * @param eps accuracy of M-Lanczos method
     * @param res_fac residual factor for stopping criterium of M-Lanczos method
     * 
     * @return number of iterations of M-Lanczos routine
     */    
    template < class MatrixType, class ContainerType0, class ContainerType1 , class SquareNorm1, class SquareNorm2, class UnaryOp>
    unsigned operator()(const ContainerType0& x, ContainerType1& b, UnaryOp f,  MatrixType& A, SquareNorm1& Minv, SquareNorm2& M, value_type eps, value_type res_fac )
    {
        value_type xnorm = sqrt(dg::blas2::dot(M, x)); 
        if( xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return 0;
        }
        
        m_TH = m_lanczos(A, x, b, Minv, M, eps, false, res_fac); 

        unsigned iter = m_lanczos.get_iter();
        //resize
        m_alpha.resize(iter);
        m_beta.resize(iter-1);
        m_evals.resize(iter);
        m_evecs.resize(iter,iter);
        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize(iter);
        //fill diagonal entries
        for(unsigned i = 0; i<iter; i++)
        {
            m_alpha[i] = m_TH.values(i,1);   
            if (i<iter-1) m_beta[i]  = m_TH.values(i,2);  
        }
        //Compute eigendecomposition
        cusp::lapack::stev(m_alpha, m_beta, m_evals, m_evecs);
        //convert to COO matrix format
        cusp::convert(m_evecs, m_EH);     
        cusp::transpose(m_EH, m_EHt); 
        //Compute f(T) e1 = E f(Lambda) E^t e1
        dg::blas2::symv(m_EHt, m_e1H, m_yH);
        dg::blas1::transform(m_evals, m_e1H, f); 
        dg::blas1::pointwiseDot(m_e1H, m_yH, m_e1H);
        dg::blas2::symv(m_EH, m_e1H, m_yH);
        //Compute |x|_M V f(T) e1
        m_lanczos.normMxVy(A, m_TH, Minv, M,  m_yH,  b, x, xnorm, iter);

        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
        return iter;
    }
  private:
    unsigned m_max_iter;
    HVec m_e1H, m_yH;
    HDiaMatrix m_TH; 
    HCooMatrix m_EH, m_EHt;
    HArray2d m_evecs;
    HArray1d m_alpha, m_beta, m_evals;
    dg::Lanczos< Container> m_lanczos;
};


/*! 
 * @brief Shortcut for \f$x \approx \sqrt{A}^{-1} b  \f$ solve via exploiting first a Krylov projection achieved by the M-CG method and and secondly a sqrt cauchy solve
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$x = f(A)^{-1} b  \approx  R f(T)^{-1} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the MCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$f(T)^{-1} e_1\f$ is computed via the eigen decomposition and similarity transform of T.
 */
template<class Container>
class KrylovFuncEigenInvert
{
  public:
    using value_type = dg::get_value_type<Container>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HArray2d = cusp::array2d< value_type, cusp::host_memory>;
    using HArray1d = cusp::array1d< value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovFuncEigenInvert(){}
    ///@copydoc construct()
    /**
     * @brief Construct KrylovFuncEigenInvert
     *
     * @param copyable a copyable container
     * @param max_iterations Max iterations of Lanczos method
     */
    KrylovFuncEigenInvert( const Container& copyable, unsigned max_iterations)
    { 
        construct(copyable,  max_iterations);
    }
    void construct( const Container& copyable, unsigned max_iterations)
    {      
        m_max_iter = max_iterations;
        m_b = copyable;
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_mcg.construct(copyable, max_iterations);
    }
    /**
     * @brief Solve the system \f$f(A)*x = b \f$ for x using M-CG method and eigen decomposition
     * 
     * @param x output vector
     * @param b input vector 
     * @param f the matrix function (e.g. dg::SQRT<double> or dg::EXP<double>)
     * @param A symmetric and semi-positive definit matrix (e.g. not normed Helmholtz operator)
     * @param Minv inverse weights
     * @param M weights
     * @param eps accuracy of M-CG method
     * @param res_fac residual factor for stopping criterium of M-CG method
     * 
     * @return number of iterations of M-CG routine      
     */
    template < class MatrixType, class ContainerType0, class ContainerType1 , class SquareNorm1, class SquareNorm2, class UnaryOp>
    unsigned operator()(ContainerType0& x, const ContainerType1& b, UnaryOp f,  MatrixType& A, SquareNorm1& Minv, SquareNorm2& M, value_type eps, value_type res_fac )
    {
        if( sqrt(dg::blas2::dot(M, b)) == 0)
        {
            dg::blas1::copy( b, x);
            return 0;
        }
        //multiply weights
        dg::blas2::symv(M, b, m_b);
        //Compute x (with initODE with gemres replacing cg invert)
        m_TH = m_mcg(A, x, m_b, Minv, M, eps, 1., false, res_fac); 
        unsigned iter = m_mcg.get_iter();
        //resize
        m_alpha.resize(iter);
        m_delta.resize(iter,1.);
        m_beta.resize(iter-1);
        m_evals.resize(iter);
        m_evecs.resize(iter,iter);
        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize(iter);
        //fill diagonal entries of similarity transformed T matrix (now symmetric)
        for(unsigned i = 0; i<iter; i++)
        {
            m_alpha[i] = m_TH.values(i,1);   
            if (i<iter-1) {
                if      (m_TH.values(i,2) > 0.) m_beta[i] =  sqrt(m_TH.values(i,2)*m_TH.values(i+1,0)); // sqrt(b_i * c_i)
                else if (m_TH.values(i,2) < 0.) m_beta[i] = -sqrt(m_TH.values(i,2)*m_TH.values(i+1,0)); //-sqrt(b_i * c_i)
                else m_beta[i] = 0.;
            }
            if (i>0) m_delta[i] = m_delta[i-1]*sqrt(m_TH.values(i,0)/m_TH.values(i-1,2));
        }
        //Compute eigendecomposition
        cusp::lapack::stev(m_alpha, m_beta, m_evals, m_evecs);
        //convert to COO matrix format
//         cusp::print(m_evals);
        cusp::convert(m_evecs, m_EH);     
        cusp::transpose(m_EH, m_EHt); 
        //Compute f(T)^{-1} e1 = D E f(Lambda)^{-1} E^t D^{-1} e1
        dg::blas1::pointwiseDivide(m_e1H, m_delta, m_e1H);
        dg::blas2::symv(m_EHt, m_e1H, m_yH);
        dg::blas1::transform(m_evals, m_e1H, [f] (double x){ 
            try{
                return 1./f(x);                
            } 
            catch(boost::exception& e) //catch boost overflow error
            {
                return 0.;
            }
        });  //f(Lambda)^{-1}
        dg::blas1::pointwiseDot(m_e1H, m_yH, m_e1H);
        dg::blas2::symv(m_EH, m_e1H, m_yH);
        dg::blas1::pointwiseDot(m_yH, m_delta, m_yH);
        
        m_mcg.Ry(A, m_TH, Minv, M, m_yH, x, m_b,  iter); // x =  R f(T)^{-1} e_1  

        m_mcg.set_iter(m_max_iter);        
        return iter;
    }
  private:
    unsigned m_max_iter;
    Container m_b;
    HVec m_e1H, m_yH;
    HDiaMatrix m_TH;
    HCooMatrix m_EH, m_EHt;
    HArray2d m_evecs;
    HArray1d m_alpha, m_beta, m_delta, m_evals;
    dg::MCG< Container > m_mcg;

};
} //namespace dg
