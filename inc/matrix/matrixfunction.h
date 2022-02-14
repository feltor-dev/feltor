#include <cmath>

#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
//#include <cusp/print.h>

#include <cusp/lapack/lapack.h>
#include "dg/algorithm.h"

#include "functors.h"
#include "lanczos.h"

namespace dg {
namespace mat {


/*!
 * @brief Shortcut for \f$x \approx f(A) b  \f$ solve via exploiting first a Krylov projection achieved by the M-CG method and a matrix function computation via Eigen-decomposition
 *
 * @ingroup matrixfunctionapproximation
 *
 * The approximation relies on Projection
 * \f$x = f(A) b  \approx  R f(\tilde T)^{-1} e_1\f$,
 * where \f$\tilde T\f$ and \f$R\f$ are the tridiagonal and orthogonal matrix
 * of the M-CG solve and \f$e_1\f$ is the normalized unit vector. The vector
 * \f$f(\tilde T) e_1\f$ is computed via the Eigen decomposition and similarity
 * transform of \f$ \tilde T\f$.
 * @note Since MCG and Lanczos are equivalent the result of this class is the
 *  same within round-off errors as a Lanczos solve with the "residual" error
 *  norm
 */
template<class Container>
class MCGFuncEigen
{
  public:
    using value_type = dg::get_value_type<Container>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HArray2d = cusp::array2d< value_type, cusp::host_memory>;
    using HArray1d = cusp::array1d< value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    MCGFuncEigen(){}
    /**
     * @brief Construct MCGFuncEigen
     *
     * @param copyable a copyable container
     * @param max_iterations Max iterations of Lanczos method
     */
    MCGFuncEigen( const Container& copyable, unsigned max_iterations)
    {
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_mcg.construct(copyable, max_iterations);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MCGFuncEigen( std::forward<Params>( ps)...);
    }
    /**
     * @brief Compute \f$x \approx f(A)*b \f$ via M-CG method and Eigen decomposition
     *
     * @param b input vector
     * @param x output vector
     * @param f the matrix function (e.g. dg::SQRT<double> or dg::EXP<double>)
     * @param A self-adjoint and semi-positive definit matrix
     * @param weights weights
     * @param eps relative accuracy of M-CG method
     * @param nrmb_correction the absolute error \c C in units of \c eps to be
     *  respected
     * @param res_fac residual factor for stopping criterium of M-CG method
     *
     * @return number of iterations of M-CG routine
     */
    template < class MatrixType, class ContainerType0, class ContainerType1 ,
             class ContainerType2, class UnaryOp>
    unsigned operator()(ContainerType0& x, UnaryOp f,
            MatrixType&& A, const ContainerType1& b,
            const ContainerType2& weights, value_type eps,
            value_type nrmb_correction, value_type res_fac )
    {
        if( sqrt(dg::blas2::dot(b, weights, b)) == 0)
        {
            dg::blas1::copy( b, x);
            return 0;
        }
        //Compute x (with initODE with gemres replacing cg invert)
        m_TH = m_mcg(std::forward<MatrixType>(A), b, weights, eps,
                nrmb_correction, res_fac);
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
        //Compute Eigendecomposition
        cusp::lapack::stev(m_alpha, m_beta, m_evals, m_evecs);
        //convert to COO matrix format
//         cusp::print(m_evals);
        cusp::convert(m_evecs, m_EH);
        cusp::transpose(m_EH, m_EHt);
        //Compute f(T) e1 = D E f(Lambda) E^t D^{-1} e1
        dg::blas1::pointwiseDivide(m_e1H, m_delta, m_e1H);
        dg::blas2::symv(m_EHt, m_e1H, m_yH);
        dg::blas1::transform(m_evals, m_e1H, [f] (double x){
            try{
                return f(x);
            }
            catch(boost::exception& e) //catch boost overflow error
            {
                return 0.;
            }
        });
        dg::blas1::pointwiseDot(m_e1H, m_yH, m_e1H);
        dg::blas2::symv(m_EH, m_e1H, m_yH);
        dg::blas1::pointwiseDot(m_yH, m_delta, m_yH);
        //Compute x = R f(T) e_1
        m_mcg.Ry(std::forward<MatrixType>(A), m_TH, m_yH, x, b);

        return iter;
    }
  private:
    HVec m_e1H, m_yH;
    HDiaMatrix m_TH;
    HCooMatrix m_EH, m_EHt;
    HArray2d m_evecs;
    HArray1d m_alpha, m_beta, m_delta, m_evals;
    dg::mat::MCG< Container > m_mcg;

};

} //namespace mat
} //namespace dg
