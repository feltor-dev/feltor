#pragma once
#include "dg/functors.h"
#include "fast_interpolation.h"

/**@file
* @brief Modal Filtering
*/

namespace dg
{

///@cond
namespace create
{

template<class UnaryOp, class real_type>
dg::Operator<real_type> modal_op( UnaryOp op, const DLT<real_type>& dlt )
{
    Operator<real_type> backward=dlt.backward();
    Operator<real_type> forward=dlt.forward();
    Operator<real_type> filter( dlt.n(), 0);
    for( unsigned i=0; i<dlt.n(); i++)
        filter(i,i) = op( i);
    filter = backward*filter*forward;
    return filter;
}

} //namespace create
///@endcond

/**
 * @brief Struct that applies a given modal filter to a vector
 *
 * \f[ y = V D V^{-1}\f]
 * where \f$ V\f$ is the Vandermonde matrix (the backward transformation matrix)
 * and \f$ D \f$ is a diagonal matrix with \f$ D_{ii} = \sigma(i)\f$
 * @sa A discussion of the effects of the modal filter on advection schemes can be found here https://mwiesenberger.github.io/advection
 * @note basically the result is that it is usually not advantageous to use a modal filter
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 * @ingroup misc
 */
template<class MatrixType, class ContainerType>
struct ModalFilter
{
    using real_type = get_value_type<ContainerType>;
    ModalFilter(){}
    /**
     * @brief Create arbitrary filter
     *
     * @tparam Topology Any grid
     * @tparam UnaryOp Model of Unary Function \c real_type \c sigma(unsigned) The input will be the modal number \c i where \f$ i=0,...,n-1\f$ and \c n is the number of polynomial coefficients in use. The output is the filter strength for the given mode number
     * @param sigma The filter to evaluate on the normalized modal coefficients
     * @param t The topology to apply the modal filter on
     * @param ps parameters that are forwarded to the creation of a ContainerType (e.g. when a std::vector is to be created it is the vector size)
     */
    template<class UnaryOp, class Topology, class ...Params>
    ModalFilter( UnaryOp sigma, const Topology& t, Params&& ...ps) :
        m_tmp( dg::construct<ContainerType>(dg::evaluate( dg::zero, t),
        std::forward<Params>(ps)...)), m_filter ( dg::create::fast_transform(
        create::modal_op(sigma, t.dltx()), create::modal_op(sigma, t.dlty()),
        t), std::forward<Params>(ps)...)
            { }

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
        *this = ModalFilter( std::forward<Params>( ps)...);
    }

    void operator()( ContainerType& y) {
        operator()( 1., y, 0., m_tmp);
        using std::swap;
        swap( y, m_tmp);
    }
    void operator()( const ContainerType& x, ContainerType& y) const{ operator()( 1., x,0,y);}
    void operator()(real_type alpha, const ContainerType& x, real_type beta, ContainerType& y) const
    {
        m_filter.symv( alpha, x, beta, y);
    }
    private:
    ContainerType m_tmp;
    MultiMatrix<MatrixType, ContainerType> m_filter;
};

}//namespace dg
