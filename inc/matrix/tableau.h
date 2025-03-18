#pragma once

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include "dg/algorithm.h"
#include "functors.h"

namespace dg{
namespace mat{


/*! @brief Manage coefficients of a functional (extended) Butcher tableau
 *
 * The goal of this class is to represent a Butcher tableau for the use
 * in Exponential Runge Kutta type time integrators.
 * The coefficients of the tableau
 * are easily constructible and accessible.
 * Furthermore, we provide utilities like the number of stages, whether the
 * tableau is embedded or not and the order of the method.
 *
 * Currently available are
 * @copydoc hide_func_explicit_butcher_tableaus
 *
 * @tparam real_type type of the coefficients
 * @sa dg::mat::ExpRungeKutta
 * @ingroup exp_int
 */
template<class real_type>
struct FunctionalButcherTableau{
    using value_type = real_type;
    using function_type = std::function<value_type(value_type)>;
    ///No memory allocation
    FunctionalButcherTableau() = default;
    /*! @brief Construct a classic non-embedded tableau
     * @param s number of stages
     * @param order (global) order of the resulting method
     * @param a pointer to s*s real functions interpreted as a_{ij}=a[i*s+j]
     * @param b pointer to s real functions interpreted as b_j=b[j]
     * @param c pointer to s real numbers interpreted as c_i=c[i]
     * @note This constructor initializes the embedded coefficients bt=b
     * which makes the embedded method equal to the actual method
     */
    FunctionalButcherTableau(unsigned s, unsigned order,
                   const function_type* a , const function_type* b , const real_type* c):
        m_a(a, a+s*s), m_b(b, b+s), m_c(c, c+s), m_bt(b,b+s), m_q(order), m_p(order), m_s(s){}
    /*! @brief Construct an embedded tableau
     *
     * @param s number of stages
     * @param embedded_order (global) order of the embedded method (corresponding to \c bt)
     * @param order (global) order of the method (corresponding to \c b)
     * @param a pointer to s*s real functions interpreted as a_{ij}=a[i*s+j]
     * @param b pointer to s real functions interpreted as b_j=b[j]
     * @param bt pointer to s real functions interpreted as bt_j=bt[j]
     * @param c pointer to s real numbers interpreted as c_i=c[i]
     */
    FunctionalButcherTableau(unsigned s, unsigned embedded_order, unsigned order,
               const function_type* a, const function_type* b, const function_type* bt, const real_type* c):
        m_a(a, a+s*s), m_b(b,b+s), m_c(c,c+s), m_bt(bt, bt+s), m_q(order), m_p(embedded_order), m_s(s), m_embedded(true){}

    /**
    * @brief Read the a_ij coefficients
    * @param i row number 0<=i<s, i>=s results in undefined behaviour
    * @param j col number 0<=j<s, j>=s results in undefined behaviour
    * @return a_ij
    * @note The returned function is either strictly positive or zero everywhere, so a test <tt> a(i,j)(0) == 0 </tt> is valid
    */
    function_type a( unsigned i, unsigned j) const {
        return m_a(i,j);
    }
    /**
    * @brief Read the c_i coefficients
    * @param i row number 0<=i<s, i>=s results in undefined behaviour
    * @return c_i
    */
    real_type c( unsigned i) const {
        return m_c[i];
    }
    /**
    * @brief Read the b_j coefficients
    * @param j col number 0<=j<s, j>=s results in undefined behaviour
    * @return b_j
    * @note The returned function is either strictly positive or zero everywhere, so a test <tt> b(j)(0) == 0 </tt> is valid
    */
    function_type b( unsigned j) const {
        return m_b[j];
    }
    /**
    * @brief Read the embedded bt_j coefficients
    * @param j col number 0<=j<s, j>=s results in undefined behaviour
    * @return bt_j
    * @note The returned function is either strictly positive or zero everywhere, so a test <tt> bt(j)(0) == 0 </tt> is valid
    */
    function_type bt( unsigned j) const {
        return m_bt[j];
    }
    ///The number of stages s
    unsigned num_stages() const  {
        return m_s;
    }
    ///global order of accuracy for the method represented by b
    unsigned order() const {
        return m_q;
    }
    ///global order of accuracy for the embedded method represented by bt
    unsigned embedded_order() const{
        return m_p;
    }
    ///True if the method has an embedding
    bool isEmbedded()const{
        return m_embedded;
    }
    /// True if an element on or above the diagonal in a is non-zero
    bool isImplicit()const{
        for( unsigned i=0; i<m_s; i++)
            for( unsigned j=i; j<m_s; j++)
                if( a(i,j) != 0)
                    return true;
        return false;
    }
    // ///True if the method has the "First Same As Last" property:
    // ///the last stage is evaluated at the same point as the first stage of the next step.
    // bool isFsal()const{
    //     if( m_c[m_s-1] != 1)
    //         return false;
    //     for (unsigned j=0; j<m_s; j++)
    //         if( a(m_s-1,j) != b(j) )
    //             return false;
    //     return true;
    // }
    private:
    dg::Operator<function_type> m_a;
    std::vector<function_type> m_b;
    std::vector<real_type> m_c;
    std::vector<function_type> m_bt;
    unsigned m_q, m_p, m_s;
    bool m_embedded = false;
};

///@cond
namespace func_tableau{

///%%%%%%%%%%%%%%%%%%%%%%%%%%%Classic Butcher tables%%%%%%%%%%%%%%%%%%
//https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
template<class real_type>
FunctionalButcherTableau<real_type> explicit_euler_1_1( )
{
    auto zero = [&](real_type){return 0;};
    using function_type = std::function<real_type(real_type)>;
    function_type a[1] = {zero};
    function_type b[1] = {[&](real_type x){return dg::mat::phi1(x);}};
    real_type c[1] = {0.};
    return FunctionalButcherTableau<real_type>( 1,1, a,b,c);
}
template<class real_type>
FunctionalButcherTableau<real_type> midpoint_2_2()
{
    auto zero = [&](real_type){return 0;};
    using function_type = std::function<real_type(real_type)>;
    function_type a[4] = {  zero, zero,
        [&](real_type x){return 0.5*dg::mat::phi1(x/2.);},
                       zero};
    function_type b[2] = { zero,
        [&](real_type x){return dg::mat::phi1(x);}};
    real_type c[2] = {0, 0.5};
    return FunctionalButcherTableau<real_type>( 2,2, a,b,c);
}
template<class real_type>
FunctionalButcherTableau<real_type> classic_4_4()
{
    auto zero = [&](real_type){return 0;};
    using function_type = std::function<real_type(real_type)>;
    function_type a[16] = {
        zero,zero,zero,zero,
        [&](real_type x){return 0.5*dg::mat::phi1(x/2.);}, zero,zero,zero,
        zero, [&](real_type x){return 0.5*dg::mat::phi1(x/2.);}, zero,zero,
        [&](real_type x){return 0.5*dg::mat::phi1(x/2.)*(exp(x/2.)-1);},zero,[&](real_type x){return dg::mat::phi1(x/2.);},zero
    };
    function_type b[4] = {
    [&](real_type x){return dg::mat::phi1(x)-3.*dg::mat::phi2(x)+4.*dg::mat::phi3(x);},
    [&](real_type x){return 2.*dg::mat::phi2(x)-4.*dg::mat::phi3(x);},
    [&](real_type x){return 2.*dg::mat::phi2(x)-4.*dg::mat::phi3(x);},
    [&](real_type x){return -dg::mat::phi2(x)+4.*dg::mat::phi3(x);}
    };
    real_type c[4] = {0, 0.5, 0.5, 1.};
    return FunctionalButcherTableau<real_type>( 4,4, a,b,c);
}
template<class real_type>
FunctionalButcherTableau<real_type> hochbruck_3_3_4()
{
    auto zero = [&](real_type){return 0;};
    using function_type = std::function<real_type(real_type)>;
    function_type a[9] = {
        zero,zero,zero,
        [&](real_type x){return 0.5*dg::mat::phi1(x/2.);}, zero,zero,
        zero, [&](real_type x){return dg::mat::phi1(x);}, zero
    };
    function_type b[3] = {
    [&](real_type x){return dg::mat::phi1(x)-14.*dg::mat::phi3(x)+36.*dg::mat::phi4(x);},
    [&](real_type x){return 16.*dg::mat::phi3(x)-48.*dg::mat::phi4(x);},
    [&](real_type x){return -2.*dg::mat::phi3(x)+12.*dg::mat::phi4(x);},
    };
    function_type bt[3] = {
    [&](real_type x){return dg::mat::phi1(x)-14.*dg::mat::phi3(x);},
    [&](real_type x){return 16.*dg::mat::phi3(x);},
    [&](real_type x){return -2.*dg::mat::phi3(x);}
    };
    real_type c[3] = {0, 0.5, 1.};
    return FunctionalButcherTableau<real_type>( 3,3,4, a,b,bt,c);
}


}//namespace func_tableau
///@endcond

/**
* @brief Identifiers for Butcher Tableaus
*
* We follow the naming convention of the ARKode library https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html
* as **NAME-S-P-Q** or **NAME-S-Q**, where
*  - NAME is the author or name of the method
*  - S is the number of stages in the method
*  - P is the global order of the embedding
*  - Q is the global order of the method
*
*  @ingroup exp_int
*/
enum func_tableau_identifier{
    EXPLICIT_EULER_1_1, //!
    MIDPOINT_2_2, //!
    CLASSIC_4_4,//!
    HOCHBRUCK_3_3_4
};

///@cond
namespace create{

inline const std::unordered_map<std::string, enum func_tableau_identifier> str2id{
    //Explicit methods
    {"Euler", EXPLICIT_EULER_1_1},
    {"Midpoint-2-2", MIDPOINT_2_2},
    {"Runge-Kutta-4-4", CLASSIC_4_4},
    {"Hochbruck-3-3-4", HOCHBRUCK_3_3_4},
};
inline enum func_tableau_identifier str2func_tableau( std::string name)
{
    auto it = str2id.find(name);
    if( it == str2id.end())
        throw dg::Error(dg::Message(_ping_)<<"Tableau "<<name<<" not found!");
    return it->second;
}
inline std::string func_tableau2str( enum func_tableau_identifier id)
{
    for( auto name: str2id)
    {
        if( name.second == id)
            return name.first;
    }
    throw dg::Error(dg::Message(_ping_)<<"Tableau conversion failed!");
}

template<class real_type>
FunctionalButcherTableau<real_type> func_tableau( enum func_tableau_identifier id)
{
    switch(id){
        case EXPLICIT_EULER_1_1:
            return func_tableau::explicit_euler_1_1<real_type>();
        case MIDPOINT_2_2:
            return func_tableau::midpoint_2_2<real_type>();
        case CLASSIC_4_4:
            return func_tableau::classic_4_4<real_type>();
        case HOCHBRUCK_3_3_4:
            return func_tableau::hochbruck_3_3_4<real_type>();
    }
    return FunctionalButcherTableau<real_type>(); //avoid compiler warning
}


template<class real_type>
FunctionalButcherTableau<real_type> func_tableau( std::string name)
{
        return func_tableau<real_type>( str2func_tableau(name));
}

}//namespace create
///@endcond

/*! @class hide_func_explicit_butcher_tableaus
 *
 * We follow the naming convention of the ARKode library https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html (They also provide nice stability plots for their methods)
 * as **NAME-S-P-Q** or **NAME-S-Q**, where
 *  - NAME is the author or name of the method
 *  - S is the number of stages in the method
 *  - P is the global order of the embedding
 *  - Q is the global order of the method
 *
 *    Name  | Identifier | Description
 *   -------|------------| -----------
 *   Euler                  | dg::mat::EXPLICIT_EULER_1_1     | <a href="https://doi.org/10.1017/S0962492910000048" target="_blank">Hochbruck and Ostermann, Exponential Integrators, Acta Numerica (2010)</a>
 *   Midpoint-2-2           | dg::mat::MIDPOINT_2_2           | <a href="https://doi.org/10.1017/S0962492910000048" target="_blank">Hochbruck and Ostermann, Exponential Integrators, Acta Numerica (2010)</a>
 *   Runge-Kutta-4-4        | dg::mat::CLASSIC_4_4            | <a href="https://doi.org/10.1006/jcph.2002.6995" target="_blank">Cox and Matthews, J. Comput. Phys., 176 (2002)</a>
 *   Hochbruck-3-3-4        | dg::mat::HOCHBRUCK_3_3_4            | <a href="https://doi.org/10.1017/S0962492910000048" target="_blank">Hochbruck and Ostermann, Exponential Integrators, Acta Numerica (2010)</a> (The exprb43 method)
 *
 * @note In exponential Rosenbrock type schemes it is assumed that \f$ A\f$ (the matrix) is the
    Jacobian of the system. If it is not, then the order conditions are different
    and the order and embedded orders are not what is indicated in our names.
 */


/*! @brief Convert identifiers to their corresponding \c dg::mat::FunctionalButcherTableau
 *
 * This is a helper class to simplify the interfaces of our timestepper functions and classes.
 * The sole purpose is to implicitly convert either a FunctionalButcherTableau or one of
 * the following identifiers to an instance of a FunctionalButcherTableau.
 *
 * @copydoc hide_func_explicit_butcher_tableaus
 * @param real_type The type of the coefficients in the FunctionalButcherTableau
 * @ingroup exp_int
 */
template<class real_type>
struct ConvertsToFunctionalButcherTableau
{
    using value_type = real_type;
    ///Of course a FunctionalButcherTableau converts to a FunctionalButcherTableau
    ///Useful if you constructed your very own coefficients
    ConvertsToFunctionalButcherTableau( FunctionalButcherTableau<real_type> tableau): m_t(tableau){}

    /*! @brief Create FunctionalButcherTableau from \c dg::mat::func_tableau_identifier
    *
    * The use of this constructor might be a bit awkward because you'll have to write all caps.
    * @param id the identifier, for example \c dg::mat::RUNGE_KUTTA_4_4
    */
    ConvertsToFunctionalButcherTableau( enum tableau_identifier id):m_t( create::func_tableau<real_type>(id)){}
    /*! @brief Create FunctionalButcherTableau from its name (very useful)
    *
    *  @note In some of the links in the Description below you might want to use the search function of your browser to find the indicated method
    *
    * Explicit methods
    * @copydoc hide_func_explicit_butcher_tableaus
    * @param name The name of the tableau as stated in the Name column above, as a string, for example "Euler"
    */
    ConvertsToFunctionalButcherTableau( std::string name):m_t( create::func_tableau<real_type>(name)){}
    ///@copydoc ConvertsToFunctionalButcherTableau(std::string)
    ConvertsToFunctionalButcherTableau( const char* name):m_t( create::func_tableau<real_type>(std::string(name))){}
    ///Convert to FunctionalButcherTableau
    ///
    ///which means an object can be directly assigned to a FunctionalButcherTableau
    operator FunctionalButcherTableau<real_type>( )const{
        return m_t;
    }
    private:
    FunctionalButcherTableau<real_type> m_t;
};

}//namespace mat
}//namespace dg
