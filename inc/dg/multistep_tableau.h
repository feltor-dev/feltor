#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace dg{
/**
 * @brief Manage coefficients of Multistep methods
 *
 * The general s-step multistep method has the form
 * \f[ y^{n+1} = \sum_{i=0}^{s-1} a_i y^{n-i} + h \sum_{i=0}^{s-1} b_i E( t_{n-i}, y_{n-i}) + h \sum_{i=0}^s c_i I( t_{n+1-i}, y^{n+1-i})
 * \f]
 * where E is the explicit and I is implicit part. A purely implicit method is
 * one where all \f$ b_i\f$ are zero, while an explicit one is one where all
 * \f$ c_i\f$ are zero.
 * A tableau thus consists of the three arrays a, b and c the number of steps
 * and the order of the method.
 * Currently available methods are:
 *
 * ImEx methods
 * @copydoc hide_imex_multistep_tableaus
 * @note ImEx multistep tableaus can be used in ExplicitMultistep, ImplicitMultistep and ImExMultistep
 *
 * Explicit methods
 * @copydoc hide_explicit_multistep_tableaus
 *
 * @tparam real_type type of the coefficients
 * @sa ExplicitMultistep, ImplicitMultistep, ImExMultistep
 * @ingroup time_utils
 */
template<class real_type>
struct MultistepTableau
{
    using value_type = real_type;
    ///No memory allocation
    MultistepTableau(){}
    /*! @brief Construct a tableau
     *
     * @param steps number of stages
     * @param order (global) order of the resulting method
     * @param a_v s real numbers
     * @param b_v s real numbers (can be empty, which then sets them all to 0 constructing an implicit method)
     * @param c_v s+1 real numbers (can be empty, which constructs an explicit method by assigning all c_i 0)
     */
    MultistepTableau( unsigned steps, unsigned order, const
            std::vector<real_type>& a_v, const std::vector<real_type>& b_v,
            const std::vector<real_type>& c_v): m_steps(steps), m_order(order),
            m_a(a_v), m_b(b_v), m_c(c_v){
                if( m_c.empty())
                    m_c.assign( steps+1, 0);
                if( m_b.empty())
                    m_b.assign( steps, 0);
    }

    /**
    * @brief Read the a_i coefficients
    * @param i idx number 0<=i<s, i>=s results in undefined behaviour
    * @return a_i
    */
    real_type a( unsigned i){ return m_a[i];}
    /**
    * @brief Read the explicit (b_i) coefficients
    * @param i idx number 0<=i<s, i>=s results in undefined behaviour
    * @return b_i
    */
    real_type ex( unsigned i){ return m_b[i];}
    /**
    * @brief Read the implicit (c_i) coefficients
    * @param i idx number 0<=i<s+1, i>=s+1 results in undefined behaviour
    * @return c_i
    */
    real_type im( unsigned i){ return m_c[i];}
    ///The number of stages s
    unsigned steps() const  {
        return m_steps;
    }
    ///global order of accuracy for the method
    unsigned order() const {
        return m_order;
    }
    ///True if any of the explicit coefficients b_i are non-zero
    bool isExplicit() const{
        for( unsigned i=0; i<m_steps; i++)
            if( m_b[i]!=0)
                return true;
        return false;
    }
    ///True if any of the implicit coefficients c_i are non-zero
    bool isImplicit() const{
        for( unsigned i=0; i<m_steps+1; i++)
            if( m_c[i]!=0)
                return true;
        return false;
    }
    private:
    unsigned m_steps, m_order;
    std::vector<real_type> m_a, m_b, m_c;
};

///@cond
namespace tableau
{
template<class real_type>
MultistepTableau<real_type> imex_euler_1_1()
{
    unsigned steps = 1, order = 1;
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    a[0] = b[0] = c[0] = 1;
    return MultistepTableau<real_type>( steps, order, a, b, c);
}
template<class real_type>
MultistepTableau<real_type> imex_adams_2_2()
{
    // 2nd order AB method extended to  ImEx
    // C = 4/9 ~ 0.444  D = 0.33
    unsigned steps = 2, order = 2;
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    a[0] = 1.;
    b[0] =  3./2.;
    b[1] = -1./2.;
    c[0] = 9./16.;
    c[1] = 3./8.;
    c[2] = 1./16.;
    return MultistepTableau<real_type>( steps, order, a, b, c);
}
//For some reason this tableau does not pass the convergence test
//template<class real_type>
//MultistepTableau<real_type> imex_adams_3_3()
//{
//    // 3rd order AB method extended to ImEx
//    // C ~ 0.16, D = 0.67
//    unsigned steps = 3, order = 3;
//    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
//    a[0] = 1.;
//    b[0] =  23./12.;
//    b[1] = -4./3.;
//    b[2] = 5./12.;
//    c[0] =   4661./10000.;
//    c[1] =  15551./30000.;
//    c[2] =     1949/30000;
//    c[3] = -1483./30000.;
//    return MultistepTableau<real_type>( steps, order, a, b, c);
//}
template<class real_type>
MultistepTableau<real_type> imex_koto_2_2()
{
    // stabilized 2nd order method
    unsigned steps = 2, order = 2;
    std::vector<real_type> am(steps,0), bm(steps, 0), cm(steps+1,0);
    std::vector<real_type> ap(steps+1,0), bp(steps+1, 0), cp(steps+1,0);
    real_type a = 1.5, b = 1.5;
    //real_type a = 20., b = 20.;
    ap[0] = a;
    ap[1] = 1-2.*a;
    ap[2] = a-1;
    cp[0] =  b;
    cp[1] = 0.5+a-2*b;
    cp[2] = 0.5-a+b;
    bp[1] = 0.5+a;
    bp[2] = 0.5-a;
    am[0] = -ap[1]/a, am[1] = -ap[2]/a;
    bm[0] = bp[1]/a, bm[1] = bp[2]/a;
    cm[0] = cp[0]/a, cm[1] = cp[1]/a, cm[2] = cp[2]/a;
    return MultistepTableau<real_type>( steps, order, am, bm, cm);
}

template<class real_type>
MultistepTableau<real_type> imex_bdf(unsigned steps)
{
    unsigned order = steps;
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    switch( steps)
    {
        case( 2):
        // C = 5/8 ~ 0.625  D = 0
        a[0] =  4./3.;  b[0] = 4./3.;
        a[1] = -1./3.;  b[1] = -2./3.;
        c[0] = 2./3.;
        break;
        case(3):
        //The Karniadakis method
        // C = 7/18 ~ 0.39  D = 0
        a[0] =  18./11.;    b[0] =  18./11.;
        a[1] = -9./11.;     b[1] = -18./11.;
        a[2] = 2./11.;      b[2] = 6./11.;
        c[0] = 6./11.;
        break;
        case(4):
        // C = 7/32 ~ 0.22 , D = 0
        a[0] =  48./25.;    b[0] =  48./25.;
        a[1] = -36./25.;    b[1] = -72./25.;
        a[2] =  16./25.;    b[2] =  48./25.;
        a[3] = - 3./25.;    b[3] = -12./25.;
        c[0] = 12./25.;
        break;
        case(5):
        // C = 0.0867 , D = 0
        a[0] = 300./137.;    b[0] = 300./137.;
        a[1] = -300./137.;   b[1] = -600./137.;
        a[2] = 200./137.;    b[2] = 600./137.;
        a[3] = -75./137.;    b[3] = -300./137.;
        a[4] = 12./137.;     b[4] = 60./137.;
        c[0] = 60./137.;
        break;
        case (6):
        a = {360./147.,-450./147.,400./147.,-225./147.,72./147.,-10./147.};
        b = {360./147.,-900./147.,1200./147.,-900./147.,360./147.,-60./147.};
        c[0] = 60./147.;
        break;
    }
    return MultistepTableau<real_type>( steps, order, a, b, c);
}

template<class real_type>
MultistepTableau<real_type> imex_tvb(unsigned steps)
{
    unsigned order = steps;
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    switch( steps)
    {
        case(3):
        // C = 0.536 D = 0.639
        a[0] =  3909./2048.;     b[0] =  18463./12288.;
        a[1] = -1367./1024.;     b[1] = -1271./768.;
        a[2] =  873./2048.;      b[2] = 8233./12288.;
        c[0] =  1089./2048.;
        c[1] = -1139./12288.;
        c[2] = -367./6144.;
        c[3] =  1699./12288.;
        break;
        case(4):
        // C = 0.458 , D = 0.685
        a[0] =  21531./8192.;     b[0] =  13261./8192.;
        a[1] = -22753./8192.;     b[1] = -75029./24576.;
        a[2] =  12245./8192.;     b[2] =  54799./24576.;
        a[3] = -2831./8192. ;     b[3] = -15245./24576.;
        c[0] =  4207./8192.;
        c[1] = -3567./8192.;
        c[2] =  697./24576.;
        c[3] = 4315./24576.;
        c[4] = -41./384.;
        break;
        case(5):
        // C = 0.376 , D = 0.709
        a[0] =  13553./4096.;     b[0] = 10306951./5898240.;
        a[1] = -38121./8192.;     b[1] = -13656497./2949120.;
        a[2] =  7315./2048.;      b[2] = 1249949./245760.;
        a[3] = -6161/4096. ;      b[3] = -7937687./2949120.;
        a[4] = 2269./8192.;       b[4] = 3387361./5898240.;
        c[0] =  4007./8192.;
        c[1] =  -4118249./5898240.;
        c[2] =  768703./2949120.;
        c[3] = 47849./245760.;
        c[4] = -725087./2949120.;
        c[5] = 502321./5898240.;
        break;
    }
    return MultistepTableau<real_type>( steps, order, a, b, c);
}

template<class real_type>
MultistepTableau<real_type> ab(unsigned order)
{
    unsigned steps = order;
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    a[0]= 1.;
    switch (order){
        case 1: b = {1}; break;
        case 2: b = {1.5, -0.5}; break;
        case 3: b = { 23./12., -4./3., 5./12.}; break;
        case 4: b = {55./24., -59./24., 37./24., -3./8.}; break;
        case 5: b = { 1901./720., -1387./360., 109./30., -637./360., 251./720.}; break;
        default: throw dg::Error(dg::Message()<<"Order "<<order<<" not implemented in AdamsBashforth!");
    }
    return MultistepTableau<real_type>( steps, order, a, b, c);
}

template<class real_type>
MultistepTableau<real_type> tvb(unsigned steps)
{
    unsigned order = steps;
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    switch (steps){
        case 1:
            a = {1.};
                b = {1.}; break;
        case 2:
            a = {4./3., -1./3.};
            b = {4./3., -2./3.}; break; //CLM = 0.5
        case 3: //CLM = 0.54...
            a[0] =  1.908535476882378;  b[0] =  1.502575553858997;
            a[1] = -1.334951446162515;  b[1] = -1.654746338401493;
            a[2] = 0.426415969280137;   b[2] = 0.670051276940255;
            break;
        case 4: //CLM = 0.45...
            a[0] = 2.628241000683208;   b[0] = 1.618795874276609;
            a[1] = -2.777506277494861;  b[1] = -3.052866947601049;
            a[2] = 1.494730011212510;   b[2] = 2.229909318681302;
            a[3] = -0.345464734400857;  b[3] = -0.620278703629274;
            break;
        case 5: //CLM = 0.37...
            a[0] = 3.308891758551210;   b[0] = 1.747442076919292;
            a[1] = -4.653490937946655;  b[1] = -4.630745565661800;
            a[2] = 3.571762873789854;   b[2] = 5.086056171401077;
            a[3] = -1.504199914126327;  b[3] = -2.691494591660196;
            a[4] = 0.277036219731918;   b[4] = 0.574321855183372;
            break;
        case 6: //CLM = 0.32...
            a[0] = 4.113382628475685;   b[0] = 1.825457674048542;
            a[1] = -7.345730559324184;  b[1] = -6.414174588309508;
            a[2] = 7.393648314992094;   b[2] = 9.591671249204753;
            a[3] = -4.455158576186636;  b[3] = -7.583521888026967;
            a[4] = 1.523638279938299;   b[4] = 3.147082225022105;
            a[5] = -0.229780087895259;  b[5] = -0.544771649561925;
            break;
        default: throw dg::Error(dg::Message()<<"Order "<<steps<<" not implemented in TVB scheme!");
    }
    return MultistepTableau<real_type>( steps, order, a, b, c);
}
template<class real_type>
MultistepTableau<real_type> ssp(unsigned steps)
{
    std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
    unsigned order = 0;
    switch (steps){
        case 1: order = 1;
                a = {1.};
                b = {1.}; break;
        case 2: order = 2;
                a = {4./5., 1./5.};
                b = {8./5., -2./5.}; break; //CLM = 0.5 ... ,order 2
        case 3: order = 2;
                a = { 3./4., 0., 1./4.};
                b = { 3./2., 0., 0. }; break; //CLM = 0.5..., order 2
        case 4: order = 2;
                a = {8./9., 0., 0., 1./9.};
                b = {4./3., 0., 0., 0.}; break; //CLM = 0.66..., order 2
        case 5: order = 3;
                a = {25./32., 0., 0., 0., 7./32.};
                b = {25./16.,0.,0.,0.,5./16.}; break; //CLM 0.5, order 3
        case 6: order = 3;
                a = {108./125.,0.,0.,0.,0.,17./125.};
                b = {36./25.,0.,0.,0.,0.,6./25.}; break; //CLM 0.567, order 3
        default: throw dg::Error(dg::Message()<<"Stage "<<steps<<" not implemented in SSP scheme!");
    }
    return MultistepTableau<real_type>( steps, order, a, b, c);
}

}//namespace tableau
///@endcond
/**
* @brief Identifiers for Multistep Tableaus
*
* We follow the naming convention
* as **NAME-S-Q**
*  - NAME is the author or name of the method
*  - S is the number of steps in the method
*  - Q is the global order of the method
*
*  @ingroup time_utils
*  @sa ExplicitMultistep, ImplicitMultistep, ImExMultistep
*/
enum multistep_identifier{
    //IMEX methods
    IMEX_EULER_1_1,
    IMEX_ADAMS_2_2,
    IMEX_KOTO_2_2,
    IMEX_BDF_2_2,
    IMEX_BDF_3_3,
    IMEX_BDF_4_4,
    IMEX_BDF_5_5,
    IMEX_BDF_6_6,
    IMEX_TVB_3_3,
    IMEX_TVB_4_4,
    IMEX_TVB_5_5,
    // Explicit methods
    AB_1_1,
    AB_2_2,
    AB_3_3,
    AB_4_4,
    AB_5_5,
    eBDF_1_1,
    eBDF_2_2,
    eBDF_3_3,
    eBDF_4_4,
    eBDF_5_5,
    eBDF_6_6,
    TVB_1_1,
    TVB_2_2,
    TVB_3_3,
    TVB_4_4,
    TVB_5_5,
    TVB_6_6,
    SSP_1_1,
    SSP_2_2,
    SSP_3_2,
    SSP_4_2,
    SSP_5_3,
    SSP_6_3,
    // implicit methods
    BDF_1_1,
    BDF_2_2,
    BDF_3_3,
    BDF_4_4,
    BDF_5_5,
    BDF_6_6,
};

///@cond
namespace create{

inline const std::unordered_map<std::string, enum multistep_identifier> str2lmsid{
    //Implicit-Explicit methods
    {"Euler", IMEX_EULER_1_1},
    {"Euler-1-1", IMEX_EULER_1_1},
    {"ImEx-Adams-2-2", IMEX_ADAMS_2_2},
    {"ImEx-Koto-2-2", IMEX_KOTO_2_2},
    {"ImEx-BDF-2-2", IMEX_BDF_2_2},
    {"ImEx-BDF-3-3", IMEX_BDF_3_3},
    {"Karniadakis",  IMEX_BDF_3_3},
    {"ImEx-BDF-4-4", IMEX_BDF_4_4},
    {"ImEx-BDF-5-5", IMEX_BDF_5_5},
    {"ImEx-BDF-6-6", IMEX_BDF_6_6},
    {"ImEx-TVB-3-3", IMEX_TVB_3_3},
    {"ImEx-TVB-4-4", IMEX_TVB_4_4},
    {"ImEx-TVB-5-5", IMEX_TVB_5_5},
    //Explicit methods
    {"AB-1-1", AB_1_1},
    {"AB-2-2", AB_2_2},
    {"AB-3-3", AB_3_3},
    {"AB-4-4", AB_4_4},
    {"AB-5-5", AB_5_5},
    {"eBDF-1-1", eBDF_1_1},
    {"eBDF-2-2", eBDF_2_2},
    {"eBDF-3-3", eBDF_3_3},
    {"eBDF-4-4", eBDF_4_4},
    {"eBDF-5-5", eBDF_5_5},
    {"eBDF-6-6", eBDF_6_6},
    {"TVB-1-1", TVB_1_1},
    {"TVB-2-2", TVB_2_2},
    {"TVB-3-3", TVB_3_3},
    {"TVB-4-4", TVB_4_4},
    {"TVB-5-5", TVB_5_5},
    {"TVB-6-6", TVB_6_6},
    {"SSP-1-1", SSP_1_1},
    {"SSP-2-2", SSP_2_2},
    {"SSP-3-2", SSP_3_2},
    {"SSP-4-2", SSP_4_2},
    {"SSP-5-3", SSP_5_3},
    {"SSP-6-3", SSP_6_3},
    // implicit methods
    {"BDF-1-1", BDF_1_1},
    {"BDF-2-2", BDF_2_2},
    {"BDF-3-3", BDF_3_3},
    {"BDF-4-4", BDF_4_4},
    {"BDF-5-5", BDF_5_5},
    {"BDF-6-6", BDF_6_6},
};
inline enum multistep_identifier str2lmstableau( std::string name)
{
    auto it = str2lmsid.find(name);
    if( it == str2lmsid.end())
        throw dg::Error(dg::Message(_ping_)<<"Multistep coefficients for "<<name<<" not found!");
    return it->second;
}
inline std::string lmstableau2str( enum multistep_identifier id)
{
    for( auto name: str2lmsid)
    {
        if( name.second == id)
            return name.first;
    }
    throw dg::Error(dg::Message(_ping_)<<"Tableau conversion failed!");
}

template<class real_type>
MultistepTableau<real_type> lmstableau( enum multistep_identifier id)
{
    switch(id){
        case IMEX_EULER_1_1:
            return dg::tableau::imex_euler_1_1<real_type>();
        case IMEX_ADAMS_2_2:
            return dg::tableau::imex_adams_2_2<real_type>();
        case IMEX_KOTO_2_2:
            return dg::tableau::imex_koto_2_2<real_type>();
        case IMEX_BDF_2_2:
            return dg::tableau::imex_bdf<real_type>(2);
        case IMEX_BDF_3_3:
            return dg::tableau::imex_bdf<real_type>(3);
        case IMEX_BDF_4_4:
            return dg::tableau::imex_bdf<real_type>(4);
        case IMEX_BDF_5_5:
            return dg::tableau::imex_bdf<real_type>(5);
        case IMEX_BDF_6_6:
            return dg::tableau::imex_bdf<real_type>(6);
        case IMEX_TVB_3_3:
            return dg::tableau::imex_tvb<real_type>(3);
        case IMEX_TVB_4_4:
            return dg::tableau::imex_tvb<real_type>(4);
        case IMEX_TVB_5_5:
            return dg::tableau::imex_tvb<real_type>(5);
        case AB_1_1:
            return dg::tableau::ab<real_type>(1);
        case AB_2_2:
            return dg::tableau::ab<real_type>(2);
        case AB_3_3:
            return dg::tableau::ab<real_type>(3);
        case AB_4_4:
            return dg::tableau::ab<real_type>(4);
        case AB_5_5:
            return dg::tableau::ab<real_type>(5);
        case eBDF_1_1:
            return dg::tableau::imex_euler_1_1<real_type>();
        case eBDF_2_2:
            return dg::tableau::imex_bdf<real_type>(2);
        case eBDF_3_3:
            return dg::tableau::imex_bdf<real_type>(3);
        case eBDF_4_4:
            return dg::tableau::imex_bdf<real_type>(4);
        case eBDF_5_5:
            return dg::tableau::imex_bdf<real_type>(5);
        case eBDF_6_6:
            return dg::tableau::imex_bdf<real_type>(6);
        case TVB_1_1:
            return dg::tableau::imex_euler_1_1<real_type>();
        case TVB_2_2:
            return dg::tableau::tvb<real_type>(2);
        case TVB_3_3:
            return dg::tableau::tvb<real_type>(3);
        case TVB_4_4:
            return dg::tableau::tvb<real_type>(4);
        case TVB_5_5:
            return dg::tableau::tvb<real_type>(5);
        case TVB_6_6:
            return dg::tableau::tvb<real_type>(6);
        case SSP_1_1:
            return dg::tableau::ssp<real_type>(1);
        case SSP_2_2:
            return dg::tableau::ssp<real_type>(2);
        case SSP_3_2:
            return dg::tableau::ssp<real_type>(3);
        case SSP_4_2:
            return dg::tableau::ssp<real_type>(4);
        case SSP_5_3:
            return dg::tableau::ssp<real_type>(5);
        case SSP_6_3:
            return dg::tableau::ssp<real_type>(6);
        case BDF_1_1:
            return dg::tableau::imex_euler_1_1<real_type>();
        case BDF_2_2:
            return dg::tableau::imex_bdf<real_type>(2);
        case BDF_3_3:
            return dg::tableau::imex_bdf<real_type>(3);
        case BDF_4_4:
            return dg::tableau::imex_bdf<real_type>(4);
        case BDF_5_5:
            return dg::tableau::imex_bdf<real_type>(5);
        case BDF_6_6:
            return dg::tableau::imex_bdf<real_type>(6);
    }
    return MultistepTableau<real_type>(); //avoid compiler warning
}


template<class real_type>
MultistepTableau<real_type> lmstableau( std::string name)
{
    return lmstableau<real_type>( str2lmstableau(name));
}

}//namespace create
///@endcond
/*! @class hide_imex_multistep_tableaus
 *
 * We follow the naming convention
 * as **NAME-S-Q**
 *  - NAME is the author or name of the method
 *  - S is the number of steps in the method
 *  - Q is the global order of the method
 *
 *    Name  | Identifier | Description
 *   -------|------------| -----------
 *   ImEx-Euler-1-1         | dg::IMEX_EULER_1_1 | Explicit Euler combined with Implicit Euler
 *   Euler                  | dg::IMEX_EULER_1_1 | For convenience
    ImEx-Koto-2-2 | dg::IMEX_KOTO_2_2 | <a href="https://dx.doi.org/10.1007/s11464-009-0005-9">Koto T. Front. Math. China 2009, 4(1): 113-129</a> A stabilized 2nd order scheme with a large region of stability
    ImEx-Adams-2-2 | dg::IMEX_ADAMS_2_2 | <a href="https://dx.doi.org/10.1016/j.jcp.2007.03.003">Hundsdorfer and Ruuth, Journal of Computational Physics 225 (2007)</a> @note **(C=0.44)**
    ImEx-BDF-X-X | dg::IMEX_BDF_X_X | The family of schems described in <a href="https://dx.doi.org/10.1016/j.jcp.2007.03.003">Hundsdorfer and Ruuth, Journal of Computational Physics 225 (2007)</a> <br>The implicit part is a normal BDF scheme https://en.wikipedia.org/wiki/Backward_differentiation_formula while the explicit part equals the Minimal Projecting method by <a href = "https://www.ams.org/journals/mcom/1979-33-148/S0025-5718-1979-0537965-0/S0025-5718-1979-0537965-0.pdf"> Alfeld, P., Math. Comput. 33.148 1195-1212 (1979)</a> or **extrapolated BDF** in <a href = "https://doi.org/10.1137/S0036142902406326"> Hundsdorfer, W., Ruuth, S. J., & Spiteri, R. J. (2003). Monotonicity-preserving linear multistep methods. SIAM Journal on Numerical Analysis, 41(2), 605-623 </a> <br> @note Possible values for **X: 1 (C=1.00), 2 (C=0.63), 3 (C=0.39), 4 (C=0.22), 5 (C=0.09), 6** <br> Note that X=3 is identical to the "Karniadakis" scheme
    * Karniadakis | dg::IMEX_BDF_3_3 | The ImEx-BDF-3-3 scheme is identical to the widely used "Karniadakis" scheme <a href = "https://dx.doi.org/10.1016/0021-9991(91)90007-8"> Karniadakis, et al. J. Comput. Phys. 97 (1991)</a>
    ImEx-TVB-X-X | dg::IMEX_TVB_X_X | The family of schems described in < <a href="https://dx.doi.org/10.1016/j.jcp.2007.03.003">Hundsdorfer and Ruuth, Journal of Computational Physics 225 (2007)</a> <br> The explicit part is a TVB scheme while the implicit part is optimized to maximize damping of high wavelength <br> @note Possible values for **X: 3 (C=0.54), 4 (C=0.46), 5 (C=0.38)**
    *
 @note the CFL coefficient C is given relative to the forward Euler method: \f$ \Delta t < C \Delta t_{FE}\f$.
 @attention The coefficient C is the one that ensures the TVD property of the scheme and is **not** directly related to the stability region of the scheme
 */

/*! @class hide_explicit_multistep_tableaus
 *
 * We follow the naming convention
 * as **NAME-S-Q**
 *  - NAME is the author or name of the method
 *  - S is the number of steps in the method
 *  - Q is the global order of the method

 *    Name  | Identifier | Description
 *   -------|------------| -----------
 *   AB-X-X | dg::AB_X_X | The family of schemes described in <a href = "https://en.wikipedia.org/wiki/Linear_multistep_method"> Linear multistep methods </a> as **Adams-Bashforth** \f[ u^{n+1} = u^n + \Delta t\sum_{j=0}^{s-1} b_j f\left(t^n - j \Delta t, u^{n-j}\right) \f] @note **Possible stages are X: 1, 2,..., 5**, the order of the method is the same as its stages @note The Adams-Bashforth schemes implemented here need less storage but may have **a smaller region of absolute stability** than for example an extrapolated BDF method of the same order.
 * eBDF-X-X | dg::eBDF_X_X | The family of schemes described in <a href = "https://doi.org/10.1137/S0036142902406326"> Hundsdorfer, W., Ruuth, S.  J., & Spiteri, R. J. (2003). Monotonicity-preserving linear multistep methods. SIAM Journal on Numerical Analysis, 41(2), 605-623 </a> as **extrapolated BDF**  where it is found to be TVB (**total variation bound**). The schemes also appear as **Minimal Projecting** scheme described in <a href = "https://www.ams.org/journals/mcom/1979-33-148/S0025-5718-1979-0537965-0/S0025-5718-1979-0537965-0.pdf"> Alfeld, P., Math. Comput. 33.148 1195-1212 (1979)</a> <br> @note **Possible stages are X: 1 (C=1), 2 (C=0.63), 3 (C=0.39), 4 (C=0.22), 5 (C=0.09), 6** with the order the same as the number of stages
 * TVB-X-X | dg::TVB_X_X | The family of schemes described in <a href="https://doi.org/10.1016/j.jcp.2005.02.029">S.J. Ruuth and W. Hundsdorfer, High-order linear multistep methods with general monotonicity and boundedness properties, Journal of Computational Physics, Volume 209, Issue 1, 2005 </a> as Total variation Bound. These schemes have larger allowable step sizes than the eBDF family, <br> @note **Possible values for X are 1 (C=1), 2 (C=0.5), 3 (C=0.54), 4 (C=0.46), 5 (C=0.38) 6 (C=0.33)**. We highlight that TVB-3-3 has 38% larger allowable stepsize than eBDF-3-3 and TVB-4-4 has 109% larger stepsize than eBDF-4-4 (to ensure the TVB property, not stability).
 * SSP-X-Y | dg::SSP_X_Y | The family of schemes described in <a href="https://doi.org/10.1007/BF02728985">Gottlieb, S. On high order strong stability preserving runge-kutta and multi step time discretizations. J Sci Comput 25, 105–128 (2005)</a> as Strong Stability preserving. We implement the lowest order schemes for each stage and disregard the remaining schemes in the paper since their CFL conditions are worse than the TVB scheme of the same order.  @note **Possible values for X-Y : 1-1 (C=1), 2-2 (C=0.5), 3-2 (C=0.5), 4-2 (C=0.66), 5-3 (C=0.5), 6-3 (C=0.567)**.@note These schemes are noteworthy because the coefficients b_i are all positive except for the 2-2 method and **the "4-2" and "6-3" methods allow slightly larger allowable stepsize but increased storage requirements than TVB** of same order (2 and 3).

 *@note Total variation bound (TVB) means \f$ || v^n|| \leq M ||v^0||\f$  where the norm signifies the total variation semi-norm. Total variation diminishing
     (TVD) means M=1, and strong stability preserving (SSP) is the same as TVD, TVB schemes converge to the correct entropy solutions of hyperbolic conservation laws
 @note the CFL coefficient C is given relative to the forward Euler method: \f$ \Delta t < C \Delta t_{FE}\f$.
 @attention The coefficient C is the one that ensures the TVD property of the scheme and is **not** directly related to the stability region of the scheme
 */

/*! @class hide_implicit_multistep_tableaus
 *
 * We follow the naming convention
 * as **NAME-S-Q**
 *  - NAME is the author or name of the method
 *  - S is the number of steps in the method
 *  - Q is the global order of the method
 *
 *    Name  | Identifier | Description
 *   -------|------------| -----------
 *   BDF-X-X | dg::BDF_X_X | The coefficients for backward differences can be found at https://en.wikipedia.org/wiki/Backward_differentiation_formula <br> @note **Possible values for X: 1, 2, 3, 4, 5, 6** @note A BDF scheme is simply constructed by discretizing the time derivative with a n-th order backward difference formula and evaluating the right hand side at the new timestep. @note Methods with s>6 are not zero-stable so they cannot be used
 *
*/

/*! @brief Convert identifiers to their corresponding \c dg::MultistepTableau
 *
 * This is a helper class to simplify the interfaces of our timestepper functions and classes.
 * The sole purpose is to implicitly convert either a MultistepTableau or one of
 * the following identifiers to an instance of a MultistepTableau.
 *
 * Explicit methods
 * @copydoc hide_explicit_multistep_tableaus
 * Implicit methods
 * @copydoc hide_implicit_multistep_tableaus
 * Implicit-Explicit methods
 * @copydoc hide_imex_multistep_tableaus
 * @param real_type The type of the coefficients in the MultistepTableau
 * @ingroup time_utils
 */
template<class real_type>
struct ConvertsToMultistepTableau
{
    using value_type = real_type;
    ///Of course a MultistepTableau converts to a MultistepTableau
    ///Useful if you constructed your very own coefficients
    ConvertsToMultistepTableau( MultistepTableau<real_type> tableau): m_t(tableau){}

    /*! @brief Create MultistepTableau from \c dg::tableau_identifier
    *
    * The use of this constructor might be a bit awkward because you'll have to write all caps.
    * @param id the identifier, for example \c dg::eBDF_3_3
    */
    ConvertsToMultistepTableau( enum tableau_identifier id):m_t( dg::create::lmstableau<real_type>(id)){}
    /*! @brief Create MultistepTableau from its name (very useful)
    *
    * Explicit methods
    * @copydoc hide_explicit_multistep_tableaus
    * Implicit methods
    * @copydoc hide_implicit_multistep_tableaus
    * Implicit-Explicit methods
    * @copydoc hide_imex_multistep_tableaus
    * @param name The name of the tableau as stated in the Name column above, as a string, for example "eBDF-3-3"
    */
    ConvertsToMultistepTableau( std::string name):m_t(
            dg::create::lmstableau<real_type>(name)){}

    ///@copydoc ConvertsToMultistepTableau(std::string)
    ConvertsToMultistepTableau( const char* name):m_t(
            dg::create::lmstableau<real_type>(std::string(name))){}
    ///Convert to MultistepTableau
    ///
    ///which means an object can be directly assigned to a MultistepTableau
    operator MultistepTableau<real_type>( )const{
        return m_t;
    }
    private:
    MultistepTableau<real_type> m_t;
};

}//namespace dg
