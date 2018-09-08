#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <array>

#include "backend/exceptions.h"
#include "blas1.h"


/*! @file
 * @brief contains runge-kutta explicit time-integrators
 */
namespace dg{

//namespace detail <--------- ??

/*! @brief coefficients for explicit RK methods
 *
 * The coefficients are in the form Cockburn proposed in his paper.
 * It's just a reformulation in that you don't store the sequence of
 * k_j but rather the abscissas u_j with k_j = f(u_j)
 *  Note that if you knew all k_j you can compute this
 *  sequence u_j via u=Bk. To derive these coefficients from the butcher tableau
 * consider.
 * \f[ u = Bhk = (B-D)hk + Dhk = (B-D)B^{-1}y + Dhk
 *       = ( 1- DB^{-1})u + Dhk = \alpha u + \beta h k\f]
 *  where \f$ B\f$ is the butcher tableau without the c's and extended
 *  by ones on the left and \f$ D\f$ its
 *  diagonal part.
 * @tparam k Order of the method. Currently 2,3 and 4 are available
 */
template< size_t k, class real_type>
struct rk_coeff;
/*! @brief coefficients for explicit RK methods
 *
 * The coefficients are in the classical form
 * * @tparam s # of stages of the method. Currently 1,2,3,4 and 6 are available
 */
template< size_t s, class real_type>
struct rk_classic;
///@cond
/*
template<>
const real_type rk_coeff<1>::alpha[1][1] = { {1}};
template<>
const real_type rk_coeff<1>::beta[1] = {1};
*/

//from Cockburn paper
template<class real_type>
struct rk_coeff<2, real_type>{
const real_type alpha[2][2] = {
    { 1,   0},
    { 0.5, 0.5}
};
const real_type beta[2] = { 1, 0.5 };
};
//from Cockburn paper
template<class real_type>
struct rk_coeff<3, real_type>{
const real_type alpha[3][3] = {
    { 1,     0,    0},
    { 0.75,  0.25, 0},
    { 1./3., 0.,   2./3.}
};
const real_type beta[3] = {
     1, 0.25, 2./3.
};
};
template<class real_type>
struct rk_coeff<4, real_type>{
//classic RK4 coefficients (matlab used to compute from normal form)
const real_type alpha[4][4] = {
    { 1,    0., 0, 0 },
    { 1.,   0., 0, 0 },
    { 1.,   0., 0, 0 },
    {-1./3., 1./3., 2./3., 1./3.}
};
const real_type beta[4] = {
     0.5, 0.5, 1.0, 1./6.
};
};
/////////////////////rk_classic//////////////////
template<class real_type>
struct rk_classic<1, real_type>{
const real_type a[1][1] = {
    {0}
};
const real_type b[1] = {
    1.
};
};
template<class real_type>
struct rk_classic<2, real_type>{
const real_type a[2][2] = {
    {0,0},
    {.5,0}
};
const real_type b[2] = {
    0.,1.
};
};
template<class real_type>
struct rk_classic<3, real_type>{
const real_type a[3][3] = {
    {0,0,0},
    {.5,0,0},
    {-1,2,0},
};
const real_type b[3] = {
    1./6.,2./3.,1./6.
};
};
template<class real_type>
struct rk_classic<4, real_type>{
const real_type a[4][4] = {
    {0,0,0,0},
    {0.5, 0,0,0},
    {0,0.5,0,0},
    {0,0,1,0}
};
const real_type b[4] = {
    1./6., 1./3., 1./3., 1./6.
};
};
template<class real_type>
struct rk_classic<6, real_type>{
//Fehlberg
const real_type a[6][6] = {
    {0,0,0,0,0,0},
    {0.25, 0,0,0,0,0},
    {3./32., 9./32.,0,0,0,0},
    {1932./2197., -7200./2197., 7296./2197.,0,0,0},
    {439./216., -8, 3680./513.,   -845./4104.,0,0},
    {-8./27.,   2.,   -3544./2565.,  1859./4104.,   -11./40., 0}
};
const real_type b[6] = {
    16./135.,  0,   6656./12825.,  28561./56430.,     -9./50.,   2./55.
};

};
template<class real_type>
struct rk_classic<17, real_type>{
//RK(10)
const real_type a[17][17] = {
{0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0},
{0.100000000000000000000000000000000000000000000000000000000000, 0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0},
{-0.915176561375291440520015019275342154318951387664369720564660,
 1.45453440217827322805250021715664459117622483736537873607016,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0},
{0.202259190301118170324681949205488413821477543637878380814562,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.606777570903354510974045847616465241464432630913635142443687,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0},
{0.184024714708643575149100693471120664216774047979591417844635,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.197966831227192369068141770510388793370637287463360401555746,
-0.0729547847313632629185146671595558023015011608914382961421311,0, 0,0,0,0,0, 0,0,0,0,0, 0,0},
{ 0.0879007340206681337319777094132125475918886824944548534041378,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.410459702520260645318174895920453426088035325902848695210406,
 0.482713753678866489204726942976896106809132737721421333413261, 0,0,0,0,0, 0,0,0,0,0, 0,0},
{ 0.0859700504902460302188480225945808401411132615636600222593880,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.330885963040722183948884057658753173648240154838402033448632,
 0.489662957309450192844507011135898201178015478433790097210790,
-0.0731856375070850736789057580558988816340355615025188195854775, 0,0,0,0, 0,0,0,0,0, 0,0},
{ 0.120930449125333720660378854927668953958938996999703678812621,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.260124675758295622809007617838335174368108756484693361887839,
 0.0325402621549091330158899334391231259332716675992700000776101,
-0.0595780211817361001560122202563305121444953672762930724538856, 0,0,0, 0,0,0,0,0, 0,0},
{0.110854379580391483508936171010218441909425780168656559807038,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.0605761488255005587620924953655516875526344415354339234619466,
 0.321763705601778390100898799049878904081404368603077129251110,
 0.510485725608063031577759012285123416744672137031752354067590, 0,0, 0,0,0,0,0, 0,0},
{0.112054414752879004829715002761802363003717611158172229329393,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.144942775902865915672349828340980777181668499748506838876185,
-0.333269719096256706589705211415746871709467423992115497968724,
 0.499269229556880061353316843969978567860276816592673201240332,
 0.509504608929686104236098690045386253986643232352989602185060, 0, 0,0,0,0,0, 0,0},
{0.113976783964185986138004186736901163890724752541486831640341,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.0768813364203356938586214289120895270821349023390922987406384,
 0.239527360324390649107711455271882373019741311201004119339563,
 0.397774662368094639047830462488952104564716416343454639902613,
 0.0107558956873607455550609147441477450257136782823280838547024,
-0.327769124164018874147061087350233395378262992392394071906457, 0,0,0,0,0, 0,0},
{0.0798314528280196046351426864486400322758737630423413945356284,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.0520329686800603076514949887612959068721311443881683526937298,
-0.0576954146168548881732784355283433509066159287152968723021864,
 0.194781915712104164976306262147382871156142921354409364738090,
 0.145384923188325069727524825977071194859203467568236523866582,
-0.0782942710351670777553986729725692447252077047239160551335016,
-0.114503299361098912184303164290554670970133218405658122674674, 0,0,0,0, 0,0},
{0.985115610164857280120041500306517278413646677314195559520529,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.330885963040722183948884057658753173648240154838402033448632,
 0.489662957309450192844507011135898201178015478433790097210790,
-1.37896486574843567582112720930751902353904327148559471526397,
-0.861164195027635666673916999665534573351026060987427093314412,
 5.78428813637537220022999785486578436006872789689499172601856,
 3.28807761985103566890460615937314805477268252903342356581925,
-2.38633905093136384013422325215527866148401465975954104585807,
-3.25479342483643918654589367587788726747711504674780680269911,
-2.16343541686422982353954211300054820889678036420109999154887, 0,0,0, 0,0},
{0.895080295771632891049613132336585138148156279241561345991710,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.197966831227192369068141770510388793370637287463360401555746,
-0.0729547847313632629185146671595558023015011608914382961421311,
 0.0000000000000000000000000000000000000000000000000000000000000,
-0.851236239662007619739049371445966793289359722875702227166105,
 0.398320112318533301719718614174373643336480918103773904231856,
 3.63937263181035606029412920047090044132027387893977804176229,
 1.54822877039830322365301663075174564919981736348973496313065,
-2.12221714704053716026062427460427261025318461146260124401561,
-1.58350398545326172713384349625753212757269188934434237975291,
-1.71561608285936264922031819751349098912615880827551992973034,
-0.0244036405750127452135415444412216875465593598370910566069132, 0,0, 0,0},
{-0.915176561375291440520015019275342154318951387664369720564660,
 1.45453440217827322805250021715664459117622483736537873607016,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.777333643644968233538931228575302137803351053629547286334469,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.0910895662155176069593203555807484200111889091770101799647985,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.0910895662155176069593203555807484200111889091770101799647985,
 0.777333643644968233538931228575302137803351053629547286334469, 0, 0,0},
{0.100000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
-0.157178665799771163367058998273128921867183754126709419409654,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.157178665799771163367058998273128921867183754126709419409654, 0,0},
{0.181781300700095283888472062582262379650443831463199521664945,
 0.675000000000000000000000000000000000000000000000000000000000,
 0.342758159847189839942220553413850871742338734703958919937260,
 0.000000000000000000000000000000000000000000000000000000000000,
 0.259111214548322744512977076191767379267783684543182428778156,
-0.358278966717952089048961276721979397739750634673268802484271,
-1.04594895940883306095050068756409905131588123172378489286080,
 0.930327845415626983292300564432428777137601651182965794680397,
 1.77950959431708102446142106794824453926275743243327790536000,
 0.100000000000000000000000000000000000000000000000000000000000,
-0.282547569539044081612477785222287276408489375976211189952877,
-0.159327350119972549169261984373485859278031542127551931461821,
-0.145515894647001510860991961081084111308650130578626404945571,
-0.259111214548322744512977076191767379267783684543182428778156,
-0.342758159847189839942220553413850871742338734703958919937260,
-0.675000000000000000000000000000000000000000000000000000000000, 0}
};

const real_type b[17] = {
0.0333333333333333333333333333333333333333333333333333333333333,
0.0250000000000000000000000000000000000000000000000000000000000,
0.0333333333333333333333333333333333333333333333333333333333333,
0.000000000000000000000000000000000000000000000000000000000000,
0.0500000000000000000000000000000000000000000000000000000000000,
0.000000000000000000000000000000000000000000000000000000000000,
0.0400000000000000000000000000000000000000000000000000000000000,
0.000000000000000000000000000000000000000000000000000000000000,
0.189237478148923490158306404106012326238162346948625830327194,
0.277429188517743176508360262560654340428504319718040836339472,
0.277429188517743176508360262560654340428504319718040836339472,
0.189237478148923490158306404106012326238162346948625830327194,
-0.0400000000000000000000000000000000000000000000000000000000000,
-0.0500000000000000000000000000000000000000000000000000000000000,
-0.0333333333333333333333333333333333333333333333333333333333333,
-0.0250000000000000000000000000000000000000000000000000000000000,
0.0333333333333333333333333333333333333333333333333333333333333
};
};
///@endcond

 /** @class hide_rhs
  * @tparam RHS The right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
  */

/**
* @brief Struct for Runge-Kutta explicit time-integration optimized for few vector additions
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]
*
* @ingroup time
*
* Uses only \c dg::blas1::axpby() routines to integrate one step.
 * The coefficients are in the form that is optimized for number of vector additions.
 * It's just a reformulation in that you don't store the sequence of
 * \f$ k_j\f$ but rather the abscissas \f$ u_j\f$  with \f$ k_j = f(u_j)\f$
 *  Note that if you knew all \f$ k_j\f$ you can compute this
 *  sequence \f$ u_j\f$ via \f$ u=Bk\f$. To derive these coefficients from the butcher tableau
 * consider.
 * \f[ u = Bhk = (B-D)hk + Dhk = (B-D)B^{-1}y + Dhk
 *       = ( 1- DB^{-1})u + Dhk = \alpha u + \beta h k\f]
 *  where \f$ B\f$ is the Butcher tableau without the c's and extended
 *  by ones on the left and \f$ D\f$ its
 *  diagonal part.
* @tparam s Order of the method (1, 2, 3 or 4)
* @copydoc hide_ContainerType
* @sa RK
*/
template< size_t s, class ContainerType>
struct RK_opt
{
    using real_type = get_value_type<ContainerType>;
    ///@brief No memory allocation, Call \c construct before using the object
    RK_opt(){}

    ///@copydoc construct()
    RK_opt( const ContainerType& copyable){
        construct(copyable);
    }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param copyable ContainerType of the size that is used in \c step
    * @note it does not matter what values \c copyable contains, but its size is important
    */
    void construct( const ContainerType& copyable){
        u_.fill(copyable);
    }

    /**
    * @brief Advance one step
    *
    * @copydoc hide_rhs
    * @param rhs right hand side subroutine
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on output, may alias \c t0)
    * @param u1 (write only) contains result on output (may alias u0)
    * @param dt timestep
    */
    template< class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt);
    ///global order of the algorithm
    constexpr size_t order() const {
        return s;
    }
  private:
    std::array<ContainerType,s> u_; //the order determines the amount of memory needed
    rk_coeff<s,real_type> m_rk;
};

template< size_t k, class ContainerType>
template< class RHS>
void RK_opt<k, ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
{
    f(t0, u0, u_[0]);
    blas1::axpby( m_rk.alpha[0][0], u0, dt*m_rk.beta[0], u_[0]);
    std::array<real_type,k-1> tu;
    tu[0] =  dt*m_rk.beta[0];
    tu[0] = DG_FMA( (m_rk.alpha[0][0]), t0, tu[0]);
    for( unsigned i=1; i<k-1; i++)
    {
        f(tu[i-1], u_[i-1], u_[i] );
        blas1::axpby( m_rk.alpha[i][0], u0, dt*m_rk.beta[i], u_[i]);
        tu[i] = dt*m_rk.beta[i];
        tu[i] = DG_FMA( m_rk.alpha[i][0],t0,tu[i]);
        for( unsigned l=1; l<=i; l++)
        {
            blas1::axpby(   m_rk.alpha[i][l], u_[l-1], 1., u_[i]);
            tu[i] = DG_FMA( m_rk.alpha[i][l], tu[l-1], tu[i]);
        }
    }
    //Now add everything up to u1
    f(tu[k-2], u_[k-2], u_[k-1]); //u1 may alias u0, so we need u_[k-1]
    blas1::axpby( m_rk.alpha[k-1][0], u0, dt*m_rk.beta[k-1], u_[k-1], u1);
    for( unsigned l=1; l<=k-1; l++)
        blas1::axpby(m_rk.alpha[k-1][l], u_[l-1],1., u1);
    t1 = t0 + dt;
}
///@cond
//Euler specialisation
template < class ContainerType>
struct RK_opt<1, ContainerType>
{
    using real_type = get_value_type<ContainerType>;
    RK_opt(){}
    RK_opt( const ContainerType& copyable){
        construct(copyable);
    }
    void construct( const ContainerType& copyable){
        u_ = copyable;
    }
    template < class RHS>
    void step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
    {
        f( t0, u0, u_); //we need u_ if u1 aliases u0
        blas1::axpby( 1., u0, dt, u_, u1);
        t1 = t0 + dt;
    }
    ///global order of the algorithm
    constexpr size_t order() const {
        return 1;
    }
    private:
    ContainerType u_;
};
///@endcond

/**
* @brief Struct for Runge-Kutta explicit time-integration, classic formulation
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]

@snippet runge_kutta_t.cu function
@snippet runge_kutta_t.cu doxygen
* @ingroup time
*
* Uses only \c dg::blas1::axpby() routine to integrate one step.
* The coefficients are chosen in the classic form given by Runge and Kutta.
* Needs more vector additions than our RK_opt class but we implemented higher orders
* @tparam s Order of the method (1, 2, 3, 4, 6, 17)
* @copydoc hide_ContainerType
*/
template< size_t s, class ContainerType>
struct RK
{
    using real_type = get_value_type<ContainerType>;
    ///@copydoc RK_opt::RK_opt()
    RK(){}
    ///@copydoc RK_opt::construct(const ContainerType&)
    RK( const ContainerType& copyable){
        construct( copyable);
    }
    ///@copydoc RK_opt::construct(const ContainerType&)
    void construct( const ContainerType& copyable){
        k_.fill(copyable);
        u_ = copyable;
    }
    ///@copydoc RK_opt::step(RHS&,real_type,const ContainerType&,real_type&,ContainerType&,real_type)
    template<class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt);
    ///global order of the algorithm
    constexpr size_t order() const {
        return s;
    }
  private:
    std::array<ContainerType,s> k_;
    ContainerType u_;
    rk_classic<s, real_type> m_rk;
};

template< size_t s, class ContainerType>
template< class RHS>
void RK<s, ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt)
{
    f(t0, u0, k_[0]); //compute k_0
    real_type tu = t0;
    for( unsigned i=1; i<s; i++) //compute k_i
    {
        blas1::axpby( 1., u0, dt*m_rk.a[i][0],k_[0], u_); //l=0
        tu = DG_FMA( dt,m_rk.a[i][0],t0); //l=0
        for( unsigned l=1; l<i; l++)
        {
            blas1::axpby( dt*m_rk.a[i][l], k_[l],1., u_);
            tu = DG_FMA(dt,m_rk.a[i][l],tu);
        }
        f( tu, u_, k_[i]);

    }
    //Now add everything up to u1
    blas1::axpby( dt*m_rk.b[0], k_[0],1.,u0, u1);
    for( unsigned i=1; i<s; i++)
        blas1::axpby( dt*m_rk.b[i], k_[i],1., u1);
    t1 = t0 + dt;
}

///@addtogroup time
///@{

/**
 * @brief Integrate differential equation with a stage s Runge-Kutta scheme and a fixed number of steps
 *
 * @tparam s # of stages (1, 2, 3, 4, 6, 17)
 * @copydoc hide_rhs
 * @copydoc hide_ContainerType
 * @param rhs The right-hand-side
 * @param t_begin initial time
 * @param begin initial condition
 * @param t_end final time
 * @param end (write-only) contains solution at \c t_end on output (may alias begin)
 * @param N number of steps
 */
template< unsigned s, class RHS, class ContainerType>
void stepperRK(RHS& rhs, get_value_type<ContainerType>  t_begin, const ContainerType& begin, get_value_type<ContainerType> t_end, ContainerType& end, unsigned N )
{
    using real_type = get_value_type<ContainerType>;
    RK<s, ContainerType > rk( begin);
    if( t_end == t_begin){ end = begin; return;}
    real_type dt = (t_end-t_begin)/(real_type)N;
    end = begin;
    real_type t0 = t_begin;
    for( unsigned i=0; i<N; i++)
        rk.step( rhs, t0, end, t0, end, dt);
}


/**
 * @brief Integrates the differential equation using a stage s Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
 *
 * Doubles the number of timesteps until the desired accuracy is reached
 *
 * @tparam s Order of the method (1, 2, 3, 4, 6, 17)
 * @copydoc hide_rhs
 * @tparam RHS
 * In addition, there must be the function \c bool \c monitor( const ContainerType& end);
 * available, which is called after every step.
 * Return \c true if everything is ok and \c false if the integrator certainly fails.
 * The other function is the \c real_type \c error( const ContainerType& end0, const ContainerType& end1); which computes the error norm in which the integrator should converge.
 * @copydoc hide_ContainerType
 * @param rhs The right-hand-side
 * @param t_begin initial time
 * @param begin initial condition
 * @param t_end final time
 * @param end (write-only) contains solution on output
 * @param eps_abs desired accuracy in the error function between \c end and \c end_old
 * @param NT_init initial number of steps
 * @return number of iterations if converged, -1 and a warning to \c std::cerr when \c isnan appears, -2 if failed to reach \c eps_abs
 * @attention This function is superseded by the better \c dg::integrateHRK Halfstep function
 */
template<unsigned s, class RHS, class ContainerType>
int integrateRK(RHS& rhs, get_value_type<ContainerType> t_begin, const ContainerType& begin, get_value_type<ContainerType> t_end, ContainerType& end, get_value_type<ContainerType> eps_abs, unsigned NT_init = 2 )
{
    using real_type = get_value_type<ContainerType>;
    RK<s, ContainerType > rk( begin);
    ContainerType old_end(begin);
    blas1::copy( begin, end );
    if( t_end == t_begin) return 0;
    int NT = NT_init;
    real_type dt = (t_end-t_begin)/(real_type)NT;
    real_type error = 1e10;
    real_type t0 = t_begin;

    while( error > eps_abs && NT < pow( 2, 18) )
    {
        blas1::copy( begin, end );

        int i=0;
        while (i<NT)
        {
            rk.step( rhs, t0, end, t0, end, dt);
            if( !rhs.monitor( end ) )  //sanity check
            {
                #ifdef DG_DEBUG
                    std::cout << "---------Got sanity error -> choosing smaller step size and redo integration" << " NT "<<NT<<" dt "<<dt<< std::endl;
                #endif
                break;
            }
            i++;
        }
        error = rhs.error( end, old_end);
        blas1::copy( end, old_end);
        t0 = t_begin;
        dt /= 2.;
        NT *= 2;
    }
    if( std::isnan( error) )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is NAN! "<<std::endl;
        return -1;
    }
    if( error > eps_abs )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is "<<error<<" with "<<NT<<" steps"<<std::endl;
        return -2;
    }
    return NT;


}

///@}

} //namespace dg

#endif //_DG_RK_
