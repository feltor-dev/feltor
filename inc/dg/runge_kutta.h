#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <vector>

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
template< size_t k>
struct rk_coeff
{
    static const double alpha[k][k];  //!< alpha
    static const double beta[k]; //!< beta
};
/*! @brief coefficients for explicit RK methods
 *
 * The coefficients are in the classical form  
 * * @tparam s # of stages of the method. Currently 1,2,3,4 and 6 are available
 */
template< size_t s>
struct rk_classic
{
    static const double a[s][s];  //!< a
    static const double b[s]; //!< b
};
///@cond
/*
template<>
const double rk_coeff<1>::alpha[1][1] = { {1}};
template<>
const double rk_coeff<1>::beta[1] = {1};
*/

//from Cockburn paper
template<>
const double rk_coeff<2>::alpha[2][2] = {
    { 1,   0},
    { 0.5, 0.5}
};
template<>
const double rk_coeff<2>::beta[2] = {
     1, 0.5
};
//from Cockburn paper
template<>
const double rk_coeff<3>::alpha[3][3] = {
    { 1,     0,    0},
    { 0.75,  0.25, 0},
    { 1./3., 0.,   2./3.}
};
template<>
const double rk_coeff<3>::beta[3] = {
     1, 0.25, 2./3.
};
//classic RK4 coefficients (matlab used to compute from normal form)
template<>
const double rk_coeff<4>::alpha[4][4] = {
    { 1,    0., 0, 0 },
    { 1.,   0., 0, 0 },
    { 1.,   0., 0, 0 },
    {-1./3., 1./3., 2./3., 1./3.}
};
template<>
const double rk_coeff<4>::beta[4] = {
     0.5, 0.5, 1.0, 1./6.
};
template<> 
const double rk_classic<1>::a[1][1] = {
    {0}
};
template<> 
const double rk_classic<1>::b[1] = {
    1.
};
template<> 
const double rk_classic<2>::a[2][2] = {
    {0,0},
    {.5,0}
};
template<> 
const double rk_classic<2>::b[2] = {
    0.,1.
};
template<> 
const double rk_classic<3>::a[3][3] = {
    {0,0,0},
    {.5,0,0},
    {-1,2,0},
};
template<> 
const double rk_classic<3>::b[3] = {
    1./6.,2./3.,1./6.
};
template<> 
const double rk_classic<4>::a[4][4] = {
    {0,0,0,0},
    {0.5, 0,0,0},
    {0,0.5,0,0},
    {0,0,1,0}
};
template<> 
const double rk_classic<4>::b[4] = {
    1./6., 1./3., 1./3., 1./6.
};
//Fehlberg
template<> 
const double rk_classic<6>::a[6][6] = {
    {0,0,0,0,0,0},
    {0.25, 0,0,0,0,0},
    {3./32., 9./32.,0,0,0,0},
    {1932./2197., -7200./2197., 7296./2197.,0,0,0},
    {439./216., -8, 3680./513.,   -845./4104.,0,0},
    {-8./27.,   2.,   -3544./2565.,  1859./4104.,   -11./40., 0}
};
template<> 
const double rk_classic<6>::b[6] = {
    16./135.,  0,   6656./12825.,  28561./56430.,     -9./50.,   2./55.
};

//RK(10)
template<> 
const double rk_classic<17>::a[17][17] = {
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

template<> 
const double rk_classic<17>::b[17] = {
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
///@endcond
//RHS contains Information about container type it uses
//k is the order of the method
// container f( const container& v)
// container should probably be rvalue assignable

/**
* @brief Struct for Runge-Kutta explicit time-integration
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]
*
* @ingroup time
*
* Uses only dg::blas1::axpby() routines to integrate one step.
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
* @tparam k Order of the method (1, 2, 3 or 4)
* @copydoc hide_container
*/
template< size_t k, class container>
struct RK
{
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable container of size which is used in integration. 
    * A container object must be copy-constructible from copyable.
    */
    RK( const container& copyable): u_(k-1, container(copyable)){ }
    /**
    * @brief Advance u0 one timestep
    *
    * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type container.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
    * @param f right hand side function
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may currently not be the same.
    * @param dt The timestep.
    */
    template< class Functor>
    void operator()( Functor& f, const container& u0, container& u1, double dt);
  private:
    std::vector<container> u_; //TODO std::array is more natural here (but unfortunately not available)
};

//u0 and u1 may not be the same vector
//TO DO: this might be cured by adding u0 first to u_[0] in the last step
//f( y, yp) where y is const and yp contains the result
template< size_t k, class container>
template< class Functor>
void RK<k, container>::operator()( Functor& f, const container& u0, container& u1, double dt)
{
    assert( &u0 != &u1);
    u1 = u0;
    f(u1, u_[0]);
    blas1::axpby( rk_coeff<k>::alpha[0][0], u0, dt*rk_coeff<k>::beta[0], u_[0]);
    for( unsigned i=1; i<k-1; i++)
    {
        f( u_[i-1], u_[i]);
        blas1::axpby( rk_coeff<k>::alpha[i][0], u0, dt*rk_coeff<k>::beta[i], u_[i]);
        for( unsigned l=1; l<=i; l++)
        {
            blas1::axpby( rk_coeff<k>::alpha[i][l], u_[l-1],1., u_[i]); //Fall alpha = 0 muss axpby abfangen!!
        }

    }
    //Now add everything up to u1
    f( u_[k-2], u1);
    blas1::axpby( rk_coeff<k>::alpha[k-1][0], u0, dt*rk_coeff<k>::beta[k-1], u1);
    for( unsigned l=1; l<=k-1; l++)
    {
        blas1::axpby( rk_coeff<k>::alpha[k-1][l], u_[l-1],1., u1);
    }
}

/**
* @brief Struct for Runge-Kutta explicit time-integration
* \f[
 \begin{align}
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    k_j = f\left( u^n + \Delta t \sum_{l=1}^j a_{jl} k_l\right)
 \end{align}
\f]
*
* @ingroup time 
*
* Uses only dg::blas1::axpby() routines to integrate one step.
* The coefficients are chosen in the classic form given by Runge and Kutta. 
* Needs more calls for axpby than our RK class but we implemented higher orders
* @tparam s Order of the method (1, 2, 3, 4, 6, 17)
* @copydoc hide_container
*/
template< size_t s, class container>
struct RK_classic
{
    /**
    * @brief Reserve memory for the integration
    *
    * @param copyable container of size which is used in integration. 
    * A container object must be copy-constructible from copyable.
    */
    RK_classic( const container& copyable): k_(s, container(copyable)), u_(copyable){ }
    /**
    * @brief Advance u0 one timestep
    *
    * @tparam Functor models BinaryFunction with no return type (subroutine)
        Its arguments both have to be of type container.
        The first argument is the actual argument, The second contains
        the return value, i.e. y' = f(y) translates to f( y, y').
    * @param f right hand side function
    * @param u0 initial value
    * @param u1 contains result on output. u0 and u1 may currently not be the same.
    * @param dt The timestep.
    */
    template< class Functor>
    void operator()( Functor& f, const container& u0, container& u1, double dt);
  private:
    std::vector<container> k_;
    container u_;
};

//u0 and u1 may not be the same vector
//TO DO: this might be cured by adding u0 first to u_[0] in the last step
//f( y, yp) where y is const and yp contains the result
template< size_t s, class container>
template< class Functor>
void RK_classic<s, container>::operator()( Functor& f, const container& u0, container& u1, double dt)
{
    assert( &u0 != &u1);
    u1 = u0; //to let u0 const and later for summation
    f(u1, k_[0]); //compute k_0
    for( unsigned i=1; i<s; i++) //compute k_i 
    {
        blas1::axpby( 1., u0, dt*rk_classic<s>::a[i][0],k_[0], u_); //l=0
        for( unsigned l=1; l<i; l++)
            blas1::axpby( dt*rk_classic<s>::a[i][l], k_[l],1., u_); 
        f( u_, k_[i]);

    }
    //Now add everything up to u1
    for( unsigned i=0; i<s; i++)
        blas1::axpby( dt*rk_classic<s>::b[i], k_[i],1., u1);
}

///@addtogroup time
///@{
/**
 * @brief Integrate differential equation with a stage s Runge-Kutta scheme and a fixed number of steps
 *
 * @tparam s # of stages (1, 2, 3, 4, 6, 17)
 * @copydetails stepperRK1()
 */
template< class RHS, class container, unsigned s>
void stepperRK(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    RK_classic<s, container > rk( begin); 
    container temp(begin);
    if( T_max == T_min){ end = begin; return;}
    double dt = (T_max-T_min)/(double)N;
    end = begin;
    for( unsigned i=0; i<N; i++)
    {
        rk( rhs, end, temp, dt); 
        end.swap( temp); //end is one step further 
    }
}

/**
 * @brief Integrate differential equation with a stage 1 Runge-Kutta scheme and a fixed number of steps
 *
 * @tparam RHS The right-hand side class
 * @copydoc hide_container
 * @param rhs The right-hand-side
 * @param begin initial condition 
 * @param end (write-only) contains solution on output
 * @param T_min initial time
 * @param T_max final time
 * @param N number of steps 
 */
template< class RHS, class container>
void stepperRK1(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    stepperRK<RHS, container, 1>( rhs, begin, end, T_min, T_max, N);
}
///@brief Integrate differential equation with a stage 2 Runge-Kutta scheme and a fixed number of steps
///@copydetails stepperRK1()
template< class RHS, class container>
void stepperRK2(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    stepperRK<RHS, container, 2>( rhs, begin, end, T_min, T_max, N);
}

///@brief Integrate differential equation with a stage 3 Runge-Kutta scheme and a fixed number of steps
///@copydetails stepperRK1()
template< class RHS, class container>
void stepperRK3(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    stepperRK<RHS, container, 3>( rhs, begin, end, T_min, T_max, N);
}

///@brief Integrate differential equation with a stage 4 Runge-Kutta scheme and a fixed number of steps
///@copydetails stepperRK1()
template< class RHS, class container>
void stepperRK4(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    stepperRK<RHS, container, 4>( rhs, begin, end, T_min, T_max, N);
}
///@brief Integrate differential equation with a stage 6 Runge-Kutta scheme and a fixed number of steps
///@copydetails stepperRK1()
template< class RHS, class container>
void stepperRK6(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    stepperRK<RHS, container, 6>( rhs, begin, end, T_min, T_max, N);
}
///@brief Integrate differential equation with a stage 17 Runge-Kutta scheme and a fixed number of steps
///@copydetails stepperRK1()
template< class RHS, class container>
void stepperRK17(RHS& rhs, const container& begin, container& end, double T_min, double T_max, unsigned N )
{
    stepperRK<RHS, container, 17>( rhs, begin, end, T_min, T_max, N);
}


/**
 * @brief Integrates the differential equation using a stage s Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
 *
 * Doubles the number of timesteps until the desired accuracy is reached
 *
 * @tparam RHS The right-hand side class. There must be the function bool monitor( const container& end); available which is called after every step. Return true if everything is ok and false if the integrator certainly fails.
 * The other function is the double error( const container& end0, const container& end1); which computes the error norm in which the integrator should converge. 
 * @copydoc hide_container
 * @param rhs The right-hand-side
 * @param begin initial condition (size 3)
 * @param end (write-only) contains solution on output
 * @param T_max time difference
 * @param eps_abs desired accuracy in the error function between end and end_old
 * @return 0 if converged, -1 and a warning to std::cerr when isnan appears, -2 if failed to reach eps_abs
 */
template< class RHS, class container, unsigned s>
int integrateRK(RHS& rhs, const container& begin, container& end, double T_max, double eps_abs )
{
    RK_classic<s, container > rk( begin); 
    container old_end(begin), temp(begin);
    end = begin;
    if( T_max == 0) return 0;
    double dt = T_max/10;
    int NT = 10;
    double error = 1e10;
 
    while( error > eps_abs && NT < pow( 2, 18) )
    {
        dt /= 2.;
        NT *= 2;
        end = begin;

        int i=0;
        while (i<NT)
        {
            rk( rhs, end, temp, dt); 
            if( !rhs.monitor( temp ) )  //sanity check
            {
                #ifdef DG_DEBUG
                    std::cout << "---------Got sanity error -> choosing smaller step size and redo integration" << " NT "<<NT<<" dt "<<dt<< std::endl;
                #endif
                break;
            }
            end.swap( temp); //end is one step further 
            i++;
        }  
        error = rhs.error( end, old_end);
        old_end = end;
    }
    if( std::isnan( error) )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is NAN! "<<std::endl;
        return -1;
    }
    if( error > eps_abs )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is "<<error<<std::endl;
        return -2;
    }
    return 0;


}

/// @brief Integrates the differential equation using a stage 4 Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
///@copydetails integrateRK()
template< class RHS, class container>
int integrateRK4(RHS& rhs, const container& begin, container& end, double T_max, double eps_abs )
{
    return integrateRK<RHS, container, 4>( rhs, begin, end, T_max, eps_abs);
}

/// @brief Integrates the differential equation using a stage 6 Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
/// @copydetails integrateRK()
template< class RHS, class container>
int integrateRK6(RHS& rhs, const container& begin, container& end, double T_max, double eps_abs )
{
    return integrateRK<RHS, container, 6>( rhs, begin, end, T_max, eps_abs);
}
/// @brief Integrates the differential equation using a stage 17 Runge-Kutta scheme, a rudimentary stepsize-control and monitoring the sanity of integration
///@copydetails integrateRK()
template< class RHS, class container>
int integrateRK17(RHS& rhs, const container& begin, container& end, double T_max, double eps_abs )
{
    return integrateRK<RHS, container, 17>( rhs, begin, end, T_max, eps_abs);
}

///@}
//
///@cond
//Euler specialisation
template < class container>
struct RK<1, container>
{
    RK(){}
    RK( const container& copyable){}
    template < class Functor>
    void operator()( Functor& f, const container& u0, container& u1, double dt)
    {
        f( u0, u1);
        blas1::axpby( 1., u0, dt, u1);
    }
};
///@endcond





} //namespace dg

#endif //_DG_RK_
