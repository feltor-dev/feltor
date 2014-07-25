#ifndef _BLUEPRINT_
#define _BLUEPRINT_

#include <iostream>
#include <cmath>
#include "toefl/ghostmatrix.h" // holds boundary conditions
#include "toefl/message.h"

namespace toefl{
/*! @addtogroup parameters
 * @{
 */
/*! @brief Possible capacities of a toefl solver
 */
enum cap{   TL_IMPURITY, //!< Include impurities
            TL_GLOBAL, //!< Solve global equations
            TL_MHW //!< Modify parallel term in electron density equation
};

/*! @brief Possible targets for memory buffer
 */
enum target{ 
    TL_ELECTRONS, //!< Electron density
    TL_IONS, //!< Ion density
    TL_IMPURITIES, //!< Impurity density
    TL_POTENTIAL, //!< Potential
    TL_ALL //!< all buffers
};


/*! @brief Holds the physical parameters of the problem.
 *
 * @note This is an aggregate and thus you can use initializer lists
 */
struct Physical
{
    double d;  //!< The coupling constant
    double nu; //!< The artificial viscosity
    double kappa; //!< The curvature in y-direction
    double g_e; //!< The background gradient for electrons
    double g[2]; //!< The background gradient for ions 0 and impurities 1
    double a[2]; //!< Charge of ions 0 and impurities 1
    double mu[2]; //!< The mass of ions 0 and impurities 1
    double tau[2]; //!< temperature of ions 0 and impurities 1
    /*! @brief This is a POD
     */ 
    Physical() = default;
    /*! @brief Print Physical parameters to outstream
     *
     * @param os The outstream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Physical parameters are: \n"
            <<"    Coupling:        = "<<d<<"\n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"   Species/Parameter   g\ta\tmu\ttau\n"
            <<"    Electrons:         "<<g_e  <<"\t"<<"-1"<<"\t"<<"0"<<"\t"<<"1\n"
            <<"    Ions:              "<<g[0] <<"\t"<<a[0]<<"\t"<<mu[0]<<"\t"<<tau[0]<<"\n"
            <<"    Impurities:        "<<g[1] <<"\t"<<a[1]<<"\t"<<mu[1]<<"\t"<<tau[1]<<"\n";
    }
};

/*! @brief Describes the boundary and the boundary conditions of the problem.
 *
 * @note This is an aggregate and thus you can use initializer lists
 */
struct Boundary
{
    double lx; //!< Physical extension of x-direction
    double ly; //!< Physical extension of y-direction
    enum bc bc_x;  //!< Boundary condition in x (y is always periodic)
    Boundary() = default;
    /*! @brief Print Boundary parameters to outstream
     *
     * @param os The outstream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n"
            <<"Boundary conditions in x are: \n";
        switch(bc_x)
        {
            case(TL_PERIODIC): os << "    PERIODIC \n"; break;
            case(   TL_DST00): os << "    DST-1-like \n"; break;
            case(   TL_DST10): os << "    DST-2-like \n"; break;
            case(   TL_DST01): os << "    DST-3-like \n"; break;
            case(   TL_DST11): os << "    DST-4-like \n"; break;
            default: os << "    Not specified!!\n"; 
        }
    }
};

/*! @brief Describes the algorithmic (notably discretization) issues of the solver.
 *
 * @note This is an aggregate and thus you can use initializer lists
 */
struct Algorithmic
{
    size_t nx;  //!< # of gridpoints in x
    size_t ny;  //!< # of gridpoints in y
    double h;  //!< ly/ny (Only quadratic grid elements are usable.)
    double dt; //!< The time step
    Algorithmic() = default;
    /*! @brief Print Algorithmic parameters to outstream
     *
     * @param os The outstream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Algorithmic parameters are: \n"
            <<"    nx = "<<nx<<"\n"
            <<"    ny = "<<ny<<"\n"
            <<"    h  = "<<h<<"\n"
            <<"    dt = "<<dt<<"\n";
    }
};


/*! @brief The Setting for the pipeline 
 *
 * The Setting consists of parameters and capacities!
 * With this construction plan you can go to 
 * the pipeline manufacturer who constructs the pipeline. 
 * It is recommended to call 
 * \code
 *  try{ blueprint.consistencyCheck();}
 *  catch( toefl::Message& m){m.display();}
 *  \endcode
 * before constructing a Pipeline to catch any Messages before construction.
 */
class Blueprint
{
    Physical phys;
    Boundary bound;
    Algorithmic alg;
    bool imp, global, mhw;
  public:
    /*! @brief Construct empty blueprint
     */
    Blueprint():imp(false), global(false), mhw(false){}
    /*! @brief Init parameters
     *
     * All capacities are disabled by default!
     * @param phys The physical parameters of the equations including numeric viscosity
     * @param bound The parameters describing boundary and boundary conditions
     * @param alg The parameters describing algorithmic issues
     */
    Blueprint( const Physical& phys, const Boundary& bound, const Algorithmic& alg): phys(phys), bound(bound), alg(alg)
    {
        imp = global = mhw = false; 
    }

    Blueprint( const std::vector<double>& para)
    {
        imp = global = mhw = false;
        alg.nx = para[1];
        alg.ny = para[2];
        alg.dt = para[3];

        bound.ly = para[4];
        switch( (unsigned)para[5])
        {
            case( 0): bound.bc_x = TL_PERIODIC; break;
            case( 1): bound.bc_x = TL_DST10; break;
            case( 2): bound.bc_x = TL_DST01; break;
        }
        if( para[6])
            mhw = true;

        phys.d = para[7];
        phys.nu = para[8];
        phys.kappa = para[9];

        phys.g_e = phys.g[0] = para[11];
        phys.tau[0] = para[12];
        if( para[13])
        {
            imp = true;
            //imp_amp = para[14];
            phys.g[1] = para[15];
            phys.a[1] = para[16];
            phys.mu[1] = para[17];
            phys.tau[1] = para[18];
        }
        else 
            phys.g[1] = phys.a[1] = phys.mu[1] = phys.tau[1] = 0;

        phys.a[0] = 1. -phys.a[1];
        phys.g[0] = (phys.g_e - phys.a[1] * phys.g[1])/(1.-phys.a[1]);
        phys.mu[0] = 1.0;//single charged ions

        //N = para[19];
        //omp_set_num_threads( para[20]);
        //blob_width = para[21];
        //std::cout<< "With "<<omp_get_max_threads()<<" threads\n";

        alg.h = bound.ly / (double)alg.ny;
        bound.lx = (double)alg.nx * alg.h;
    }

    /*! @brief Get Physical 
     */
    const Physical& physical() const {return phys;}
    /*! @brief Get Boundary 
     */
    const Boundary& boundary() const {return bound;}
    /*! @brief Get Algorithmic 
     */
    const Algorithmic& algorithmic() const {return alg;}
    /*! @brief Set Physical 
     */
    Physical& physical() {return phys;}
    /*! @brief Set Boundary 
     */
    Boundary& boundary() {return bound;}
    /*! @brief Set Algorithmic 
     */
    Algorithmic& algorithmic() {return alg;}
    /*! @brief Enable a capacity 
     *
     * @param capacity Capacity to be enabled
     */
    void enable(enum cap capacity)
    {
        switch( capacity)
        {
            case( TL_IMPURITY) : imp = true;     break;
            case( TL_GLOBAL):    global = true;  break;
            case( TL_MHW):       mhw = true;     break;
            default: throw Message( "Unknown Capacity\n", _ping_); //is this necessary?
        }
    }
    /*! @brief Check if a capacity is enabled
     *
     * @param capacity Capacity to check for
     */
    bool isEnabled( enum cap capacity) const
    {
        switch( capacity)
        {
            case( TL_IMPURITY) : return imp;
            case( TL_GLOBAL):    return global;
            case( TL_MHW):       return mhw;
            default: throw Message( "Unknown Capacity\n", _ping_);
        }
    }
    /*! @brief Perform several consistency checks on the set of parameters
     *
     */
    void consistencyCheck() const;
    /*! @brief Print all parameters to an outstream
     *
     * @param os The outstream
     */
    void display( std::ostream& os = std::cout) const
    {
        phys.display( os);
        bound.display( os);
        alg.display( os);
        char enabled[] = "ENABLED", disabled[] = "DISABLED";

        os << "Impurities are: \n"
            <<"    "<<(imp?enabled:disabled)<<"\n"
            //<<"Global solvers are: \n"
            //<<"    "<<(global?enabled:disabled)<<"\n"
            <<"Modified Hasegawa Wakatani: \n"
            <<"    "<<(mhw?enabled:disabled)<<"\n";
    }

};
///@}
void Blueprint::consistencyCheck() const
{
    //Check algorithm and boundaries
    if( alg.dt <= 0) 
        throw Message( "dt <= 0!\n", _ping_);
    if( fabs( alg.h - bound.lx/(double)alg.nx) > 1e-15) 
        throw Message( "h != lx/nx\n", _ping_); 
    if( fabs( alg.h - bound.ly/(double)alg.ny) > 1e-15) 
        throw Message( "h != ly/ny\n", _ping_);
    if( alg.nx == 0||alg.ny == 0) 
        throw Message( "Set nx and ny!\n", _ping_);
    //Check physical parameters
    if( phys.nu < 0) 
        throw Message( "nu < 0!\n", _ping_);
    if( phys.a[0] <= 0 || phys.mu[0] <= 0 || phys.tau[0] < 0) 
        throw Message( "Ion species badly set\n", _ping_);
    if( imp && (phys.a[1] < 0 || phys.mu[1] <= 0 || phys.tau[1] < 0)) 
        throw Message( "Impuritiy species badly set\n", _ping_);
    if( fabs(phys.a[0] + phys.a[1] - 1) > 1e-15)
        throw Message( "a_i + a_z != 1 (background not neutral)\n", _ping_);
    if( fabs( phys.g_e - phys.a[0]*phys.g[0]- phys.a[1]*phys.g[1]) > 1e-15)
        throw Message( "Background is not neutral! \n", _ping_);
    //inconsistency when impurities are not set??
    if( !imp && (phys.a[1] != 0 || phys.mu[1] != 0 || phys.tau[1] != 0)) 
        throw Message( "Impurity disabled but z species not 0!\n", _ping_);
    //Some Warnings
    if( global && (phys.g_e != 0||phys.g[0] != 0||phys.g[1] != 0))
        std::cerr << "TL_WARNING: Global solver ignores gradients\n";
        
}



} //namespace toefl

#endif //_BLUEPRINT_
