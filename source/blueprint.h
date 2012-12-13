#ifndef _BLUEPRINT_
#define _BLUEPRINT_

#include <iostream>
#include <cmath>
#include "../lib/ghostmatrix.h" // holds boundary conditions
#include "../lib/message.h"

namespace toefl{
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
    TL_POTENTIAL //!< Potential
};


/*! @brief Holds the physical parameters of the problem.
 *
 * @note This is an aggregate and thus you can use initializer lists
 */
struct Physical
{
    double d;  //!< The coupling constant
    double nu; //!< The artificial viscosity
    double g_e; //!< The background gradient for electrons
    double kappa; //!< The curvature in y-direction
    double g[2]; //!< The background gradient for ions 0 and impurities 1
    double a[2]; //!< Charge of ions 0 and impurities 1
    double mu[2]; //!< The mass of ions 0 and impurities 1
    double tau[2]; //!< temperature of ions 0 and impurities 1
    /*! @brief This is a POD
     */ 
    Physical() = default;
    void display( std::ostream& os = std::cout) const
    {
        os << "Physical parameters are: \n"
            <<"    Coupling:        = "<<d<<"\n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"   Species/Parameter   g      a     mu    tau\n"
            <<"    Electrons:         "<<g_e  <<"     "<<"-1"<<"      "<<"0"  <<"      "<<"1\n"
            <<"    Ions:              "<<g[0] <<"      "<<a[0]<<"      "<<mu[0]<<"      "<<tau[0]<<"\n"
            <<"    Impurities:        "<<g[1] <<"      "<<a[1]<<"      "<<mu[1]<<"      "<<tau[1]<<"\n";
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
    const Physical& physical() const {return phys;}
    const Boundary& boundary() const {return bound;}
    const Algorithmic& algorithmic() const {return alg;}
    Physical& physical() {return phys;}
    Boundary& boundary() {return bound;}
    Algorithmic& algorithmic() {return alg;}
    void enable(enum cap capacity)
    {
        switch( capacity)
        {
            case( TL_IMPURITY) : imp = true;     break;
            case( TL_GLOBAL):    global = true;  break;
            case( TL_MHW):       mhw = true;     break;
            default: throw Message( "Unknown Capacity\n", ping); //is this necessary?
        }
    }
    bool isEnabled( enum cap capacity) const
    {
        switch( capacity)
        {
            case( TL_IMPURITY) : return imp;
            case( TL_GLOBAL):    return global;
            case( TL_MHW):       return mhw;
            default: throw Message( "Unknown Capacity\n", ping);
        }
    }
    void consistencyCheck() const;
    void display( std::ostream& os = std::cout) const
    {
        phys.display( os);
        bound.display( os);
        alg.display( os);
        char enabled[] = "ENABLED", disabled[] = "DISABLED";

        os << "Impurities are: \n"
            <<"    "<<(imp?enabled:disabled)<<"\n"
            <<"Global solvers are: \n"
            <<"    "<<(global?enabled:disabled)<<"\n"
            <<"Modified Hasegawa Wakatani: \n"
            <<"    "<<(mhw?enabled:disabled)<<"\n";
    }

};

void Blueprint::consistencyCheck() const
{
    //Check algorithm and boundaries
    if( alg.dt <= 0) 
        throw Message( "dt <= 0!\n", ping);
    if( fabs( alg.h - bound.lx/(double)alg.nx) > 1e-15) 
        throw Message( "h != lx/nx\n", ping); 
    if( fabs( alg.h - bound.ly/(double)alg.ny) > 1e-15) 
        throw Message( "h != ly/ny\n", ping);
    if( alg.nx == 0||alg.ny == 0) 
        throw Message( "Set nx and ny!\n", ping);
    //Check physical parameters
    if( phys.nu < 0) 
        throw Message( "nu < 0!\n", ping);
    if( phys.a[0] <= 0 || phys.mu[0] <= 0 || phys.tau[0] < 0) 
        throw Message( "Ion species badly set\n", ping);
    if( imp && (phys.a[1] <= 0 || phys.mu[1] <= 0 || phys.tau[1] < 0)) 
        throw Message( "Impuritiy species badly set\n", ping);
    if( fabs(phys.a[0] + phys.a[1] - 1) > 1e-15)
        throw Message( "a_i + a_z != 1\n", ping);
    if( fabs( phys.g[0] - (phys.g_e - phys.a[1]*phys.g[1])/(1.-phys.a[1])) > 1e-15)
        throw Message( "g_i is wrong\n", ping);
    //Some Warnings
    if( !imp && (phys.a[1] != 0 || phys.mu[1] != 0 || phys.tau[1] != 0)) 
        std::cerr << "TL_WARNING: Impurity disabled but z species not 0 (will be ignored)!\n";
    if( global && (phys.g_e != 0||phys.g[0] != 0||phys.g[1] != 0))
        std::cerr << "TL_WARNING: Global solver ignores gradients\n";
        
}


} //namespace toefl

#endif //_BLUEPRINT_
