#ifndef _BLUEPRINT_
#define _BLUEPRINT_

#include <iostream>
#include <cmath>
#include "../lib/ghostmatrix.h" // holds boundary conditions
#include "../lib/message.h"

namespace toefl{
//toefl brauch libraries, um zu funktionieren
//z.B. fftw3 für dfts, cuda für graphikkarten, oder sparse matrix solver 
enum cap{ TL_IMPURITY, TL_GLOBAL};
enum target{ TL_ELECTRONS, TL_IONS, TL_IMPURITIES, TL_POTENTIAL};


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
            <<"    Electrons:   g_e = "<<g_e<<"\n"
            <<"    Ions:        g_i = "<<g[0] <<", a_i = "<<a[0]<<", mu_i = "<<mu[0]<<", tau_i = "<<tau[0]<<"\n"
            <<"    Impurities:  g_z = "<<g[1] <<", a_z = "<<a[1]<<", mu_z = "<<mu[1]<<", tau_z = "<<tau[1]<<"\n";
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
            case(   TL_DST00): os << "    DST 1 like \n"; break;
            case(   TL_DST10): os << "    DST 2 like \n"; break;
            case(   TL_DST01): os << "    DST 3 like \n"; break;
            case(   TL_DST11): os << "    DST 4 like \n"; break;
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
    const Physical phys;
    const Boundary bound;
    const Algorithmic alg;
    bool imp, global;
  public:
    /*! @brief Init parameters
     *
     * All capacities are disabled by default!
     * @param phys The physical parameters of the equations including numeric viscosity
     */
    Blueprint( const Physical& phys, const Boundary& bound, const Algorithmic& alg): phys(phys), bound(bound), alg(alg)
    {
        imp = global = false; 
    }
    const Physical& getPhysical() const {return phys;}
    const Boundary& getBoundary() const {return bound;}
    const Algorithmic& getAlgorithmic() const {return alg;}
    void enable(enum cap capacity)
    {
        switch( capacity)
        {
            case( TL_IMPURITY) : imp = true;
                                 break;
            case( TL_GLOBAL):    global = true;
                                 break;
            default: throw Message( "Unknown Capacity\n", ping);
        }
    }
    bool isEnabled( enum cap capacity) const
    {
        switch( capacity)
        {
            case( TL_IMPURITY) : return imp;
            case( TL_GLOBAL):    return global;
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
            <<"    "<<(global?enabled:disabled)<<"\n";
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
    if( global) 
        throw Message( "Global solver not yet implemented\n", ping);
    //Some Warnings
    if( !imp && (phys.a[1] != 0 || phys.mu[1] != 0 || phys.tau[1] != 0)) 
        std::cerr << "TL_WARNING: Impurity disabled but z species not 0 (will be ignored)!\n";
        
}


} //namespace toefl

#endif //_BLUEPRINT_
