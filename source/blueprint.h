#ifndef _BLUEPRINT_
#define _BLUEPRINT_

#include <iostream>
#include "message.h"

namespace toefl{
//toefl brauch libraries, um zu funktionieren
//z.B. fftw3 für dfts, cuda für graphikkarten, oder sparse matrix solver 
enum cap{ TL_CURVATURE, TL_COUPLING, TL_IMPURITY, TL_GLOBAL};
enum method{ TL_KARNIADAKIS};
enum target{ TL_ELECTRONS, TL_IONS, TL_IMPURITIES, TL_POTENTIAL};


struct Physical
{
    double d, nu;
    double kappa_x, kappa_y;
    double g_e, g_i, g_z;
    double a_i, mu_i, tau_i;
    double a_z, mu_z, tau_z;
    /*! @brief Construct container with all values set to zero
     */ 
    Physical(){
        d = nu = g_e = g_i = g_z = 0;
        kappa_x = kappa_y = 0;
        a_i = a_z = 0;
        mu_i = mu_z = 0; 
        tau_i = tau_z = 0;
    }
    void display( std::ostream& os = std::cout) const
    {
        os << "Physical parameters are: \n"
            <<"Coupling = "<<d<<"\n"
            <<"viscosity = "<<nu<<"\n"
            <<"Curvature_x = "<<kappa_x<<" Curvature_y = "<<kappa_y<<"\n"
            <<"gradients: g_e ="<<g_e<<" g_i="<<g_i<<" g_z="<<g_z<<"\n"
            <<"Ions       a_i ="<<a_i<<" mu_i="<<mu_i<<" tau_i="<<tau_i<<"\n"
            <<"Impurities a_z ="<<a_z<<" mu_z="<<mu_z<<" tau_z="<<tau_z<<"\n";
    }
};

struct Boundary
{
    double lx;
    double ly;
    enum bc bc_x, bc_y;
    Boundary():lx(0), ly(0), bc_x(TL_PERIODIC), bc_y( TL_PERIODIC){}
    void display( std::ostream& os = std::cout) const
    {
        os << "Boundary parameters are: \n"
            <<" lx="<<lx<<"\n"
            <<" ly="<<ly<<"\n"
            <<"Boundary conditions are ";
        switch(bc_x)
        {
            case(TL_PERIODIC): os << "periodic in x\n";
                               break;
        }
        switch(bc_y)
        {
            case(TL_PERIODIC): os << "periodic in y\n";
                               break;
        }
    }
};

struct Algorithmic
{
    size_t nx; 
    size_t ny;
    double h, dt;
    enum method algorithm;
    Algorithmic():nx(0),ny(0), h(0), dt(0), algorithm(TL_KARNIADAKIS){}
    void display( std::ostream& os = std::cout) const
    {
        os << "Algorithmic parameters are: \n"
            <<"nx="<<nx<<"\n"
            <<"ny="<<ny<<"\n"
            <<"h ="<<h<<"\n"
            <<"dt="<<dt<<"\n";
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
    bool curvature, coupling, imp, global;
  public:
    /*! @brief Init parameters
     *
     * All capacities are disabled by default!
     * @param phys The physical parameters of the equations including numeric viscosity
     */
    Blueprint( const Physical phys, const Boundary bound, const Algorithmic alg): phys(phys), bound(bound), alg(alg)
    {
        curvature = coupling = imp = global = false; 
    }
    const Physical& getPhysical() const {return phys;}
    const Boundary& getBoundary() const {return bound;}
    const Algorithmic& getAlgorithmic() const {return alg;}
    void enable(enum cap capacity)
    {
        switch( capacity)
        {
            case( TL_CURVATURE): curvature = true;
                                 break;
            case( TL_COUPLING) : coupling = true;
                                 break;
            case( TL_IMPURITY) : imp = true;
                                 break;
            case( TL_GLOBAL):    global = true;
                                 break;
            default: throw toefl::Message( "Unknown Capacity\n", ping);
        }
    }
    bool isEnabled( enum cap capacity) const
    {
        switch( capacity)
        {
            case( TL_CURVATURE): return curvature;
            case( TL_COUPLING) : return coupling;
            case( TL_IMPURITY) : return imp;
            case( TL_GLOBAL):    return global;
            default: throw toefl::Message( "Unknown Capacity\n", ping);
        }
    }
    void consistencyCheck() const;
    void display( std::ostream& os = std::cout) const
    {
        phys.display( os);
        bound.display( os);
        alg.display( os);
        os << "Enabled capacities are \n"
            <<"curvature "<< curvature <<"\n"
            <<"coupling  "<<coupling<<"\n"
            <<"imp       "<<imp<<"\n"
            <<"global    "<<global<<"\n";
    }

};

void Blueprint::consistencyCheck() const
{
    //Check algorithm and boundaries
    if( alg.dt <= 0) 
        throw toefl::Message( "dt <= 0!\n", ping);
    if( alg.h - bound.lx/(double)alg.nx > 1e-15) 
        throw toefl::Message( "h != lx/nx\n", ping); 
    if( alg.h - bound.ly/(double)alg.ny > 1e-15) 
        throw toefl::Message( "h != ly/ny\n", ping);
    if( alg.nx == 0||alg.ny == 0) 
        throw toefl::Message( "Set nx and ny!\n", ping);
    //Check physical parameters
    if( curvature && phys.kappa_x == 0 && phys.kappa_y ==0 ) 
        throw toefl::Message( "Curvature enabled but zero!\n", ping);
    if( phys.nu < 0) 
        throw toefl::Message( "nu < 0!\n", ping);
    if( phys.a_i <= 0 || phys.mu_i <= 0 || phys.tau_i < 0) 
        throw toefl::Message( "Ion species badly set\n", ping);
    if( imp && (phys.a_z <= 0 || phys.mu_z <= 0 || phys.tau_z < 0)) 
        throw toefl::Message( "Impuritiy species badly set\n", ping);
    if( phys.a_i + phys.a_z != 1)
        throw toefl::Message( "a_i + a_z != 1\n", ping);
    if( phys.g_i != (phys.g_e - phys.a_z*phys.g_z)/(1.-phys.a_z))
        throw toefl::Message( "g_i is wrong\n", ping);
    if( global) 
        throw toefl::Message( "Global solver not yet implemented\n", ping);
    //Some Warnings
    if( !curvature && (phys.kappa_x != 0 || phys.kappa_y != 0)) 
        std::cerr <<  "TL_WARNING: Curvature disabled but kappa not zero (will be ignored)!\n";
    if( !imp && (phys.a_z != 0 || phys.mu_z != 0 || phys.tau_z != 0)) 
        std::cerr << "TL_WARNING: Impurity disabled but z species not 0 (will be ignored)!\n";
        
}


}

#endif //_BLUEPRINT_
