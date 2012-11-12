#ifndef _BLUEPRINT_
#define _BLUEPRINT_

//toefl brauch libraries, um zu funktionieren
//z.B. fftw3 für dfts, cuda für graphikkarten, oder sparse matrix solver 
enum cap{ TL_CUVATURE, TL_COUPLING, TL_IMPURITY, TL_GLOBAL};
enum bc{ TL_PERIODIC, TL_DIRICHLET};
enum method{ TL_KARNIADAKIS};
enum target{ TL_ELECTRONS, TL_IONS, TL_IMPURITIES, TL_POTENTIAL};


struct Physical
{
    private:
    double d, nu;
    double kappa_x, kappa_y;
    double g_e, g_i, g_z;
    double a_i, mu_i, tau_i;
    double a_z, mu_z, tau_z;
    public:
    /*! @brief Construct container with all values set to zero
     */ 
    Physical(){
        d = nu = g_e = g_i = g_z = 0;
        kappa_x = kappa_y = 0;
        a_i = a_z = 0;
        mu_i = mu_z = 0; 
        tau_i = tau_z = 0;
    }
    void setSpecies( enum target t, double a, double mu, double tau, double gradient);
    friend class Equations;
};

struct Boundary
{
    double lx;
    double ly;
};

struct Algorithmic
{
    size_t nx; 
    size_t ny;
};


    
/*! @brief The Setting for the pipeline 
 *
 * The Setting consists of parameters and capacities!
 * With this construction plan you can go to 
 * the pipeline manufacturer who constructs the pipeline. 
 */
class BluePrint
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
    Toefl( const Physical phys, const Boundary bound, const Algorithmic alg): phys(phys), bound(bound), alg(alg)
    {
        curvature = coupling = impurity = global = false; 
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
            default: throw Message( "Unknown Capacity\n");
        }
    }
    bool isEnabled( enum cap capacity)
    {
        switch( capacity)
        {
            case( TL_CURVATURE): return curvature;
            case( TL_COUPLING) : return coupling;
            case( TL_IMPURITY) : return imp;
            case( TL_GLOBAL):    return global;
            default: throw Message( "Unknown Capacity\n");
        }
    }

}

#endif //_BLUEPRINT_
