
/*
 * Implements probe diagnostics on a DG grid
 *
 */


#ifndef PROBES_H
#define PROBES_H


#include "dg/algorithm.h"
#include <fstream>
#include <sstream>


/**
 * @brief Class that takes care of probe output
 * 
 * Upon instantiation, generate N probesm positioned in equidistant radial
 * positions at the vertical center of the simulation domain:
 *
 * x_probe = n * p.lx / num_probes, y_probe = p.ly / 2 , n = 0 .. num_probes
 *
 * Creates files probe_[0-9][0-9][0-9].dat in which to write output
 *
 */
template<class IMatrix, class Matrix, class container = thrust::device_vector<double> >
struct probes
{
    public:
        template<class Grid2d>
        probes(container, container, const Grid2d&);
        // Write time series of electron dcensity, electric potental, and radial particle flux
        void fluxes(double, container&, container&);
        // Write radial electron density and potential profile
        void profiles(double, container&, container&);

    private:
        const unsigned Nx;                                  // Number of points in the grid, radial direction
        const unsigned Ny;                                  // Number of points in the grid, poloidal direction
        const container x_coords;                           // Radial position of the probes
        const container y_coords;                           // Poloidal position of the probes
        const size_t num_probes;                            // Number of probes
        IMatrix probe_interp;                                // Interpolates field (nem phi) on the probe positions
        Matrix dy;                                          // Derivative matrix
        dg::Average<container> pol_avg;  // Poloidal Average operator
        std::vector<std::string> fnames;                    // Vector of file names for probe output
};


/* 
 * Create interpolation matrixm derivation matrix
 * Create log files
 */
template<class IMatrix, class Matrix, class container>
template<class Grid2d>
probes<IMatrix, Matrix, container> :: probes (container x_c, container y_c, const Grid2d& g) :
    Nx(g.Nx()),
    Ny(g.Ny()),
    x_coords(x_c),
    y_coords(y_c),
    num_probes(x_coords.size()),
    dy(dg::create::dy(g)),
    pol_avg(g, dg::coo2d::y)
{ 
    thrust::host_vector<double> t1, t2;
    dg::assign( x_c, t1);
    dg::assign( y_c, t2);
    dg::blas2::transfer( dg::create::interpolation( t1, t2, g, dg::NEU), probe_interp);
    assert(x_coords.size () == y_coords.size());
    std::ofstream of;
    std::stringstream fn;

    /* Create datafiles for probe data */
    for(unsigned n = 0; n < num_probes; n++)
    {
        fn << "probe_" << std::setfill('0') << std::setw(3) << n << ".dat";
        fnames.push_back(fn.str());

        of.open(fn.str().data(), std::ios::trunc);
        of << x_coords[n] << "\t" << y_coords[n] << "\n";
        of.close();

        fn.str(std::string(""));
    }

    /* Create datafiles for radial profiles */
    of.open("ne_prof.dat");
    of.close();

    of.open("phi_prof.dat");
    of.close();
}



/*
 * Compute radial profiles of electron density and electric potential
 *
 * Write current profiles to ne_prof.dat, phi_prof.dat
 *
 */
template<class IMatrix, class Matrix, class container>
void probes<IMatrix, Matrix, container> :: profiles(double time, container& npe, container& phi)
{
    std::cout << "Computing profiles " << std::endl;

    static container prof_phi(Nx);
    static container prof_ne(Nx);

    std::ofstream of_ne;
    std::ofstream of_phi;

    pol_avg(phi, prof_phi,false);
    pol_avg(npe, prof_ne,false);

    of_ne.open("ne_prof.dat", std::ios::trunc);
    of_phi.open("phi_prof.dat", std::ios::trunc);
    of_ne << time << "\n";
    of_phi << time << "\n";

    for(unsigned n = 0; n < Nx; n++)
    {
        of_ne << "\t" << prof_ne[n];
        of_phi << "\t" << prof_phi[n];
    }
    of_ne.close();
    of_phi.close();
}


/* 
 * Write out particle density, electric potential, radial particle flux at the probe sites
 * In FELTOR:
 * npe[0] = ne, npe[1] = Ni
 * phi[0] = The real phi, phi[1]: Generalized potential
 */
template<class IMatrix, class Matrix, class container>
void probes<IMatrix, Matrix, container> :: fluxes(double time, container& npe, container& phi)
{
    double ip_gamma_n = 0.0;
    static container phi_y(phi);
    static container ip_n(num_probes);
    static container ip_phi(num_probes);
    static container ip_phi_y(num_probes);


    // Compute phi_y
    dg::blas2::gemv(dy, phi, phi_y);

    // Get ne at interpolation values
    dg::blas2::gemv(probe_interp, npe, ip_n);
    // Get phi at interpolation values
    dg::blas2::gemv(probe_interp, phi, ip_phi);
    // Compute radial flux
    dg::blas2::gemv(probe_interp, phi_y, ip_phi_y);

    std::ofstream of;

    for(unsigned n = 0; n < num_probes; n++)
    {
        // Compute radial flux on-the-fly
        ip_gamma_n = -1.0 * ip_n[n] * ip_phi_y[n];

        of.open(fnames[n].data(), std::ios::app);
        of << std::setw(20) << std::setprecision(16);
        of << time << "\t";
        of << ip_n[n] << "\t";
        of << ip_phi[n] << "\t";
        of << ip_gamma_n;
        of << std::endl;
        of.close();
    }

}

#endif // PROBES_H
