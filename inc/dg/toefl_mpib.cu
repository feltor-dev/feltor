#include <iostream>
#include <iomanip>
#include <vector>
#include <mpi.h>


#include "toefl.cuh"
#include "functors.h"

#include "backend/mpi_init.h"

#include "netcdf_par.h"
#include "file/nc_utilities.h"


unsigned n = 3;
unsigned Nx = 100;
unsigned Ny = 20;
const double lx = 5.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e5;

const unsigned k = 3;
const double dt = 2e-7;
const unsigned N = 5; //steps between output

double eps = 1e-3;


double groundState( double x, double y) { return ly/2. - y;}
/**
 * @brief Functor returning a gaussian
 * \f[
   f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
   \f]
 */
struct Gaussian
{
    /**
     * @brief Functor returning a gaussian
     *
     * @param x0 x-center-coordinate
     * @param y0 y-center-coordinate
     * @param sigma_x x - variance
     * @param sigma_y y - variance 
     * @param amp Amplitude
     */
    Gaussian( float x0, float y0, float sigma_x, float sigma_y, float amp)
        : x00(x0), y00(y0), sigma_x(sigma_x), sigma_y(sigma_y), amplitude(amp){}
    /**
     * @brief Return the value of the gaussian
     *
     * \f[
       f(x,y) = Ae^{-(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}} 
       \f]
     * @param x x - coordinate
     * @param y y - coordinate
     *
     * @return gaussian
     */
    float operator()(float x, float y)
    {
        return  amplitude*
                   exp( -((x-x00)*(x-x00)/2./sigma_x/sigma_x +
                          (y-y00)*(y-y00)/2./sigma_y/sigma_y) );
    }
  private:
    float  x00, y00, sigma_x, sigma_y, amplitude;

};

int main( int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    MPI_Comm comm;
    mpi_init2d( dg::PER, dg::DIR, n, Nx, Ny, comm);
    int rank;
    MPI_Comm_rank( comm, &rank);
    ///////////////////////////////////////////////////////////////////////
    //create initial vector
    const dg::MPI_Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::DIR, comm);
    dg::Gaussian gaussian( 1., ly/2., .1, .1, 1);
    dg::MHVec theta = dg::evaluate ( gaussian, grid);
    std::vector<dg::MHVec> y0(2, theta);
    y0[1] = dg::MHVec( dg::evaluate(dg::zero, grid) ); //omega is zero

    //create RHS and AB
    dg::Toefl< dg::CartesianMPIGrid2d, dg::MHMatrix, dg::MHVec> test( grid, Ra, Pr, eps); 
    dg::AB< k, std::vector<dg::MHVec> > ab( y0);
    ab.init( test, y0, dt);

    int ncid;
    MPI_Info info;
    MPI_Info_create( &info);
    file::NC_Error_Handle h;
    h = nc_create_par( "toefl.nc", NC_NETCDF4|NC_MPIIO|NC_CLOBBER, comm, info, &ncid) ;
    int dimIDs[3], tvarID;
    h = file::define_dimensions( ncid, dimIDs, &tvarID, grid.global());
    int dataID;
    h = nc_def_var( ncid, "temperature", NC_DOUBLE, 3, dimIDs, &dataID);
    h = nc_var_par_access(ncid, dataID, NC_COLLECTIVE);
    h = nc_var_par_access(ncid, tvarID, NC_COLLECTIVE);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( comm, 2, dims, periods, coords);
    size_t count[3] = {1, grid.n()*(grid.Ny()-2), grid.n()*(grid.Nx()-2)};
    size_t start[3] = {0, coords[1]*count[1], coords[0]*count[2]};
    dg::MHVec ground = dg::evaluate( groundState, grid), temperature( ground);
    double time = 0;
    size_t tcount = 1;
    if(rank==0) std::cout << "Writing file toefl.nc...\n";
    for( unsigned i=0; i<1000; i++)
    {
        //compute the total temperature
        dg::blas1::axpby( 1., y0[0],  0., temperature);
        dg::blas1::axpby( 1., ground, 1., temperature);
        thrust::host_vector<double> reduc = temperature.data();
        h = nc_put_vara_double( ncid, dataID, start, count, reduc.data());
        h = nc_put_vara_double( ncid, tvarID, &start[0], &tcount, &time);
        start[0]++;

        //step 
        for( unsigned k=0; k<N; k++)
        {
            ab( test, y0);
            time += (double)N*dt;
        } 
    }
    if(rank==0) std::cout << "Done!\n";

    h = nc_close( ncid);
    MPI_Finalize();
    return 0;

}

