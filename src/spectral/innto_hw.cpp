#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "file/file.h"
//#include "utility.h"
#include "particle_density.h"
#include "dft_dft_solver.h"
//#include "drt_dft_solver.h"
#include "blueprint.h"
/*
 * Same as innblobs but outputs results in hdf5 - file
 */

using namespace std;
using namespace toefl;
typedef std::complex<double> Complex;

const unsigned n = 3;
typedef DFT_DFT_Solver<n> Sol;
typedef typename Sol::Matrix_Type Mat;
    
unsigned itstp; //initialized by init function
unsigned max_out;
double amp, imp_amp; //
double blob_width, posX, posY;
unsigned energy_interval;

void write_probe( const Mat& field, std::vector<double>& total, std::vector<double>& fluct)
{
    unsigned nx = field.cols();
    unsigned ny = field.rows();
    std::vector<double> average(8,0);
    for( unsigned l=0; l<8;l++)
    {
        unsigned posX = nx/16+nx*l/8;
        for( unsigned i=0; i<ny; i++)
            average[l] += field( i, posX);
        average[l] /= ny;
    }
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            total[k*8+l] = field( posY, posX );
            fluct[k*8+l] = field( posY, posX ) - average[l];
        }
    }
}
void write_vx( const Mat& phi, std::vector<double>& v, double h)
{
    unsigned nx = phi.cols();
    unsigned ny = phi.rows();
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            v[k*8+l] = -(phi( posY +1, posX ) - phi( posY-1, posX))/2./h;//-dy phi
        }
    }
}
void write_vy( const Mat& phi, std::vector<double>& vy, std::vector<double>& vy_fluc, double h)
{
    unsigned nx = phi.cols();
    unsigned ny = phi.rows();
    std::vector<double> average(24,0);
    for( unsigned l=0; l<8;l++)
    {
        unsigned posX = nx/16+nx*l/8;
        for( unsigned j=0; j<ny; j++)
            average[l] += (phi( j, posX+1)-phi(j, posX-1))/2./h; //dx phi
        average[l] /= ny;
    }
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            vy[k*8+l]       = (phi( posY, posX+1 )-phi(posY, posX-1))/2./h;
            vy_fluc[k*8+l]  = vy[k*8+l] - average[l];
        }
    }
}

void dy( const Matrix<Complex>& in, Matrix<Complex >& m, double ly, double norm)
{
    unsigned crows = m.rows(), ccols = m.cols();
    const Complex dymin( 0, 2.*M_PI/ly);

    // dft_dft is not transposing so i is the y index by default
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            unsigned ik = (i>crows/2) ? (i-crows) : i;
            m(i,j) = (double)ik/norm*dymin*in(i,j);
        }
}
void remove_average_y( const Matrix<Complex>& in, Matrix<Complex>& m)
{
    m = in;
    for( unsigned j=0; j<m.cols(); j++)
        m(0,j) = 0;
}
void dx( const Matrix<Complex>& in, Matrix<Complex >& m, double lx, double norm)
{
    unsigned crows = m.rows(), ccols = m.cols();
    const Complex dxmin( 0, 2.*M_PI/lx);

    // dft_dft is not transposing so i is the y index by default
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            m(i,j) = (double)j/norm*dxmin*in(i,j);
        }
}


void dxx( const Matrix<Complex>& in, Matrix<Complex >& m, double lx, double norm)
{
    unsigned crows = m.rows(), ccols = m.cols();
    const Complex dxmin( 0, 2.*M_PI/lx);
    const double kxmin2 = 2.*2.*M_PI*M_PI/(double)(lx*lx);

    // dft_dft is not transposing so i is the y index by default
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            m(i,j) = -kxmin2*(double)(j*j)/norm*in(i,j);
        }
}

Blueprint read( char const * file)
{
    std::cout << "Reading from "<<file<<"\n";
    std::vector<double> para;
    try{ para = file::read_input( file); }
    catch (Message& m) 
    {  
        m.display(); 
        throw m;
    }
    Blueprint bp( para);
    amp = para[10];
    imp_amp = para[14];
    itstp = para[19];
    omp_set_num_threads( para[20]);
    blob_width = para[21];
    max_out = para[22];
    posX = para[23];
    posY = para[24];
    energy_interval = para[25];
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}

double integral( const Mat& src, double h)
{
    double sum=0;
    for( unsigned i=0; i<src.rows(); i++)
        for( unsigned j=0; j<src.cols(); j++)
            sum+=h*h*src(i,j);
    return sum;
}

void xpa( std::vector<double>& x, double a)
{
    for( unsigned i =0; i<x.size(); i++)
        x[i] += a;
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    Blueprint bp_mod;
    std::vector<double> v;
    std::string input;
    if( argc != 5)
    {
        cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [fields.h5] [probe.h5] [energies.dat]\n";
        return -1;
    }
    else 
    {
        bp_mod = read(argv[1]);
        input = file::read_file( argv[1]);
    }
    const Blueprint bp = bp_mod;
    bp.display( );
    if( bp.boundary().bc_x != TL_PERIODIC)
    {
        cerr << "Only periodic boundaries allowed!\n";
        return -1;
    }
    if( !bp.isEnabled( TL_IMPURITY) )
    {
        cerr << "Only allowed with impurities!\n";
        return -1;
    }
    //construct solvers 
    try{ Sol solver( bp); }catch( Message& m){m.display();}
    Sol solver (bp);
    unsigned rows = bp.algorithmic().ny, cols = bp.algorithmic().nx;
    DFT_DFT dft_dft(rows, cols);
    unsigned crows = rows, ccols = cols/2+1;
    Matrix<Complex > cphi(crows, ccols), cne(cphi), cni( cne), cnz( cne);

    const Algorithmic& alg = bp.algorithmic();
    Mat ne{ alg.ny, alg.nx, 0.}, ni(ne), nz(ne), phi{ ne};
    const Boundary& bound = bp.boundary();
    // init ne and nz! 
    try{
        init_gaussian( ne, posX, posY, blob_width/bound.lx, blob_width/bound.ly, amp);
        init_gaussian_column( nz, posX, blob_width/bound.lx, imp_amp);
        std::array< Mat, n> arr{{ ne, nz, phi}};
        //now set the field to be computed
        solver.init( arr, TL_IONS);
    }catch( Message& m){m.display();}
    double meanMassE = integral( ne, alg.h)/bound.lx/bound.ly;

    /////////////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    file::Probe probe( argv[3], input);
    std::ofstream  os( argv[4]);
    double time = 3.*alg.dt;
    std::vector<double> probe_array( 64), probe_fluct( 64);
    std::vector<double> average(8,0);
    std::vector<double> out( alg.nx*alg.ny);
    std::vector<double> output[n+1] = {out, out, out, out};
    for( unsigned i=0; i<max_out; i++)
    {
        output[0] = solver.getField( TL_ELECTRONS).copy();
        xpa( output[0], meanMassE);
        output[1] = solver.getField( TL_IONS).copy();
        output[2] = solver.getField( TL_IMPURITIES).copy();
        output[3] = solver.getField( TL_POTENTIAL).copy();
        t5file.write( output[0], output[1], output[2],output[3], time, alg.nx, alg.ny);
        for( unsigned j=0; j<itstp; j++)
        {
            probe.createGroup( time);
            const Mat& electrons = solver.getField( TL_ELECTRONS);
            write_probe( electrons, probe_array, probe_fluct);
            probe.write( probe_array, "ne", 8,8);
            probe.write( probe_fluct, "ne_fluc", 8,8);
            const Mat& ions = solver.getField( TL_IONS);
            write_probe( ions, probe_array, probe_fluct);
            probe.write( probe_array, "ni", 8,8);
            probe.write( probe_fluct, "ni_fluc", 8,8);
            const Mat& imp = solver.getField( TL_IMPURITIES);
            write_probe( imp, probe_array, probe_fluct);
            probe.write( probe_array, "nz", 8,8);
            probe.write( probe_fluct, "nz_fluc", 8,8);
            const Mat& potential = solver.getField( TL_POTENTIAL);
            write_probe( potential, probe_array, probe_fluct);
            probe.write( probe_array, "phi", 8,8);
            probe.write( probe_fluct, "phi_fluc", 8,8);
            write_vx( potential, probe_array, alg.h);
            probe.write( probe_array, "vx", 8,8);
            write_vy( potential, probe_array, probe_fluct, alg.h);
            probe.write( probe_array, "vy", 8,8);
            probe.write( probe_fluct, "vy_fluc", 8,8);
            probe.closeGroup();
            if( !(j%energy_interval))
            {
                os << time << " ";
                ne = solver.getField( TL_ELECTRONS);
                phi = solver.getField( TL_POTENTIAL);
                dft_dft.r2c( ne, cne);
                dft_dft.r2c( phi, cphi);
                dy( cne, cne, bound.ly,1./rows/cols);

                os << dft_dft.dot( cphi, cne)/alg.nx/alg.nx/alg.ny/alg.ny<< " ";
                os<< std::endl;
            }
            solver.step();
            time += alg.dt;
        }
    }
    output[0] = solver.getField( TL_ELECTRONS).copy();
    xpa( output[0], meanMassE);
    output[1] = solver.getField( TL_IONS).copy();
    output[2] = solver.getField( TL_IMPURITIES).copy();
    output[3] = solver.getField( TL_POTENTIAL).copy();
    t5file.write( output[0], output[1], output[2], output[3], time, alg.nx, alg.ny);
            probe.createGroup( time);
            const Mat& electrons = solver.getField( TL_ELECTRONS);
            write_probe( electrons, probe_array, probe_fluct);
            probe.write( probe_array, "ne", 8,8);
            probe.write( probe_fluct, "ne_fluc", 8,8);
            const Mat& ions = solver.getField( TL_IONS);
            write_probe( ions, probe_array, probe_fluct);
            probe.write( probe_array, "ni", 8,8);
            probe.write( probe_fluct, "ni_fluc", 8,8);
            const Mat& imp = solver.getField( TL_IMPURITIES);
            write_probe( imp, probe_array, probe_fluct);
            probe.write( probe_array, "nz", 8,8);
            probe.write( probe_fluct, "nz_fluc", 8,8);
            const Mat& potential = solver.getField( TL_POTENTIAL);
            write_probe( potential, probe_array, probe_fluct);
            probe.write( probe_array, "phi", 8,8);
            probe.write( probe_fluct, "phi_fluc", 8,8);
            write_vx( potential, probe_array, alg.h);
            probe.write( probe_array, "vx", 8,8);
            write_vy( potential, probe_array, probe_fluct, alg.h);
            probe.write( probe_array, "vy", 8,8);
            probe.write( probe_fluct, "vy_fluc", 8,8);
            probe.closeGroup();
    //////////////////////////////////////////////////////////////////
    os.close();
    fftw_cleanup();
    return 0;

}
