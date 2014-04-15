#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "toefl/timer.h"
#include "file/read_input.h"
#include "file/file.h"
//#include "utility.h"
#include "particle_density.h"
#include "dft_dft_solver.h"

#include "energetics.h"
//#include "drt_dft_solver.h"
#include "blueprint.h"

using namespace std;
using namespace toefl;
typedef std::complex<double> Complex;

const unsigned n = 3;
typedef DFT_DFT_Solver<n> Sol;
typedef Sol::Matrix_Type Mat;
    
unsigned itstp, itstp2; //initialized by init function
unsigned max_out;
double amp, imp_amp; //
double blob_width, posX, posY;

void write_probe( const Mat& field, std::vector<double>* total, std::vector<double>* fluct)
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
            total[k*8+l].push_back(field( posY, posX ));
            fluct[k*8+l].push_back(field( posY, posX ) - average[l]);
        }
    }
}
void write_vx( const Mat& phi, std::vector<double>* v, double h)
{
    unsigned nx = phi.cols();
    unsigned ny = phi.rows();
    for( unsigned k=0; k<8; k++)
    {
        unsigned posY = ny/16+ny*k/8;
        for( unsigned l=0; l<8;l++)
        {
            unsigned posX = nx/16+nx*l/8;
            v[k*8+l].push_back( -(phi( posY +1, posX ) - phi( posY-1, posX))/2./h);//-dy phi
        }
    }
}
void write_vy( const Mat& phi, std::vector<double>* vy, std::vector<double>* vy_fluc, double h)
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
            double dxphi = (phi( posY, posX+1 )-phi(posY, posX-1))/2./h;
            vy[k*8+l].push_back(   dxphi    );
            vy_fluc[k*8+l].push_back(  dxphi - average[l]);
        }
    }
}

void write_probe_to_file( file::Probe& probe, const char* name, const std::vector<double>* data, unsigned size, unsigned offset)
{
    probe.openGroup( name);
    for( unsigned i=0; i<8; i++)
        for( unsigned j=0; j<8; j++)
            probe.writeSubset( data[i*8+j], i+1, j+1, size, offset );
    probe.closeGroup();
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
    itstp2 = para[25];
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
    if( argc != 4)
    {
        cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [fields.h5] [probe.h5] \n";
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
        cerr << "ERROR: Only periodic boundaries allowed!\n";
        return -1;
    }
    if( !bp.isEnabled( TL_IMPURITY) )
    {
        cerr << "ERROR: Only allowed with impurities!\n";
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
    //double meanMassE = integral( ne, alg.h)/bound.lx/bound.ly;
    //std::cout << setprecision(6) <<meanMassE<<std::endl;
    
    Energetics<n> energetics(bp);
    ////////////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    //std::ofstream  os( argv[4]);
    //os << "#Time(1) Ue(2) Ui(3) Uj(4) Ei(5) Ej(6) M(Ei)(7) M(Ej)(8) F_e(9) F_i(10) F_j(11) R_i(12) R_j(13) Diff(14) M(Diff)(15) A(16) J(17)\n";
    //os << std::setprecision(14);
    double time = 0.0;
    std::vector<double> probe_array( 64), probe_fluct( 64);
    std::vector<double> average(8,0);
    std::vector<double> out( alg.nx*alg.ny);
    std::vector<double> output[n+1] = {out, out, out, out};
    toefl::Timer t, t2;
    t.tic();
    file::Probe probe( argv[3], input, max_out*itstp/itstp2);
          probe.createSet( "ne", 8, 8);
          probe.createSet( "ne_fluc", 8, 8);
          probe.createSet( "ni", 8, 8);
          probe.createSet( "ni_fluc", 8, 8);
          probe.createSet( "nj", 8, 8);
          probe.createSet( "nj_fluc", 8, 8);
          probe.createSet( "phi", 8, 8);
          probe.createSet( "phi_fluc", 8, 8);
          probe.createSet( "vx", 8, 8);
          probe.createSet( "vy", 8, 8);
          probe.createSet( "vy_fluc", 8, 8);
          probe.createDataSet( "time");
          probe.createDataSet( "Ue");
          probe.createDataSet( "Ui");
          probe.createDataSet( "Uj");
          probe.createDataSet( "Ei");
          probe.createDataSet( "Ej");
          probe.createDataSet( "Mi");
          probe.createDataSet( "Mj");
          probe.createDataSet( "Fe");
          probe.createDataSet( "Fi");
          probe.createDataSet( "Fj");
          probe.createDataSet( "Ri");
          probe.createDataSet( "Rj");
          probe.createDataSet( "Diff");
          probe.createDataSet( "MDiff");
          probe.createDataSet( "A");
          probe.createDataSet( "J");

    for( unsigned i=0; i<max_out; i++)
    {
        std::vector<double> times, ue, ui, uj, ei, ej, mi, mj, fe, fi, fj, ri, rj, diff, mdiff, capital_a, capital_jot;
        std::vector<double> probe_ne[64], probe_ne_fluc[64];
        std::vector<double> probe_ni[64], probe_ni_fluc[64];
        std::vector<double> probe_nz[64], probe_nz_fluc[64];
        std::vector<double> probe_phi[64], probe_phi_fluc[64];
        std::vector<double> probe_vy[64], probe_vy_fluc[64];
        std::vector<double> probe_vx[64]; 
        output[0] = solver.getField( TL_ELECTRONS).copy();
        //xpa( output[0], meanMassE);
        output[1] = solver.getField( TL_IONS).copy();
        output[2] = solver.getField( TL_IMPURITIES).copy();
        output[3] = solver.getField( TL_POTENTIAL).copy();
        t5file.write( output[0], output[1], output[2],output[3], time, alg.nx, alg.ny);
        t2.tic();
        for( unsigned j=0; j<itstp; j++)
        {
            if( !(j%itstp2))
            {
                times.push_back(time);
                const Mat& electrons = solver.getField( TL_ELECTRONS);
                write_probe( electrons, probe_ne, probe_ne_fluc);
                const Mat& ions = solver.getField( TL_IONS);
                write_probe( ions, probe_ni, probe_ni_fluc);
                const Mat& imp = solver.getField( TL_IMPURITIES);
                write_probe( imp, probe_nz, probe_nz_fluc);
                const Mat& potential = solver.getField( TL_POTENTIAL);
                write_probe( potential, probe_phi, probe_phi_fluc);
                write_vx( potential, probe_vx, alg.h);
                write_vy( potential, probe_vy, probe_vy_fluc, alg.h);
                std::vector<double> thermal = energetics.thermal_energies( solver.getDensity());
                ue.push_back( thermal[0]), ui.push_back( thermal[1]), uj.push_back( thermal[2]);
                std::vector<double> exb = energetics.exb_energies( solver.getField(TL_POTENTIAL));
                ei.push_back( exb[0]), ej.push_back( exb[1]), mi.push_back( exb[2]), mj.push_back( exb[3]);
                std::vector<double> gradient_flux = energetics.gradient_flux( solver.getDensity(), solver.getPotential() );
                fe.push_back( gradient_flux[0]), fi.push_back( gradient_flux[1]), fj.push_back( gradient_flux[2]), ri.push_back( gradient_flux[3]), rj.push_back( gradient_flux[4]);
                std::vector<double> diffusion = energetics.diffusion( solver.getDensity(), solver.getPotential() );
                diff.push_back( diffusion[0]), mdiff.push_back( diffusion[1]);
                capital_a.push_back( energetics.capital_a( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL)));
                capital_jot.push_back( energetics.capital_jot( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL)));

            }
            if( i==0 && j==0)
                solver.first_step();
            else if( i==0 && j==1)
                solver.second_step();
            else
                solver.step();
            time += alg.dt;
        }
        //write Probe file
        probe.writeSubset( times,"time", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( ue, "Ue", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( ui, "Ui", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( uj, "Uj", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( ei, "Ei", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( ej, "Ej", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( mi, "Mi", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( mj, "Mj", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( fe, "Fe", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( fi, "Fi", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( fj, "Fj", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( ri, "Ri", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( rj, "Rj", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( diff, "Diff", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( mdiff, "MDiff", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( capital_a, "A", itstp/itstp2, i*itstp/itstp2);
        probe.writeSubset( capital_jot, "J", itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "ne", probe_ne, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "ni", probe_ni, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "nj", probe_nz, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "ne_fluc", probe_ne_fluc, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "ni_fluc", probe_ni_fluc, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "nj_fluc", probe_nz_fluc, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "phi", probe_phi, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "phi_fluc", probe_phi_fluc, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "vx", probe_vx, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "vy", probe_vy, itstp/itstp2, i*itstp/itstp2);
        write_probe_to_file( probe, "vy_fluc", probe_vy_fluc, itstp/itstp2, i*itstp/itstp2);

        t2.toc();
        std::cout << "\n\t Time "<<time <<" / "<<alg.dt*itstp*max_out;
        std::cout << "\n\t Average time for one step: "<<t2.diff()/(double)itstp<<"s\n\n"<<std::flush;
    }
    output[0] = solver.getField( TL_ELECTRONS).copy();
    //xpa( output[0], meanMassE);
    output[1] = solver.getField( TL_IONS).copy();
    output[2] = solver.getField( TL_IMPURITIES).copy();
    output[3] = solver.getField( TL_POTENTIAL).copy();
    t5file.write( output[0], output[1], output[2], output[3], time, alg.nx, alg.ny);

    /*
    //times.push_back(time);
    //const Mat& electrons = solver.getField( TL_ELECTRONS);
    //write_probe( electrons, probe_ne, probe_ne_fluc);
    //const Mat& ions = solver.getField( TL_IONS);
    //write_probe( ions, probe_ni, probe_ni_fluc);
    //const Mat& imp = solver.getField( TL_IMPURITIES);
    //write_probe( imp, probe_nz, probe_nz_fluc);
    //const Mat& potential = solver.getField( TL_POTENTIAL);
    //write_probe( potential, probe_phi, probe_phi_fluc);
    //write_vx( potential, probe_vx, alg.h);
    //write_vy( potential, probe_vy, probe_vy_fluc, alg.h);
    //{
    //os << time<<" ";
    //std::vector<double> thermal = energetics.thermal_energies( solver.getDensity());
    //std::vector<double> exb = energetics.exb_energies( solver.getField(TL_POTENTIAL));
    //std::vector<double> gradient_flux = energetics.gradient_flux( solver.getDensity(), solver.getPotential() );
    //std::vector<double> diffusion = energetics.diffusion( solver.getDensity(), solver.getPotential() );
    //double capital_a = energetics.capital_a( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL));
    //double capital_jot = energetics.capital_jot( solver.getField( TL_ELECTRONS), solver.getField(TL_POTENTIAL));

    //for( unsigned k=0; k<thermal.size(); k++)
    //    os << thermal[k]<<" ";
    //for( unsigned k=0; k<exb.size(); k++)
    //    os << exb[k]<<" ";
    //for( unsigned k=0; k<gradient_flux.size(); k++)
    //    os << gradient_flux[k]<<" ";
    //for( unsigned k=0; k<diffusion.size(); k++)
    //    os << diffusion[k]<<" ";
    //os << capital_a<<" ";
    //os << capital_jot;
    //os << std::endl;
    //}
    */

    //////////////////////////////////////////////////////////////////
    //os.close();
    fftw_cleanup();
    t.toc();
    std::cout << "Total simulation time for "<<max_out*itstp<<" steps: "<<t.diff()<<"s\n";
    std::cout << "Which is "<<t.diff()/(double)(max_out*itstp)<<"s/step\n";
    //std::cout << "Times size: "<<times.size()<<"\n";
    //std::cout << "Probes size: "<<probe_ne[0].size()<<"\n";
    //for( unsigned i=0; i<probe_ne[16].size(); i++)
    //    std::cout << probe_ne[16][i]<<"\n";
    return 0;

}
