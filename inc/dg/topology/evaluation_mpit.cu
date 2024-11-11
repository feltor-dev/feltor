#include <iostream>
#include <iomanip>
#include <cmath>

#include <mpi.h>
#include "dg/functors.h"
#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "mpi_evaluation.h"
#include "mpi_weights.h"


template<class T>
T function(T x, T y)
{
    T rho = 0.20943951023931953; //pi/15
    T delta = 0.050000000000000003;
    if( y<= M_PI)
        return delta*cos(x) - 1./rho/cosh( (y-M_PI/2.)/rho)/cosh( (y-M_PI/2.)/rho);
    return delta*cos(x) + 1./rho/cosh( (3.*M_PI/2.-y)/rho)/cosh( (3.*M_PI/2.-y)/rho);
}
double function3d( double x, double y, double z)
{
    return exp(x)*exp(y)*exp(z);
}

int main(int argc, char** argv)
{
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "This program tests the exblas::dot function. The tests succeed only if the evaluation and grid functions but also the weights and especially the exblas::dot function are correctly implemented and compiled. Furthermore, the compiler implementation of the exp function in the math library must be consistent across platforms to get reproducible results.\n";
    if(rank==0)std::cout << "A TEST is PASSED if the number in the second column shows EXACTLY 0!\n";
    unsigned n = 3, Nx = 12, Ny = 28, Nz = 100;
    if(rank==0)std::cout << "On Grid "<<n<<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<"\n";
    MPI_Comm comm1d, comm2d, comm3d;
    mpi_init1d( dg::PER, comm1d);
    dg::MPIGrid1d g1d( 1, 2, n, 12, dg::PER, comm1d);

    mpi_init2d( dg::PER, dg::PER, comm2d);
    dg::MPIGrid2d g2d( 0.0, 6.2831853071795862, 0.0, 6.2831853071795862, 3, 48, 48, dg::PER, dg::PER, comm2d);
    //dg::MPIGrid2d g2d( {0.0, 6.2831853071795862, 3, 48}, {0.0, 6.2831853071795862, 5, 28}, comm2d);
    dg::RealMPIGrid<float,2> gf2d( 0.0, 6.2831853071795862, 0.0, 6.2831853071795862, 3, 48, 48, dg::PER, dg::PER, comm2d);
    mpi_init3d( dg::PER, dg::PER, dg::PER, comm3d);
    dg::MPIGrid3d g3d( 1, 2, 3, 4, 5, 6, n, Nx, Ny, Nz, dg::PER, dg::PER, dg::PER, comm3d);
    //dg::MPIGrid3d g3d( {1, 2, n, Nx,},{ 3, 4, 7, Ny},{ 5, 6, 4, Nx}, comm3d);

    //test evaluation and expand functions
    dg::MDVec func1d = dg::construct<dg::MDVec>(dg::evaluate( exp, g1d));
    dg::MDVec func2d = dg::construct<dg::MDVec>(dg::evaluate( function<double>, g2d));
    dg::fMDVec funcf2d = dg::construct<dg::fMDVec>(dg::evaluate( function<float>, gf2d));
    dg::MDVec func3d = dg::construct<dg::MDVec>(dg::evaluate( function3d, g3d));
    //test weights
    const dg::MDVec w1d = dg::construct<dg::MDVec>( dg::create::weights(g1d));
    const dg::MDVec w2d = dg::construct<dg::MDVec>(dg::create::weights(g2d));
    const dg::fMDVec wf2d = dg::construct<dg::fMDVec>(dg::create::weights(gf2d));
    const dg::MDVec w3d = dg::construct<dg::MDVec>(dg::create::weights(g3d));
    dg::exblas::udouble res;

    double integral = dg::blas1::dot( w1d, func1d); res.d = integral;
    if(rank==0)std::cout << "1D integral               "<<std::setw(6)<<integral <<"\t" << res.i - 4616944842743393935  << "\n";
    double sol = (exp(2.) -exp(1));
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    if(rank==0)std::cout << "Relative 1d error is      "<<(integral-sol)/sol<<"\n\n";

    double integral2d = dg::blas1::dot( w2d, func2d); res.d = integral2d;
    if(rank==0)std::cout << "2D integral               "<<std::setw(6)<<integral2d <<"\t" << res.i + 4823280491526356992<< "\n";
    double sol2d = 0.;
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    if(rank==0)std::cout << "2d error is               "<<(integral2d-sol2d)<<"\n\n";
    float integralf2d = dg::blas1::dot( wf2d, funcf2d); res.d = integralf2d;
    if(rank==0)std::cout << "2D integral (float)       "<<std::setw(6)<<integralf2d <<"\n";
    float solf2d = 0.;
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<solf2d<<std::endl;
    if(rank==0)std::cout << "2d error (float)          "<<(integralf2d-solf2d)<<"\n\n";

    double integral3d = dg::blas1::dot( w3d, func3d); res.d = integral3d;
    if(rank==0)std::cout << "3D integral               "<<std::setw(6)<<integral3d <<"\t" << res.i - 4675882723962622631<< "\n";
    double sol3d = (exp(2.)-exp(1))*(exp(4.)-exp(3))*(exp(6.)-exp(5));
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol3d<<std::endl;
    if(rank==0)std::cout << "Relative 3d error is      "<<(integral3d-sol3d)/sol3d<<"\n\n";

    double norm2d = dg::blas2::dot( w2d, func2d); res.d = norm2d;
    if(rank==0)std::cout << "Square normalized 2D norm "<<std::setw(6)<<norm2d<<"\t" << res.i - 4635333359953759707<<"\n";
    double solution2d = 80.0489;
    if(rank==0)std::cout << "Correct square norm is    "<<std::setw(6)<<solution2d<<std::endl;
    if(rank==0)std::cout << "Relative 2d error is      "<<(norm2d-solution2d)/solution2d<<"\n\n";

    double norm3d = dg::blas2::dot( func3d, w3d, func3d); res.d = norm3d;
    if(rank==0)std::cout << "Square normalized 3D norm "<<std::setw(6)<<norm3d<<"\t" << res.i - 4746764681002108278<<"\n";
    double solution3d = (exp(4.)-exp(2))/2.*(exp(8.)-exp(6.))/2.*(exp(12.)-exp(10))/2.;
    if(rank==0)std::cout << "Correct square norm is    "<<std::setw(6)<<solution3d<<std::endl;
    if(rank==0)std::cout << "Relative 3d error is      "<<(norm3d-solution3d)/solution3d<<"\n";
    if(rank==0)std::cout << "TEST if dot throws on Inf or Nan:\n";
    dg::MDVec x = dg::evaluate( dg::CONSTANT( 6.12610567450009658), g2d);
    dg::blas1::transform( x, x, sin );
    dg::blas1::transform( x,x, dg::LN<double>());
    bool hasnan = dg::blas1::reduce( x, false,
            thrust::logical_or<bool>(), dg::ISNFINITE<double>());
    if(rank==0)std::cout << "x contains Inf or Nan numbers "<<std::boolalpha<<hasnan<<"\n";
    try{
        dg::blas1::dot( x,x);
    }catch ( std::exception& e)
    {
        if(rank==0)std::cerr << "Error thrown as expected\n";
        //std::cerr << e.what() << std::endl;
    }
    if(rank==0)std::cout << "COMPLEX SCALAR PRODUCTS\n";
    thrust::device_vector<thrust::complex<double>> tmp( func3d.data().size());
    dg::MPI_Vector<thrust::device_vector<thrust::complex<double>>> cc3d( tmp, func3d.communicator());
    dg::blas1::transform( func3d, cc3d, []DG_DEVICE(double x){ return thrust::complex<double>{x,x};});
    thrust::complex<double> cintegral = dg::blas1::dot( w3d, cc3d);
    res.d =cintegral.real();
    if(rank==0)std::cout << "3D integral (real)        "<<std::setw(6)<<cintegral.real() <<"\t" << res.i - 4675882723962622631<< "\n";
    res.d =cintegral.imag();
    if(rank==0)std::cout << "3D integral (imag)        "<<std::setw(6)<<cintegral.imag() <<"\t" << res.i - 4675882723962622631<< "\n";
    sol2d = 0;
    if(rank==0)std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    if(rank==0)std::cout << "3d error is               "<<(cintegral.real()-sol2d)<<"\n\n";
    tmp.resize( func1d.data().size());
    dg::MPI_Vector<thrust::device_vector<thrust::complex<double>>> cc1d( tmp, func1d.communicator());
    dg::blas1::transform( func1d, cc1d, []DG_DEVICE(double x){ return thrust::complex<double>{x,x};});
    cintegral = dg::blas1::dot( w1d, cc1d);
    res.d =cintegral.real();
    std::cout << "1D integral (real)        "<<std::setw(6)<<cintegral.real() <<"\t" << res.i - 4616944842743393935 << "\n";
    res.d =cintegral.imag();
    std::cout << "1D integral (imag)        "<<std::setw(6)<<cintegral.imag() <<"\t" << res.i - 4616944842743393935 << "\n";
    res.d = integral;
    sol = (exp(2.) -exp(1));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    std::cout << "Relative 1d error is      "<<(cintegral.real()-sol)/sol<<"\n\n";

    if(rank==0)std::cout << "\nFINISHED! Continue with topology/derivatives_mpit.cu !\n\n";

    MPI_Finalize();
    return 0;
}
