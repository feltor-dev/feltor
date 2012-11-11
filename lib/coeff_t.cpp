#include "coeff.h"
#include "matrix.h"
#include <iostream>
#include <complex>

using namespace std;
using namespace toefl;

class CalcComplexCoeff
{
  private:
    typedef std::complex<double> MyComplex;
    MyComplex dd, nu;
    MyComplex a_i, mu_i, tau_i;
    MyComplex kappa_x, kappa_y;
  public:
    CalcComplexCoeff(): dd(1), nu(1), a_i(1), mu_i(1), tau_i(1), kappa_x(1), kappa_y(1){}
    void operator()( QuadMat< std::complex<double>,2>& coeff, const size_t k, const size_t q) const;
};

void CalcComplexCoeff::operator()(QuadMat< MyComplex, 2>& coeff, const size_t k, const size_t q) const
{
    std::complex<double> dx, dy, L, rho_m, Gamma_i, curv;
    dx.real() = 0, dx.imag() = 2*M_PI*(k + 1); //imaginary
    dy.real() = 0, dy.imag() = 2*M_PI*(q + 1); //imaginary
    L = dx*dx + dy*dy; 
    rho_m = a_i*mu_i*L/(1.0-tau_i*mu_i*L);
    Gamma_i = 1.0/(1.0 - 0.5*tau_i*mu_i*L);
    curv = (kappa_x*dx+kappa_y*dy); 
    coeff(0,0) = (curv + dd)*(1./rho_m - 1.) + dy/rho_m + nu*L*L;
    coeff(0,1) = -(curv + dd + dy)*Gamma_i/rho_m;
    coeff(1,0) = (curv + dy)*Gamma_i/rho_m;
    coeff(1,1) = tau_i*curv - (curv + dy)*Gamma_i*Gamma_i/rho_m;
}

class CalcRealCoeff
{
  private:
    double dd, nu;
    double a_i, mu_i, tau_i;
    double kappa_x, kappa_y;
  public:
    CalcRealCoeff(): dd(1), nu(1), a_i(1), mu_i(1), tau_i(1), kappa_x(1), kappa_y(1){}
    void operator()( QuadMat<double ,2>& coeff, const size_t k, const size_t q) const;
};

void CalcRealCoeff::operator()(QuadMat< double, 2>& coeff, const size_t k, const size_t q) const
{
    double dx, dy, L, rho_m, Gamma_i, curv;
    dx = 2*M_PI*(k + 1); //imaginary
    dy = 2*M_PI*(q + 1); //imaginary
    L = dx*dx + dy*dy; 
    rho_m = a_i*mu_i*L/(1.0-tau_i*mu_i*L);
    Gamma_i = 1.0/(1.0 - 0.5*tau_i*mu_i*L);
    curv = (kappa_x*dx+kappa_y*dy); 
    coeff(0,0) = (curv + dd)*(1./rho_m - 1.) + dy/rho_m + nu*L*L;
    coeff(0,1) = -(curv + dd + dy)*Gamma_i/rho_m;
    coeff(1,0) = (curv + dy)*Gamma_i/rho_m;
    coeff(1,1) = tau_i*curv - (curv + dy)*Gamma_i*Gamma_i/rho_m;
}

typedef std::complex<double> MyComplex;

int main()
{
    size_t nx = 4,  ny = 5;
    CalcComplexCoeff f;
    CalcRealCoeff g;
    Matrix< QuadMat<MyComplex,2> > data( nx, ny);
    try{
        Matrix< QuadMat<MyComplex,2> > coeff( nx, ny);
        invert_and_store_coeff< CalcComplexCoeff, MyComplex, 2 > (f, coeff);
        cout << "Test of complex invert\n";
        cout << coeff << endl;
        data = coeff;
        cout << "Test of assignment\n";
        if( data != coeff) 
            cerr << "Assignment failed\n";
        else 
        cout << "Assignment passed\n";
        Matrix< QuadMat<MyComplex,2> > data2( coeff);
        cout << "Test of Copy\n";
        if( data2 != coeff) 
            cerr << "Copy failed!\n";
        else 
            cout << "Copy passed\n";
    }
    catch( Message& m){m.display();}

    cout << "Test of double invert\n";
    Matrix< QuadMat<double,2> > container( nx, ny);
    invert_and_store_coeff< CalcRealCoeff, double, 2> (g, container);
    cout << container << endl;

    return 0;

}

