#ifndef _TL_PARTICLE_DENSITY_
#define _TL_PARTICLE_DENSITY_

#include "toefl/toefl.h"
#include "equations.h"
#include "blueprint.h"

///@addtogroup solvers
///@{
struct ParticleDensity
{
    typedef toefl::Matrix<double, toefl::TL_DFT> Matrix_Type;
    ParticleDensity( const Matrix_Type& copy, const toefl::Blueprint& bp):
        bp( bp),
        rows( copy.rows()), cols( copy.cols()),
        crows( rows), ccols( cols/2+1),
        cdens( crows, ccols ),
        grad_phi( toefl::MatrixArray<double, toefl::TL_DFT,2>::construct( rows, cols)),
        cgrad_phi( toefl::MatrixArray<complex, toefl::TL_NONE, 2>::construct( crows, ccols)),
        dft_dft( rows, cols),
        poisson( bp.physical())
    {}
    void nonlinear( const Matrix_Type& n, const Matrix_Type& phi, Matrix_Type& dens);
    void linear( const Matrix_Type& n, const Matrix_Type& phi, Matrix_Type& dens, unsigned species);
    void laplace( Matrix_Type& phi);
  private:
    void nabla( );
    void gamma( Matrix_Type&);
    typedef std::complex<double> complex;
    toefl::Blueprint bp;
    unsigned rows, cols;
    unsigned crows, ccols;
    toefl::Matrix< complex> cdens;
    std::array< Matrix_Type, 2> grad_phi;
    std::array< toefl::Matrix<complex>, 2> cgrad_phi;
    toefl::DFT_DFT dft_dft;
    toefl::Poisson poisson;
};

void ParticleDensity::nabla()
{
    const toefl::Boundary& bound = bp.boundary();
    dft_dft.r2c( grad_phi[0], cgrad_phi[0]);
    dft_dft.r2c( grad_phi[1], cgrad_phi[1]);
    const complex dymin( 0, 2.*M_PI/bound.ly);
    const complex dxmin( 0, 2.*M_PI/bound.lx);
    double norm = 1./(double)(rows*cols);
    complex dx, dy;
    int ik;
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            ik = (i>rows/2) ? (i-rows) : i; //integer division rounded down
            if( rows%2 == 0 && i == rows/2) ik = 0;
            dx = (double)j*dxmin;
            dy = (double)ik*dymin;
            cgrad_phi[0]( i, j) *= dx*norm;
            cgrad_phi[1]( i, j) *= dy*norm;
        }
    dft_dft.c2r( cgrad_phi[0], grad_phi[0]);
    dft_dft.c2r( cgrad_phi[1], grad_phi[1]);
}

void ParticleDensity::gamma( Matrix_Type& dens)
{
    const toefl::Boundary& bound = bp.boundary();
    dft_dft.r2c( dens, cdens);
    const double kxmin2 = 2.*2.*M_PI*M_PI/(double)(bound.lx*bound.lx),
                 kymin2 = 2.*2.*M_PI*M_PI/(double)(bound.ly*bound.ly);
    double laplace, norm = 1./(double)(rows*cols);
    int ik;
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            ik = (i>rows/2) ? (i-rows) : i; //integer division rounded down
            laplace = - kxmin2*(double)(j*j) - kymin2*(double)(ik*ik);
            cdens(i,j) *= poisson.gamma1_i( laplace)*norm;
        }
    dft_dft.c2r( cdens, dens);
}

void ParticleDensity::nonlinear( const Matrix_Type& n, const Matrix_Type& phi, Matrix_Type& dens)
{
    const toefl::Physical& phys = bp.physical();
    //copy elements for inplace trafo
    grad_phi[0] = grad_phi[1] = phi; 
    nabla();
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols; j++)
        {
            grad_phi[0](i,j)*= phys.mu[1]*n(i,j);
            grad_phi[1](i,j)*= phys.mu[1]*n(i,j);
        }
    nabla();
    dens = n;
    gamma( dens);
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols; j++)
            dens(i,j) = dens(i,j) + grad_phi[0](i,j) + grad_phi[1](i,j);
}

//0 is ions, 1 is impurities 
void ParticleDensity::linear( const Matrix_Type& n, const Matrix_Type& phi, Matrix_Type& dens, unsigned species)
{
    const toefl::Boundary& bound = bp.boundary();
    const toefl::Physical& phys = bp.physical();
    dens = n;
    grad_phi[0] = phi;
    dft_dft.r2c( dens, cdens);
    dft_dft.r2c( grad_phi[0], cgrad_phi[0]);
    const double kxmin2 = 2.*2.*M_PI*M_PI/(double)(bound.lx*bound.lx),
                 kymin2 = 2.*2.*M_PI*M_PI/(double)(bound.ly*bound.ly);
    double laplace, norm = 1./(double)(rows*cols);
    int ik;
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            ik = (i>rows/2) ? (i-rows) : i; //integer division rounded down
            laplace = - kxmin2*(double)(j*j) - kymin2*(double)(ik*ik);
            cdens(i,j) = norm*(poisson.gamma1_i( laplace)*cdens(i,j) + phys.mu[species]*laplace*cgrad_phi[0](i,j));
        }
    dft_dft.c2r( cdens, dens);
}
void ParticleDensity::laplace( Matrix_Type& phi)
{
    const toefl::Boundary& bound = bp.boundary();
    dft_dft.r2c( phi, cgrad_phi[0]);
    const double kxmin2 = 2.*2.*M_PI*M_PI/(double)(bound.lx*bound.lx),
                 kymin2 = 2.*2.*M_PI*M_PI/(double)(bound.ly*bound.ly);
    double laplace, norm = 1./(double)(rows*cols);
    int ik;
    for( unsigned i=0; i<crows; i++)
        for( unsigned j=0; j<ccols; j++)
        {
            ik = (i>rows/2) ? (i-rows) : i; //integer division rounded down
            laplace = - kxmin2*(double)(j*j) - kymin2*(double)(ik*ik);
            cgrad_phi[0](i,j) *= norm*laplace;
        }
    dft_dft.c2r( cgrad_phi[0], phi);
}
///@}


#endif //_TL_PARTICLE_DENSITY_
