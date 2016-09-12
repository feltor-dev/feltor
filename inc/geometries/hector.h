#pragma once

#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "orthogonal.h"



namespace hector
{

//container must be compliant in blas1::transfer function
template <class Matrix, class container>
struct Hector
{
    template< class Psi, class PsiX, class PsiY, class PsiXX, class PsiXY, class PsipYY, class LaplacePsiX, class LaplacePsiY>
    Hector( Psi psi, PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, LaplacePsiX laplacePsiX, LaplacePsiY laplacePsiY, double psi0, double psi1, double X0, double Y0, unsigned n, unsigned Nx, unsigned Ny, double eps) : 
        g2d_(psi, psiX, psiY, psiXX, psiXY, psiYY, laplacePsiX, laplacePsiY, psi0, psi1, X0, Y0, n, Nx, Ny), 
        ellipticDIR_(g2d_, dg::DIR, dg::PER, dg::not_normed, dg::centered),
        ellipticNEU_(g2d_, dg::NEU, dg::PER, dg::not_normed, dg::centered)
    {
    }
    double lu() const {return g2d_.lx();}
    double lv() const {return 2.*M_PI;}
    void operator()( const thrust::host_vector<double>& u, 
                     const thrust::host_vector<double>& v, 
                     thrust::host_vector<double>& x, 
                     thrust::host_vector<double>& y, 
                     thrust::host_vector<double>& ux, 
                     thrust::host_vector<double>& uy, 
                     thrust::host_vector<double>& vx, 
                     thrust::host_vector<double>& vy);
    private
    orthogonal::RingGrid2d<container> g2d_;
    dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticDIR_;
    dg::Elliptic<orthogonal::RingGrid2d<container>, Matrix, container> ellipticNEU_;


};

}//namespace hector
