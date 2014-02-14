#pragma once
#include "polarisation.cuh"
#include "cg.cuh"


typedef thrust::host_vector<double> dg_container;

struct dg_workspace
{
    dg_workspace( const dg::Grid<double>& g):size(g.size()), pol(g), cg( thrust::host_vector<double>(g.size()) , g.size() ),
    w2d( dg::create::w2d(g)), v2d( dg::create::v2d(g))
                      {}
    unsigned size;
    dg::Polarisation2dX<dg_container > pol; 
    dg::CG<dg_container > cg;
    const dg_container w2d, v2d;
    cusp::csr_matrix<int, double, cusp::host_memory> matrix; //!< most efficient host matrix format
};


dg_workspace* dg_create_workspace( unsigned Nx, unsigned Ny, double hx, double hy, dg::bc bcx, dg::bc bcy)
{
    dg::Grid<double> g( 0, Nx*hx, 0, Ny*hy, 1, Nx, Ny, bcx, bcy); 
    dg_workspace* w = new dg_workspace( g);
    return w;
}

void dg_update_polarizability( dg_workspace* w,  const double* chi)
{
    dg_container chi_( chi, chi+w->size); //newly allocate 
    w->matrix = w->pol.create( chi_);
}

unsigned dg_solve( dg_workspace* w, double *x, const double* b, double eps)
{
    dg_container x_( x, x+w->size);
    dg_container b_( b, b+w->size);
    dg::blas2::symv( w->w2d, b_, b_);
    return w->cg( w->matrix, x_, b_, w->v2d, eps);
}


void dg_free_workspace( dg_workspace* w)
{
    delete w;
}
