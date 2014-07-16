#pragma once

#include "mpi_config.h"
#include "grid.cuh"

namespace dg
{

/**
 * @brief 2D MPI Grid class 
 *
 * Represents the local grid coordinates and the process topology
 */
struct MPIGrid2d
{
    MPIGrid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPIComm comm):
        global( x0, x1, y0, y1, n, Nx, Ny), comm( comm){}
    double x0() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() - (g.x1()-g.x0())/(double)g.Nx() + (g.x1()-g.x0())/(double)dims[0]*coords[0]; 
    }
    double x1() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() + (g.x1()-g.x0())/(double)g.Nx() + (g.x1()-g.x0())/(double)dims[0]*(coords[0]+1); 
    }
    double y0() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.y0() - (g.y1()-g.y0())/(double)g.Ny() + (g.y1()-g.y0())/(double)dims[1]*coords[1]; 
    }
    double y1() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.y0() + (g.y1()-g.y0())/(double)g.Ny() + (g.y1()-g.y0())/(double)dims[1]*(coords[1]+1); 
    }
    double lx() const {return x1()-x0();}
    double ly() const {return y1()-y0();}
    double hx() const {return g.hx();}
    double hy() const {return g.hy();}
    unsigned n() const {return g.n();}
    unsigned Nx() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.Nx()/dims[0]+2;
    }
    unsigned Ny() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.Ny()/dims[1]+2;
    }
    bc bcx() const {return g.bcx();}
    bc bcy() const {return g.bcy();}
    const DLT<double>& dlt() const{return g.dlt();}
    /**
     * @brief The total number of points
     *
     * @return n*n*Nx*Ny
     */
    unsigned size() const { return n()*n()*Nx()*Ny();}
    /**
     * @brief Display 
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        g.display();
    }


    private:
    Grid2d<double> g;
    MPI_Comm comm;

};

}//namespace dg
