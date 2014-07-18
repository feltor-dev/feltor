#pragma once

#include "../enums.h"
#include "mpi_config.h"
#include "grid.h"

namespace dg
{

/**
 * @brief 2D MPI Grid class 
 *
 * Represents the local grid coordinates and the process topology
 */
struct MPI_Grid2d
{
    MPI_Grid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm):
        g( x0, x1, y0, y1, n, Nx, Ny), comm( comm)
    {
        int rank, dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank == 0)
        {
            if(Nx%dims[0]!=0)
                std::cerr << "Nx "<<Nx<<" dims[0] "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if(Ny%dims[0]!=0)
                std::cerr << "Nx "<<Nx<<" dims[0] "<<dims[0]<<std::endl;
            assert( Ny%dims[1] == 0);
        }
    }
    MPI_Grid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
        g( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy), comm( comm)
    {
        int rank, dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank == 0)
        {
            if(Nx%dims[0]!=0)
                std::cerr << "Nx "<<Nx<<" dims[0] "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if(Ny%dims[0]!=0)
                std::cerr << "Ny "<<Ny<<" dims[1] "<<dims[1]<<std::endl;
            assert( Ny%dims[1] == 0);
        }
    }

    double x0() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() - g.hx() + g.lx()/(double)dims[0]*(double)coords[0]; 
    }
    double x1() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() + g.hx() + g.lx()/(double)dims[0]*(double)(coords[0]+1); 
    }
    double y0() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.y0() - g.hy() + g.ly()/(double)dims[1]*(double)coords[1]; 
    }
    double y1() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.y0() + g.hy() + g.ly()/(double)dims[1]*(double)(coords[1]+1); 
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
    MPI_Comm communicator() const{return comm;}
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
        os << "GLOBAL GRID \n";
        g.display();
        os << "LOCAL GRID \n";

        Grid2d<double> grid = local();
        grid.display();

    }
    Grid2d<double> local() const {return Grid2d<double>(x0(), x1(), y0(), y1(), n(), Nx(), Ny(), bcx(), bcy());}
    Grid2d<double> global() const {return g;}


    private:
    Grid2d<double> g; //global grid
    MPI_Comm comm; //just an integer...

};

}//namespace dg
