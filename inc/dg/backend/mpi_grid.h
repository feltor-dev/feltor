#pragma once

#include <cmath>
#include "../enums.h"
//#include "mpi_config.h"
#include "grid.h"

namespace dg
{

/**
 * @brief 2D MPI Grid class 
 *
 * Represents the local grid coordinates and the process topology. The 
 * grids of different processes overlap in the x- and y- coordinate. 
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
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if(Ny%dims[1]!=0)
                std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
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
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if(Ny%dims[1]!=0)
                std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
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

/**
 * @brief 3D MPI Grid class 
 *
 * Represents the local grid coordinates and the process topology. Note
 * that the grids of different processes overlap in the x- and y- coordinate but 
 * not in the z-coordinate.
 */
struct MPI_Grid3d
{
    MPI_Grid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm):
        g( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz), comm( comm)
    {
        int rank, dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank == 0)
        {
            if(!(Nx%dims[0]==0))
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if( !(Ny%dims[1]==0))
                std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
            assert( Ny%dims[1] == 0);
            if( !(Nz%dims[2]==0))
                std::cerr << "Nz "<<Nz<<" npz "<<dims[2]<<std::endl;
            assert( Nz%dims[2] == 0);
        }
    }
    MPI_Grid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        g( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz), comm( comm)
    {
        int rank, dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank == 0)
        {
            if(!(Nx%dims[0]==0))
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if( !(Ny%dims[1]==0))
                std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
            assert( Ny%dims[1] == 0);
            if( !(Nz%dims[2]==0))
                std::cerr << "Nz "<<Nz<<" npz "<<dims[2]<<std::endl;
            assert( Nz%dims[2] == 0);
        }
    }

    double x0() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.x0() - g.hx() + g.lx()/(double)dims[0]*(double)coords[0]; 
    }
    double x1() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.x0() + g.hx() + g.lx()/(double)dims[0]*(double)(coords[0]+1); 
    }
    double y0() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.y0() - g.hy() + g.ly()/(double)dims[1]*(double)coords[1]; 
    }
    double y1() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.y0() + g.hy() + g.ly()/(double)dims[1]*(double)(coords[1]+1); 
    }
    double z0() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.z0() + g.lz()/(double)dims[2]*(double)coords[2]; 
    }
    double z1() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.z0() + g.lz()/(double)dims[2]*(double)(coords[2]+1); 
    }
    double lx() const {return x1()-x0();}
    double ly() const {return y1()-y0();}
    double lz() const {return z1()-z0();}
    double hx() const {return g.hx();}
    double hy() const {return g.hy();}
    double hz() const {return g.hz();}
    unsigned n() const {return g.n();}
    unsigned Nx() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.Nx()/dims[0]+2;
    }
    unsigned Ny() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.Ny()/dims[1]+2;
    }
    unsigned Nz() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.Nz()/dims[2];
    }
    bc bcx() const {return g.bcx();}
    bc bcy() const {return g.bcy();}
    bc bcz() const {return g.bcz();}
    MPI_Comm communicator() const{return comm;}
    const DLT<double>& dlt() const{return g.dlt();}
    /**
     * @brief The total number of points
     *
     * @return n*n*Nx*Ny
     */
    unsigned size() const { return n()*n()*Nx()*Ny()*Nz();}
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

        Grid3d<double> grid = local();
        grid.display();

    }
    /**
     * @brief Return a grid local for the calling process
     *
     * The local grid returns the same values for Nx(), Ny(), ... as the grid
     * class itself
     * @return Grid object
     */
    Grid3d<double> local() const {return Grid3d<double>(x0(), x1(), y0(), y1(), z0(), z1(), n(), Nx(), Ny(), Nz(), bcx(), bcy(), bcz());}
    Grid3d<double> global() const {return g;}
    /**
     * @brief Returns the pid of the process that holds the grid surrounding a given pint
     *
     * @param x X-coord
     * @param y Y-coord
     * @param z Z-coord
     *
     * @return pid of a process, or -1 if non of the grids matches
     */
    int pidOf( double x, double y, double z) const;
    private:
    Grid3d<double> g; //global grid
    MPI_Comm comm; //just an integer...
};

int MPI_Grid3d::pidOf( double x, double y, double z) const
{
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    //std::cout << x << " "<< y<<" "<<z<<" "<<g.hx()<<" "<<g.hy()<<" "<<g.hz()<<"\n";
    if( x < g.x0() && x > g.x0() - g.hx()) x += g.hx();
    if( x > g.x1() && x < g.x1() + g.hx()) x -= g.hx();
    if( y < g.y0() && y > g.y0() - g.hy()) y += g.hy();
    if( y > g.y1() && y < g.y1() + g.hy()) y -= g.hy();
    coords[0] = (unsigned)floor( (x-g.x0())/g.lx()*(double)dims[0] );
    coords[1] = (unsigned)floor( (y-g.y0())/g.ly()*(double)dims[1] );
    coords[2] = (unsigned)floor( (z-g.z0())/g.lz()*(double)dims[2] );
    int rank;
    if( MPI_Cart_rank( comm, coords, &rank) == MPI_SUCCESS ) 
        return rank;
    else
        return -1;
}

}//namespace dg
