#pragma once

#include <cmath>
#include "../enums.h"
#include "grid.h"
/*! @file 
  
  MPI Grid objects
  */

namespace dg
{
///@addtogroup mpi_structures
///@{

/**
 * @brief 2D MPI Grid class 
 *
 * Represents the local grid coordinates including overlap. 
 * The grids of different processes overlap in the x- and y- coordinate. 
 * This helps when computing derivatives, but introduces additional 
 * bookkeeping in everything else. Recommended to change in future 
 * releases. 
 *
 * For now one is faced with three grids in mpi computations: The global grid, which holds the global boundaries and number of grid cells, the local grid (with overlap) which is the global grid divided by the # of processes plus the ghostcells, and the local grid without ghostcells.
 */
struct MPI_Grid2d
{
    /**
     * @brief Construct mpi grid
     *
     * @param x0
     * @param x1
     * @param y0
     * @param y1
     * @param n
     * @param Nx
     * @param Ny
     * @param comm
     */
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

    /**
     * @brief Construct mpi grid
     *
     * @param x0
     * @param x1
     * @param y0
     * @param y1
     * @param n
     * @param Nx
     * @param Ny
     * @param bcx
     * @param bcy
     * @param comm
     */
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

    /**
     * @brief Return local x0
     *
     * The local value is shifted by hx 
     * @return local left boundary
     */
    double x0() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() - g.hx() + g.lx()/(double)dims[0]*(double)coords[0]; 
    }

    /**
     * @brief Return local x1
     *
     * The local value is shifted by hx 
     * @return local right boundary
     */
    double x1() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.x0() + g.hx() + g.lx()/(double)dims[0]*(double)(coords[0]+1); 
    }

    /**
     * @brief Return local y0
     *
     * The local value is shifted by hy 
     * @return local left boundary
     */
    double y0() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.y0() - g.hy() + g.ly()/(double)dims[1]*(double)coords[1]; 
    }

    /**
     * @brief Return local y1
     *
     * The local value is shifted by hy 
     * @return local right boundary
     */
    double y1() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.y0() + g.hy() + g.ly()/(double)dims[1]*(double)(coords[1]+1); 
    }

    /**
     * @brief Return local lx
     *
     * @return local length
     */
    double lx() const {return x1()-x0();}

    /**
     * @brief Return local ly
     *
     * @return local length
     */
    double ly() const {return y1()-y0();}

    /**
     * @brief Return local hx
     *
     * @return local grid constant
     */
    double hx() const {return g.hx();}

    /**
     * @brief Return local hy
     *
     * @return local grid constant
     */
    double hy() const {return g.hy();}

    /**
     * @brief Return n
     *
     * @return number of polynomial coefficients
     */
    unsigned n() const {return g.n();}

    /**
     * @brief Return the local number of cells 
     *
     * Not the one given in the constructor
     * @return number of cells
     */
    unsigned Nx() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.Nx()/dims[0]+2;
    }

    /**
     * @brief Return the local number of cells 
     *
     * Not the one given in the constructor
     * @return number of cells
     */
    unsigned Ny() const {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        return g.Ny()/dims[1]+2;
    }

    /**
     * @brief global x boundary
     *
     * @return boundary condition
     */
    bc bcx() const {return g.bcx();}

    /**
     * @brief global y boundary
     *
     * @return boundary condition
     */
    bc bcy() const {return g.bcy();}

    /**
     * @brief Return mpi cartesian communicator that is used in this grid
     *
     * @return Communicator
     */
    MPI_Comm communicator() const{return comm;}

    /**
     * @brief The data of dlt
     *
     * @return 
     */
    const DLT<double>& dlt() const{return g.dlt();}

    /**
     * @brief Return cartesian
     *
     * No other is implemented yet
     * @return coordinate system
     */
    dg::system system() const{return dg::cartesian;}
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

    /**
     * @brief Return a grid local for the calling process
     *
     * The local grid returns the same values for x0(), x1(), ..., Nx(), Ny(), ... as the grid
     * class itself
     * @return Grid object
     */
    Grid2d<double> local() const {return Grid2d<double>(x0(), x1(), y0(), y1(), n(), Nx(), Ny(), bcx(), bcy());}

    /**
     * @brief Return a grid global for the calling process
     *
     * The global grid contains the global boundaries
     * @return Grid object
     */
    Grid2d<double> global() const {return g;}

    private:
    Grid2d<double> g; //global grid
    MPI_Comm comm; //just an integer...

};

/**
 * @brief 3D MPI Grid class 
 *
 * Represents the local grid coordinates and the process topology. 
 * It just divides the given box into overlapping subboxes that are attributed to each process
 * @attention
 * The boundaries in the constructors are global boundaries, the boundaries given in the access functions are local boundaries, this is because the grid represents the information given to one process
 *
 * @note Note
 * that the grids of different processes overlap in the x- and y- coordinate but 
 * not in the z-coordinate.
 * Also note that a single cell is never divided across processes.
 */
struct MPI_Grid3d
{
    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param comm mpi communicator
     * @note in the cylindrical coordinate system x, y and z are used to denote R, Z and the angle phi
     * @attention # of polynomial coefficients in z direction is always 1
     */
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

    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     * @param sys cartesian or cylindrical
     * @param comm mpi communicator
     * @note in the cylindrical coordinate system x, y and z are used to denote R, Z and the angle phi
     * @attention # of polynomial coefficients in z direction is always 1
     */
    MPI_Grid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz,dg::system sys, MPI_Comm comm):
        g( x0, x1, y0, y1, z0, z1, n, Nx, Ny, Nz, bcx, bcy, bcz, sys), comm( comm)
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

    /**
     * @brief Return local x0
     *
     * The local value is shifted by hx 
     * @return local left boundary
     */
    double x0() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.x0() - g.hx() + g.lx()/(double)dims[0]*(double)coords[0]; 
    }
    /**
     * @brief Return local x1
     *
     * The local value is shifted by hx 
     * @return local right boundary
     */
    double x1() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.x0() + g.hx() + g.lx()/(double)dims[0]*(double)(coords[0]+1); 
    }
    /**
     * @brief Return local y0
     *
     * The local value is shifted by hy 
     * @return local left boundary
     */
    double y0() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.y0() - g.hy() + g.ly()/(double)dims[1]*(double)coords[1]; 
    }
    /**
     * @brief Return local y1
     *
     * The local value is shifted by hy 
     * @return local right boundary
     */
    double y1() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.y0() + g.hy() + g.ly()/(double)dims[1]*(double)(coords[1]+1); 
    }
    /**
     * @brief Return local z0
     *
     * The z - local value is not shifted
     * @return local right boundary
     */
    double z0() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.z0() + g.lz()/(double)dims[2]*(double)coords[2]; 
    }
    /**
     * @brief Return local z1
     *
     * The z - local value is not shifted
     * @return local right boundary
     */
    double z1() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.z0() + g.lz()/(double)dims[2]*(double)(coords[2]+1); 
    }
    /**
     * @brief Return local lx
     *
     * @return local length
     */
    double lx() const {return x1()-x0();}
    /**
     * @brief Return local ly
     *
     * @return local length
     */
    double ly() const {return y1()-y0();}
    /**
     * @brief Return local lz
     *
     * @return local length
     */
    double lz() const {return z1()-z0();}
    /**
     * @brief Return local hx
     *
     * @return local grid constant
     */
    double hx() const {return g.hx();}
    /**
     * @brief Return local hy
     *
     * @return local grid constant
     */
    double hy() const {return g.hy();}
    /**
     * @brief Return local hz
     *
     * @return local grid constant
     */
    double hz() const {return g.hz();}
    /**
     * @brief Return n
     *
     * @return number of polynomial coefficients
     */
    unsigned n() const {return g.n();}
    /**
     * @brief Return the local number of cells 
     *
     * Not the one given in the constructor
     * @return number of cells
     */
    unsigned Nx() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.Nx()/dims[0]+2;
    }
    /**
     * @brief Return the local number of cells 
     *
     * Not the one given in the constructor
     * @return number of cells
     */
    unsigned Ny() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.Ny()/dims[1]+2;
    }
    /**
     * @brief Return the local number of cells 
     *
     * Not the one given in the constructor
     * @return number of cells
     */
    unsigned Nz() const {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return g.Nz()/dims[2];
    }
    /**
     * @brief global x boundary
     *
     * @return boundary condition
     */
    bc bcx() const {return g.bcx();}
    /**
     * @brief global y boundary
     *
     * @return boundary condition
     */
    bc bcy() const {return g.bcy();}
    /**
     * @brief global z boundary
     *
     * @return boundary condition
     */
    bc bcz() const {return g.bcz();}
    /**
     * @brief Return mpi cartesian communicator that is used in this grid
     *
     * @return Communicator
     */
    MPI_Comm communicator() const{return comm;}
    /**
     * @brief The data of dlt
     *
     * @return 
     */
    const DLT<double>& dlt() const{return g.dlt();}
    /**
     * @brief Return cartesian or cylindrical
     *
     * @return coordinate system
     */
    dg::system system() const {return g.system();}
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
     * The local grid returns the same values for x0(), x1(), ..., Nx(), Ny(), ... as the grid
     * class itself
     * @return Grid object
     */
    Grid3d<double> local() const {return Grid3d<double>(x0(), x1(), y0(), y1(), z0(), z1(), n(), Nx(), Ny(), Nz(), bcx(), bcy(), bcz(), system());}
    /**
     * @brief Return a grid global for the calling process
     *
     * The global grid contains the global boundaries
     * @return Grid object
     */
    Grid3d<double> global() const {return g;}
    /**
     * @brief Return a grid local to the calling process without ghostcells
     *
     * The local grid returns the unshifted values for x0(), x1(), ...
     * @return Grid object
     */
    Grid3d<double> ghostless( ) const
    {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        return Grid3d<double>( 
                g.x0() + (g.x1()-g.x0())/(double)dims[0]*(double)coords[0], 
                g.x0() + (g.x1()-g.x0())/(double)dims[0]*(double)(coords[0]+1), 
                g.y0() + (g.y1()-g.y0())/(double)dims[1]*(double)coords[1], 
                g.y0() + (g.y1()-g.y0())/(double)dims[1]*(double)(coords[1]+1), 
                g.z0() + (g.z1()-g.z0())/(double)dims[2]*(double)coords[2], 
                g.z0() + (g.z1()-g.z0())/(double)dims[2]*(double)(coords[2]+1), 
                g.n(),
                g.Nx()/dims[0],
                g.Ny()/dims[1], 
                g.Nz()/dims[2]);
    }
    /**
     * @brief Returns the pid of the process that holds the local grid surrounding a given point
     *
     * local means that there is a margin of hx, hy around the x-y planes
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
    //points in the (outer) ghost cells layer? (note the global grid used)
    if( x < g.x0() && x >= g.x0() - g.hx()) x += g.hx();
    if( x > g.x1() && x <= g.x1() + g.hx()) x -= g.hx();
    if( y < g.y0() && y >= g.y0() - g.hy()) y += g.hy();
    if( y > g.y1() && y <= g.y1() + g.hy()) y -= g.hy();
    coords[0] = (unsigned)floor( (x-g.x0())/g.lx()*(double)dims[0] );
    coords[1] = (unsigned)floor( (y-g.y0())/g.ly()*(double)dims[1] );
    coords[2] = (unsigned)floor( (z-g.z0())/g.lz()*(double)dims[2] );
    //if point lies on boundary of last cell ... (not so good for periodic boundaries)
    coords[0]=(coords[0]==dims[0]) ? coords[0]-1 :coords[0];
    coords[1]=(coords[1]==dims[1]) ? coords[1]-1 :coords[1];
    coords[2]=(coords[2]==dims[2]) ? coords[2]-1 :coords[2];
    int rank;
    if( MPI_Cart_rank( comm, coords, &rank) == MPI_SUCCESS ) 
        return rank;
    else
        return -1;
}

namespace create{
/**
 * @brief Create a local grid without ghostcells (useful for interpolation)
 *
 * @param x0 global x0
 * @param x1 global x1
 * @param y0 global y0
 * @param y1 global y1
 * @param z0 global z0
 * @param z1 global z1
 * @param n global n
 * @param Nx global Nx
 * @param Ny global Ny
 * @param Nz global Nz
 * @param comm MPI communicator
 *
 * @return local grid without overlapping cells
 */
Grid3d<double> ghostless_grid( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm)
{
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    if( coords[0] == 0 && coords[1] == 0 && coords[2] == 0)
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
    return Grid3d<double>( 
            x0 + (x1-x0)/(double)dims[0]*(double)coords[0], 
            x0 + (x1-x0)/(double)dims[0]*(double)(coords[0]+1), 
            y0 + (y1-y0)/(double)dims[1]*(double)coords[1], 
            y0 + (y1-y0)/(double)dims[1]*(double)(coords[1]+1), 
            z0 + (z1-z0)/(double)dims[2]*(double)coords[2], 
            z0 + (z1-z0)/(double)dims[2]*(double)(coords[2]+1), 
            n,
            Nx/dims[0],
            Ny/dims[1], 
            Nz/dims[2]);
}

} //namespace create

///@}
}//namespace dg
