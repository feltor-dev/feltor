#pragma once

#include "mpi.h"
#include "average.h"
#include "mpi_grid.h"
#include "mpi_weights.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

/**
 * @brief MPI specialized class for y average computations
 *
 * @snippet backend/average_mpit.cu doxygen
 * @ingroup utilities
 */
template< class container>
struct Average<MPI_Vector<container> >
{

    Average( const aMPITopology2d& g, enum coo2d direction)
    {
        m_nx = g.local().Nx()*g.n(), m_ny = g.local().Ny()*g.n();
        m_w=dg::transfer<MPI_Vector<container>>(dg::create::weights(g, direction));
        int remain_dims[] = {false,false}; //true true false
        m_transpose = false;
        if( direction == dg::coo2d::x)
        {
            dg::blas1::scal( m_w, 1./g.lx());
            remain_dims[0] = true;
        }
        else
        {
            dg::blas1::scal( m_w, 1./g.ly());
            m_transpose = true;
            remain_dims[1] = true;
        }
        m_temp1d = m_temp = m_w;
        MPI_Cart_sub( g.communicator(), remain_dims, &m_comm);
        exblas::mpi_reduce_communicator( m_comm, &m_comm_mod, &m_comm_mod_reduce);
    }

    Average( const aMPITopology3d& g, enum coo3d direction)
    {
        m_w = dg::transfer<MPI_Vector<container>>(dg::create::weights(g, direction));
        m_transpose = false;
        unsigned nx = g.n()*g.local().Nx(), ny = g.n()*g.local().Ny(), nz = g.local().Nz();
        int remain_dims[] = {false,false,false}; //true true false
        m_transpose = false;
        if( direction == dg::coo3d::x) {
            dg::blas1::scal( m_w, 1./g.lx());
            m_nx = nx, m_ny = ny*nz;
            remain_dims[0] = true;
        }
        else if( direction == dg::coo3d::z) {
            dg::blas1::scal( m_w, 1./g.lz());
            m_nx = nx*ny, m_ny = nz;
            m_transpose = true;
            remain_dims[2] = true;
        }
        else if( direction == dg::coo3d::xy) {
            dg::blas1::scal( m_w, 1./g.lx()/g.ly());
            m_nx = nx*ny, m_ny = nz;
            remain_dims[0] = remain_dims[1] = true;
        }
        else if( direction == dg::coo3d::yz) {
            dg::blas1::scal( m_w, 1./g.ly()/g.lz());
            m_nx = nx, m_ny = ny*nz;
            m_transpose = true;
            remain_dims[1] = remain_dims[2] = true;
        }
        else 
            std::cerr << "Warning: this direction is not implemented\n";
        MPI_Cart_sub( g.communicator(), remain_dims, &m_comm);
        exblas::mpi_reduce_communicator( m_comm, &m_comm_mod, &m_comm_mod_reduce);
        m_temp1d = m_temp = m_w;
    }
    /**
     * @brief Compute the average 
     *
     * @param src 2D Source Vector (must have the same size as the grid given in the constructor)
     * @param res 2D result Vector (may alias src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const MPI_Vector<container>& src, MPI_Vector<container>& res)
    {
        if( !m_transpose)
        {
            dg::mpi_average( m_nx, m_ny, src.data(), m_w.data(), m_temp.data(), m_comm, m_comm_mod, m_comm_mod_reduce);
            dg::extend_column( m_nx, m_ny, m_temp.data(), res.data());
        }
        else 
        {
            dg::transpose( m_nx, m_ny, src.data(), m_temp.data());
            dg::mpi_average( m_ny, m_nx, m_temp.data(), m_w.data(), m_temp1d.data(), m_comm, m_comm_mod, m_comm_mod_reduce);
            dg::extend_line( m_nx, m_ny, m_temp1d.data(), res.data());
        }

    }
  private:
    unsigned m_nx, m_ny;
    MPI_Vector<container> m_w, m_temp, m_temp1d; 
    bool m_transpose;
    MPI_Comm m_comm, m_comm_mod, m_comm_mod_reduce;
};


}//namespace dg
