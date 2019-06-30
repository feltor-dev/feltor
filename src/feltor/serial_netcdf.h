#pragma once

namespace feltor
{
template<class Container>
void slice_vector3d( const Container& transfer, Container& transfer2d, size_t local_size2d)
{
#ifdef FELTOR_MPI
    thrust::copy(
        transfer.data().begin(),
        transfer.data().begin() + local_size2d,
        transfer2d.data().begin()
    );
#else
    thrust::copy(
        transfer.begin(),
        transfer.begin() + local_size2d,
        transfer2d.begin()
    );
#endif
}
//Manages the communication for serial netcdf output
struct ManageOutput
{
    ManageOutput( const Geometry& g3d_out)
    {
        std::unique_ptr<typename Geometry::perpendicular_grid> g2d_out_ptr  ( dynamic_cast<typename Geometry::perpendicular_grid*>( g3d_out.perp_grid()));
#ifdef FELTOR_MPI
        m_local_size3d = g3d_out.local().size();
        m_local_size2d = g2d_out_ptr->local().size();
        m_comm3d = g3d_out.communicator();
        m_comm2d = g2d_out_ptr->communicator();
        MPI_Comm_rank( m_comm2d, &m_rank2d);
        MPI_Comm_size( m_comm2d, &m_size2d);
        MPI_Comm_rank( m_comm3d, &m_rank3d);
        MPI_Comm_size( m_comm3d, &m_size3d);

        m_coords2d.resize(m_size2d*2);
        m_coords.resize(m_size3d*3);
        for( int rrank=0; rrank<m_size2d; rrank++)
            MPI_Cart_coords( m_comm2d, rrank, 2, &m_coords2d[2*rrank]);
        for( int rrank=0; rrank<m_size3d; rrank++)
            MPI_Cart_coords( m_comm3d, rrank, 3, &m_coords[3*rrank]);
        m_count4d[0] = 1;
        m_count4d[1] = g3d_out.local().Nz();
        m_count4d[2] = g3d_out.n()*(g3d_out.local().Ny());
        m_count4d[3] = g3d_out.n()*(g3d_out.local().Nx());
        m_count3d[0] = 1;
        m_count3d[1] = g3d_out.n()*(g3d_out.local().Ny()),
        m_count3d[2] = g3d_out.n()*(g3d_out.local().Nx());
#else //FELTOR_MPI
        m_count4d[0] = 1;
        m_count4d[1] = g3d_out.Nz();
        m_count4d[2] = g3d_out.n()*g3d_out.Ny();
        m_count4d[3] = g3d_out.n()*g3d_out.Nx();
        m_count3d[0] = 1;
        m_count3d[1] = g3d_out.n()*g3d_out.Ny();
        m_count3d[2] = g3d_out.n()*g3d_out.Nx();
        m_local_size3d = g3d_out.size();
        m_local_size2d = g2d_out_ptr->size();
#endif //FELTOR_MPI
    }
    //must enddef first
    void output_static3d(int ncid, int vecID, HVec& transferH) const
    {
        file::NC_Error_Handle err;
        size_t start4d[4] = {0,0,0,0};
#ifdef FELTOR_MPI
        MPI_Status status;
        if(m_rank3d==0)
        {
            for( int rrank=0; rrank<m_size3d; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                          rrank, rrank, m_comm3d, &status);
                start4d[1] = m_coords[3*rrank+2]*m_count4d[1],
                start4d[2] = m_coords[3*rrank+1]*m_count4d[2],
                start4d[3] = m_coords[3*rrank+0]*m_count4d[3];
                err = nc_put_vara_double( ncid, vecID, &start4d[1], &m_count4d[1],
                    transferH.data().data());
            }
        }
        else
            MPI_Send( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                      0, m_rank3d, m_comm3d);
        MPI_Barrier( m_comm3d);
#else
        err = nc_put_vara_double( ncid, vecID, &start4d[1], &m_count4d[1],
            transferH.data());
#endif // FELTOR_MPI
    }
    void output_dynamic3d(int ncid, int vecID, unsigned start, HVec& transferH) const
    {
        size_t start4d[4] = {start, 0, 0, 0};
        file::NC_Error_Handle err;
#ifdef FELTOR_MPI
        MPI_Status status;
        if(m_rank3d==0)
        {
            for( int rrank=0; rrank<m_size3d; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                          rrank, rrank, m_comm3d, &status);
                start4d[1] = m_coords[3*rrank+2]*m_count4d[1],
                start4d[2] = m_coords[3*rrank+1]*m_count4d[2],
                start4d[3] = m_coords[3*rrank+0]*m_count4d[3];
                err = nc_put_vara_double( ncid, vecID, start4d, m_count4d,
                    transferH.data().data());
            }
        }
        else
            MPI_Send( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                      0, m_rank3d, m_comm3d);
        MPI_Barrier( m_comm3d);
#else
        err = nc_put_vara_double( ncid, vecID, start4d, m_count4d,
            transferH.data());
#endif // FELTOR_MPI
    }

//all send to their rank2d 0 but only rank3d 0 writes into file
    void output_dynamic2d_slice(int ncid, int vecID, unsigned start, HVec& transferH2d) const
    {
        file::NC_Error_Handle err;
        size_t start3d[3] = {start, 0, 0};
#ifdef FELTOR_MPI
        MPI_Status status;
        // 2d data of plane varphi = 0
        if(m_rank2d==0)
        {
            for( int rrank=0; rrank<m_size2d; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( transferH2d.data().data(), m_local_size2d, MPI_DOUBLE,
                          rrank, rrank, m_comm2d, &status);
                start3d[1] = m_coords2d[2*rrank+1]*m_count3d[1],
                start3d[2] = m_coords2d[2*rrank+0]*m_count3d[2];
                err = nc_put_vara_double( ncid, vecID, start3d, m_count3d,
                    transferH2d.data().data());
            }
        }
        else
            MPI_Send( transferH2d.data().data(), m_local_size2d, MPI_DOUBLE,
                      0, m_rank2d, m_comm2d);
        MPI_Barrier( m_comm2d);
        MPI_Barrier( m_comm3d); //all processes synchronize
#else
        err = nc_put_vara_double( ncid, vecID, start3d, m_count3d,
            transferH2d.data());
#endif // FELTOR_MPI
    }
    private:
    unsigned m_local_size2d;
    unsigned m_local_size3d;
    size_t m_count3d[3];
    size_t m_count4d[4];
    int m_rank3d, m_size3d, m_rank2d, m_size2d;
#ifdef FELTOR_MPI
    MPI_Comm m_comm2d, m_comm3d;
    std::vector<int> m_coords, m_coords2d;
#endif //FELTOR_MPI
};

}//namespace feltor
