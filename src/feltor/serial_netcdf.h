#pragma once

namespace feltor
{

struct ManageOutput
{

    ManageOutput( Geometry& g3d_out):
    {
        std::unique_ptr<Geometry2d> g2d_out_ptr = g3d_out.perp_grid();
        m_start4d = {0, 0, 0, 0};
        m_start3d = {0, 0, 0};
        m_local_size3d = g3d_out.size();
        m_local_size2d = g2d_out_ptr->size();
#ifdef FELTOR_MPI
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
        m_count4d = {1, g3d_out.local().Nz(),
            g3d_out.n()*(g3d_out.local().Ny()),
            g3d_out.n()*(g3d_out.local().Nx())};
        m_count3d = {1,
            g3d_out.n()*(g3d_out.local().Ny()),
            g3d_out.n()*(g3d_out.local().Nx())};
#else //FELTOR_MPI
        m_count4d = {1, g3d_out.Nz(), g3d_out.n()*g3d_out.Ny(),
            g3d_out.n()*g3d_out.Nx()};
        m_count3d = {1, g3d_out.n()*g3d_out.Ny(),
            g3d_out.n()*g3d_out.Nx()};
#endif //FELTOR_MPI
    }
    //must enddef first
    void output_static3d(int ncid, int vecID, HVec& transferH) const
    {
#ifdef FELTOR_MPI
        if(m_rank3d==0)
        {
            for( int rrank=0; rrank<m_size3d; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                          rrank, rrank, m_comm3d, &m_status);
                m_start4d[1] = m_coords[3*rrank+2]*m_count4d[1],
                m_start4d[2] = m_coords[3*rrank+1]*m_count4d[2],
                m_start4d[3] = m_coords[3*rrank+0]*m_count4d[3];
                m_err = nc_put_vara_double( ncid, vecID, &start4d[1], &count4d[1],
                    transferH.data().data());
            }
        }
        else
            MPI_Send( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                      0, m_rank3d, m_comm3d);
        MPI_Barrier( m_comm3d);
#else
        m_err = nc_put_vara_double( ncid, vecID, &m_start4d[1], &m_count4d[1],
            transferH.data());
#endif // FELTOR_MPI
    }
    void output_dynamic3d(int ncid, int vecID, int start, HVec& transferH) const
    {
        m_start4d[0] = start;
#ifdef FELTOR_MPI
        if(m_rank3d==0)
        {
            for( int rrank=0; rrank<m_size3d; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                          rrank, rrank, m_comm3d, &m_status);
                m_start4d[1] = m_coords[3*rrank+2]*m_count4d[1],
                m_start4d[2] = m_coords[3*rrank+1]*m_count4d[2],
                m_start4d[3] = m_coords[3*rrank+0]*m_count4d[3];
                m_err = nc_put_vara_double( ncid, vecID, start4d, count4d,
                    transferH.data().data());
            }
        }
        else
            MPI_Send( transferH.data().data(), m_local_size3d, MPI_DOUBLE,
                      0, m_rank3d, m_comm3d);
        MPI_Barrier( m_comm3d);
#else
        m_err = nc_put_vara_double( ncid, vecID, m_start4d, m_count4d,
            transferH.data());
#endif // FELTOR_MPI
    }

//all send to their rank2d 0 but only rank3d 0 writes into file
    void output_dynamic2d(int ncid, int vecID, int start, HVec& transferH2d) const
    {
        m_start3d[0] = start;
#ifdef FELTOR_MPI
        if(m_rank2d==0)
        {
            for( int rrank=0; rrank<m_size2d; rrank++)
            {
                if(rrank!=0)
                    MPI_Recv( transferH.data().data(), m_local_size2d, MPI_DOUBLE,
                          rrank, rrank, m_comm2d, &m_status);
                m_start3d[1] = m_coords2d[2*rrank+1]*m_count3d[1],
                m_start3d[2] = m_coords2d[2*rrank+0]*m_count3d[2];
                m_err = nc_put_vara_double( ncid, vecID, start3d, count3d,
                    transferH2d.data().data());
            }
        }
        else
            MPI_Send( transferH2d.data().data(), m_local_size2d, MPI_DOUBLE,
                      0, m_rank2d, m_comm2d);
        MPI_Barrier( m_comm2d);
        MPI_Barrier( m_comm3d); //all processes synchronize
#else
        m_err = nc_put_vara_double( ncid, vecID, m_start3d, m_count3d,
            transferH2d.data());
#endif // FELTOR_MPI
    }
    private
    unsigned m_local_size2d;
    unsigned m_local_size3d;
    size_t m_start3d[3], m_count3d[3];
    size_t m_start4d[4], m_count4d[4];
    file::NC_Error_Handle m_err;
    int m_rank3d, m_size3d, m_rank2d, m_size2d;
#ifdef FELTOR_MPI
    MPI_Status m_status;
    MPI_Comm m_comm2d, m_comm3d;
    std::vector<int> m_coords, m_coords2d;
#endif //FELTOR_MPI
};

}//namespace feltor
