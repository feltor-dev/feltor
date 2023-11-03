#pragma once

namespace feltor
{

namespace detail
{
// helpers to define a C++-17 if constexpr
template<class Geometry, unsigned ndim>
struct CreateInterpolation{};


template<class Geometry>
struct CreateInterpolation<Geometry,1>
{
    auto call( const std::vector<dg::HVec>& x, const Geometry& g)
    {
        return dg::create::interpolation( x[0], g, g.bcx());

    }
};
template<class Geometry>
struct CreateInterpolation<Geometry,2>
{
auto call( const std::vector<dg::HVec>& x, const Geometry& g)
{
    return dg::create::interpolation( x[0], x[1], g, g.bcx(), g.bcy());

}
};
template<class Geometry>
struct CreateInterpolation<Geometry,3>
{
auto call( const std::vector<dg::HVec>& x, const Geometry& g)
{
    return dg::create::interpolation( x[0], x[1], x[2], g, g.bcx(), g.bcy(), g.bcz());

}
};
}



struct Probes
{
    Probes() = default;
    template<class Geometry, class ListClass>
    Probes(
        int ncid,
        unsigned itstp,
        const dg::file::WrappedJsonValue& js,
        const Geometry& grid,
        std::vector<std::string> coords_names,
        std::vector<bool> normalize,
        const ListClass& probe_list
        )
    {
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        m_probes = js.isMember("probes");
        if( js.isMember("probe"))
            throw std::runtime_error( "Field <probe> found! Did you mean \"probes\"?");

        if(m_probes)
        {
            auto js_probes = dg::file::WrappedJsonValue( dg::file::error::is_throw);
            // only master thread reads probes
            auto coords = parse_probes( js, js_probes, coords_names, normalize);
            if ( coords_names.size() != grid.ndim())
                throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in coords_names!");
            if ( normalize.size() != grid.ndim())
                throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in normalize!");

            m_probe_interpolate = detail::CreateInterpolation<Geometry, Geometry::ndim()>().call( coords, grid);

            // Create helper storage probe variables
#ifdef WITH_MPI
            if(rank==0) m_simple_probes = dg::MHVec(coords[0], grid.communicator());
#else //WITH_MPI
            m_simple_probes = dg::HVec(m_num_pins);
#endif
            for( auto& record : probe_list)
                m_simple_probes_intern[record.name] = std::vector<dg::x::HVec>(itstp, m_simple_probes);
            m_time_intern.resize(itstp);
            m_resultD = dg::evaluate( dg::zero, grid);
            m_resultH = dg::evaluate( dg::zero, grid);
            define_nc_variables( ncid, js_probes, coords, coords_names, probe_list);
        }
    }

    // record.name, record.long_name, record.function( resultH, ps...)
    template<class ListClass, class ...Params>
    void static_write( const ListClass& diag_static_list, Params&& ... ps)
    {
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        if(m_probes)
        {
            dg::file::NC_Error_Handle err;
            for ( auto& record : diag_static_list)
            {
                int vecID;
                DG_RANK0 err = nc_def_var( m_probe_grp_id, record.name.data(), NC_DOUBLE, 1,
                    &m_probe_dim_ids[1], &vecID);
                DG_RANK0 err = nc_put_att_text( m_probe_grp_id, vecID,
                    "long_name", record.long_name.size(), record.long_name.data());
                record.function( m_resultH, std::forward<Params>(ps)...);
                dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
                DG_RANK0 nc_put_var_double( m_probe_grp_id, vecID,
#ifdef WITH_MPI
                    m_simple_probes.data().data()
#else
                    m_simple_probes.data()
#endif
                        );
            }
        }
    }

    // record.name, record.long_name, record.function( resultH, ps...)
    template<class ListClass, class ...Params>
    void write( double time, const ListClass& probe_list, Params&& ... ps)
    {
        dg::file::NC_Error_Handle err;
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        /// Probes FIRST output ///
        size_t probe_count[] = {1, m_num_pins};
        if(m_probes)
        {
            DG_RANK0 err = nc_put_vara_double( m_probe_grp_id, m_probe_timevarID,
                    &m_probe_start[0], &probe_count[0], &time);
            for( auto& record : probe_list)
            {
                record.function( m_resultD, std::forward<Params>(ps)...);
                dg::assign( m_resultD, m_resultH);
                dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
                DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                        m_probe_id_field.at(record.name), m_probe_start, probe_count,
#ifdef WITH_MPI
                        m_simple_probes.data().data()
#else
                        m_simple_probes.data()
#endif
                );
            }
        }
         /// End probes output ///

    }
    // is thought to be called itstp times before write
    template<class ListClass, class ...Params>
    void save( double time, unsigned iter, const ListClass& list, Params&& ... ps)
    {
        if(m_probes)
        {
            for( auto& record : list)
            {
                record.function( m_resultD, std::forward<Params>(ps)...);
                dg::assign( m_resultD, m_resultH);
                dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
                m_simple_probes_intern.at(record.name)[iter]=m_simple_probes;
                m_time_intern[iter]=time;
            }
        }
    }
    void write_after_save()
    {
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        size_t probe_count[] = {1, m_num_pins};
        //OUTPUT OF PROBES
        if(m_probes)
        {
            dg::file::NC_Error_Handle err;
            for( unsigned j=0; j<m_time_intern.size(); j++)
            {
                m_probe_start[0] += 1;
                DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                        m_probe_timevarID, &m_probe_start[0] , &probe_count[0],
                        &m_time_intern[j]);
                for( auto& field : m_simple_probes_intern)
                {
                    DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                            m_probe_id_field.at(field.first), m_probe_start,
                            probe_count,
#ifdef WITH_MPI
                            field.second[j].data().data()
#else
                            field.second[j].data()
#endif
                            );
                }
            }
        }
    }

    private:
    bool m_probes = false;
    int m_probe_grp_id;
    int m_probe_dim_ids[2];
    int m_probe_timevarID;
    std::map<std::string, int> m_probe_id_field;
    unsigned m_num_pins;
    dg::x::IHMatrix m_probe_interpolate;
    dg::x::HVec m_simple_probes;
    std::map<std::string, std::vector<dg::x::HVec>> m_simple_probes_intern;
    std::vector<double> m_time_intern;
    dg::x::DVec m_resultD;
    dg::x::HVec m_resultH;
    size_t m_probe_start[2] = {0,0};

    dg::HVec read_probes( const dg::file::WrappedJsonValue& probes, std::string x,
            double rhos)
    {
        unsigned size = probes[x].size();
        dg::HVec out(size);
        for( unsigned i=0; i<size; i++)
            out[i] = probes.asJson()[i].asDouble()/rhos;
        return out;
    }
    std::vector<dg::HVec> parse_probes( const dg::file::WrappedJsonValue& js,
        dg::file::WrappedJsonValue& js_probes,
        std::vector<std::string> coords_names, std::vector<bool> normalize)
    {

        std::vector<dg::HVec> coords( coords_names.size());
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank == 0)
        {
#endif //WITH_MPI
        std::string path;
        bool file = true;
        try{
            path = js["probes"].asString();
        }catch ( std::runtime_error& e) { file = false; }
        if ( file)
            js_probes.asJson() = dg::file::file2Json( path,
                    dg::file::comments::are_discarded, dg::file::error::is_throw);
        else
            js_probes.asJson() = js.asJson()["probes"];
        double rhos = 1.;
        if( file)
        {
            //try{
                rhos = js["physical"]["rho_s"].asDouble();
            //} catch( std::exception& e) {
            //    DG_RANK0 std::cerr << "rho_s needs to be present in input file "<<argv1<<" if magnetic field from file\n";
            //    DG_RANK0 std::cerr << e.what()<<std::endl;
            //    dg::abort_program();
            //}
        }
        for( unsigned i=0; i<coords_names.size(); i++)
            coords[i] = read_probes( js_probes, coords_names[i], normalize[i] ? rhos : 1);
        m_num_pins = coords[0].size();
        for( unsigned i=1; i<coords_names.size(); i++)
        {
            unsigned num_pins = coords[i].size();
            if( m_num_pins != num_pins)
                throw std::runtime_error( "Size of "+coords_names[i] +" probes array ("
                        +std::to_string(num_pins)+") does not match that of "+coords_names[0]+" ("
                        +std::to_string(m_num_pins)+")!");
        }
#ifdef WITH_MPI
        }
#endif //WITH_MPI
        return coords;
    }

    template<class ListClass>
    void define_nc_variables( int ncid, const dg::file::WrappedJsonValue& js_probes,
    const std::vector<dg::HVec>& coords, const std::vector<std::string> & coords_names,
    const ListClass& probe_list)
    {
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        dg::file::NC_Error_Handle err;
        DG_RANK0 err = nc_def_grp(ncid,"probes",&m_probe_grp_id);
        std::string format = js_probes["format"].toStyledString();
        DG_RANK0 err = nc_put_att_text( m_probe_grp_id, NC_GLOBAL,
            "format", format.size(), format.data());
        dg::Grid1d g1d( 0,1,1,m_num_pins);
        DG_RANK0 err = dg::file::define_dimensions( m_probe_grp_id,
                m_probe_dim_ids, &m_probe_timevarID, g1d, {"time", "x"});
        std::vector<int> pin_id;
        for( unsigned i=0; i<coords.size(); i++)
        {
            int pin_id;
            DG_RANK0 err = nc_def_var(m_probe_grp_id, coords_names[i].data(),
                NC_DOUBLE, 1, &m_probe_dim_ids[1], &pin_id);
            DG_RANK0 err = nc_put_var_double( m_probe_grp_id, pin_id, coords[i].data());
        }
        for( auto& record : probe_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            m_probe_id_field[name] = 0;//creates a new id4d entry for all processes
            DG_RANK0 err = nc_def_var( m_probe_grp_id, name.data(),
                    NC_DOUBLE, 2, m_probe_dim_ids,
                    &m_probe_id_field.at(name));
            DG_RANK0 err = nc_put_att_text( m_probe_grp_id,
                    m_probe_id_field.at(name), "long_name", long_name.size(),
                    long_name.data());
        }
    }

};

} //namespace feltor
