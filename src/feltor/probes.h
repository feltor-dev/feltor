#pragma once

#include "feltordiag.h"

namespace feltor
{

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



struct Probes
{
    Probes() = default;
    template<class Geometry>
    Probes(
        int ncid,
        unsigned itstp,
        const dg::file::WrappedJsonValue& js,
        const Geometry& grid,
        std::vector<std::string> coords_names,
        std::vector<bool> normalize)
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
            auto coords = parse_probes( js, js_probes, coords_names, normalize);
            if ( coords_names.size() != grid.ndim())
                throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in coords_names!");
            if ( normalize.size() != grid.ndim())
                throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in normalize!");

            static_assert( grid.ndim() == 3);

            m_probe_interpolate = CreateInterpolation<Geometry, Geometry::ndim()>().call( coords, grid);

            // Create helper storage probe variables
#ifdef WITH_MPI
            // every processor gets the probes (slightly inefficient...)
            m_simple_probes = dg::MHVec(m_R, grid.communicator());
#else //WITH_MPI
            m_simple_probes = dg::HVec(m_num_pins);
#endif
            for( auto& record : m_probe_list)
                m_simple_probes_intern[record.name] = std::vector<dg::x::HVec>(itstp, m_simple_probes);
            m_time_intern.resize(itstp);
            m_resultD = dg::evaluate( dg::zero, grid);
            m_resultH = dg::evaluate( dg::zero, grid);

            dg::file::NC_Error_Handle err;
            DG_RANK0 err = nc_def_grp(ncid,"probes",&m_probe_grp_id);
            std::string format = js_probes["format"].toStyledString();
            DG_RANK0 err = nc_put_att_text( m_probe_grp_id, NC_GLOBAL,
                "format", format.size(), format.data());
            dg::Grid1d g1d( 0,1,1,m_num_pins);
            DG_RANK0 err = dg::file::define_dimensions( m_probe_grp_id,
                    m_probe_dim_ids, &m_probe_timevarID, g1d, {"time", "x"});
            std::vector<int> pin_id;
            for( unsigned i=0; i<grid.ndim(); i++)
            {
                int pin_id;
                DG_RANK0 err = nc_def_var(m_probe_grp_id, coords_names[i].data(), NC_DOUBLE, 1, &m_probe_dim_ids[1], &pin_id);
                DG_RANK0 err = nc_put_var_double( m_probe_grp_id, pin_id, coords[i].data());
            }
            for( auto& record : m_probe_list)
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
    }

    void first_write( Variables& var, double time, const dg::x::CylindricalGrid3d& grid)
    {
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        /// Probes FIRST output ///
        size_t probe_count[] = {1, m_num_pins};
        m_time_intern[0]=time;
        if(m_probes)
        {
            dg::file::NC_Error_Handle err;
            for ( auto& record : feltor::diagnostics2d_static_list)
            {
                int vecID;
                DG_RANK0 err = nc_def_var( m_probe_grp_id, record.name.data(), NC_DOUBLE, 1,
                    &m_probe_dim_ids[1], &vecID);
                DG_RANK0 err = nc_put_att_text( m_probe_grp_id, vecID,
                    "long_name", record.long_name.size(), record.long_name.data());
                record.function( m_resultH, var, grid);
                dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
                DG_RANK0 nc_put_var_double( m_probe_grp_id, vecID,
#ifdef WITH_MPI
                    m_simple_probes.data().data()
#else
                    m_simple_probes.data()
#endif
                        );
            }
            DG_RANK0 err = nc_put_vara_double( m_probe_grp_id, m_probe_timevarID,
                    &m_probe_start[0], &probe_count[0], &m_time_intern[0]);

            for( auto& record : m_probe_list)
            {
                record.function( m_resultD, var);
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
    void save( Variables& var, double time, unsigned iter)
    {
        if(m_probes)
        {
            for( auto& record : m_probe_list)
            {
                record.function( m_resultD, var);
                dg::assign( m_resultD, m_resultH);
                dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
                m_simple_probes_intern.at(record.name)[iter]=m_simple_probes;
                m_time_intern[iter]=time;
            }
        }
    }
    void write_after_save( Variables& var)
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
                for( auto& record : m_probe_list)
                {
#ifdef WITH_MPI
                    DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                            m_probe_id_field.at(record.name), m_probe_start,
                            probe_count,
                            m_simple_probes_intern.at(record.name)[j].data().data());
#else
                    DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                            m_probe_id_field.at(record.name), m_probe_start,
                            probe_count,
                            m_simple_probes_intern.at(record.name)[j].data());
#endif
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
    dg::HVec m_R, m_Z, m_P;
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
        std::vector<std::string> coords_names, std::vector<bool> normalize){
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
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
        std::vector<dg::HVec> coords( coords_names.size());
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
        return coords;
    }

    // probes list
struct Record{
    std::string name;
    std::string long_name;
    std::function<void( dg::x::DVec&, Variables&)> function;
};
std::vector<Record> m_probe_list = {
     {"ne", "probe measurement of electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.density(0), result);
         }
     },
     {"ni", "probe measurement of ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.density(1), result);
         }
     },
     {"ue", "probe measurement of parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.velocity(0), result);
         }
     },
     {"ui", "probe measurement of parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.velocity(1), result);
         }
     },
     {"phi", "probe measurement of electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.potential(0), result);
         }
     },
     {"apar", "probe measurement of parallel magnetic potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.aparallel(), result);
         }
     },
     {"neR", "probe measurement of d/dR electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(0)[0], result);
         }
     },
     {"niR", "probe measurement of d/dR ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(1)[0], result);
         }
     },
     {"ueR", "probe measurement of d/dR parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(0)[0], result);
         }
     },
     {"uiR", "probe measurement of d/dR parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(1)[0], result);
         }
     },
     {"phiR", "probe measurement of d/dR electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradP(0)[0], result);
         }
     },
     {"aparR", "probe measurement of d/dR parallel magnetic potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradA()[0], result);
         }
     },
     {"neZ", "probe measurement of d/dZ electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(0)[1], result);
         }
     },
     {"niZ", "probe measurement of d/dZ ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(1)[1], result);
         }
     },
     {"ueZ", "probe measurement of d/dZ parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(0)[1], result);
         }
     },
     {"uiZ", "probe measurement of d/dZ parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(1)[1], result);
         }
     },
     {"phiZ", "probe measurement of d/dZ electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradP(0)[1], result);
         }
     },
     {"aparZ", "probe measurement of d/dZ parallel magnetic potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradA()[1], result);
         }
     },
     {"nePar", "probe measurement of d/dPar electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsN(0), result);
         }
     },
     {"niPar", "probe measurement of d/dPar ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsN(1), result);
         }
     },
     {"uePar", "probe measurement of d/dPar parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsU(0), result);
         }
     },
     {"uiPar", "probe measurement of d/dPar parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsU(1), result);
         }
     },
     {"phiPar", "probe measurement of d/dPar electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsP(0), result);
         }
     }
 };
};

} //namespace feltor
