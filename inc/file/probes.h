#pragma once

#include "nc_utilities.h"
#include "json_utilities.h"
#ifdef MPI_VERSION
#include "dg/topology/mpi_projection.h"
#endif //MPI_VERSION

namespace dg
{
namespace file
{

///@cond
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
///@endcond
//

/**
 * @brief Parse probe field in json file for use with Probes class
 *
 * A synthetic diagnostics in FELTOR is equivalent to outputting the
 * computational fields including their first derivatives in time interpolated
 * to any grid location (as if a measurement was done at that location). The
 * output frequency is higher than the output frequency of the entire
 * computation fields (otherwise you could just interpolate those at the end of
 * the simulation).
 *
 * In the input file, it is required to define the interpolation coordinates
 * named \c coords-names (in this example "R", "Z" and "P") as arrays.
 * The length of the position arrays must match each other.
 * There is no limit on the size of the arrays; they are typically not performance
 * relevant unless a large percentage of actual grid coordinates is reached
@code
"probes" :
{
    "input" : "coords",
    "coords" :
    {
        "format" : format, // see paragraph below
        "coords-names" : ["R","Z","P"], // name of coordinates ( need to be in order passed to interpolation function)
        "R": [90, 95, 100], // R coordinates in rho\_s
        "Z": [0, 0, 0], // Z coordinates in rho\_s
        "P": [0, 0, 3] // phi coordinates in radian (values outside the interval
        // $[0,2\pi]$ will be taken modulo $2\pi$ (unsigned))
    }
}
@endcode
 * @note
 *  By default the "probes" input field is optional and can be left away entirely.
 *  No probes will be written into the output file then.  Be
 *  sure not to have any spelling mistakes on "probes" if you do want them though.
 *
 * Alternatively the "R", "Z" and "P" fields can be read from an external json file
 * \begin{minted}[texcomments]{js}
 * "probes": "path/to/probes.json"
 * \end{minted}
 * \begin{tcolorbox}[title=Units of $R$ and $Z$]
 * Similar to the magnetic parameters, in this case the "R" and "Z" values are
 * assume to have unit "meter".
 * In that case the "physical" field (described in Section~\ref{sec:physical}) needs
 * to contain the field "rho\_s" where $\rho_s$ in meters is given which will be
 * used to convert $R$, $Z$ from meter to $\rho_s$.
 * \end{tcolorbox}
 * \paragraph{format}
 * All measurements from points, lines, surfaces and volumes with different
 * purposes and different diagnostics, must be concatenated and flattened into the
 * one-dimensional "R", "Z", "P" arrays and the measurements are written to file
 * as one-dimensional arrays.  In this way the book-keeping "which point belong
 * to which diagnostics and is neighbor to which other point" may become
 * challening. This is why the "format" field exists.
 *
 * The format value is a user-defined json value that is ignored by feltor and
 * copied "as-is" as a string attribute to the probes group in the output file.
 * Its purpose is to hold parsing information for the (flat) $R$, $Z$, $P$ arrays
 * for post-processing. For example
 * \begin{minted}[texcomments]{js}
 * "format" : [
 * {"name" : "x-probe", "pos" : [0,10], "shape" : [10]},
 * {"name" : "omp", "pos" : [10,1010], "shape" : [10,10,10]}
 * ]
 * \end{minted}
 * interprets the first ten points in the probes array as a linear "x-probe" line,
 * while the remaining 1000 points belong to a 3d measurement volume called "omp".
 * From this information e.g. array views can be easily created in python:
 * \begin{minted}[texcomments]{py}
 * named_arr = dict()
 * for f in format:
 *     named_arr[f["name"]] = arr[f["pos"][0]:f["pos"][1]].reshape( f["shape"])
 * \end{minted}
 *
 * @attention In MPI only the master thread will read in the probes the others
 * return empty vectors
 */
struct ProbesParams
{
    std::vector< dg::HVec> coords;
    std::vector<std::string> coords_names;
    std::string format;
    bool probes = false; // indicates if coords are empty or "probes" field did not
                 // exist (all MPI processes must agree)

    ProbesParams() = default;
    // err says what to do if "probes" is missing (overwrites js error mode)
    ProbesParams( const dg::file::WrappedJsonValue& js, enum error probes_err = file::error::is_silent
            )
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif // MPI_VERSION
        if( probes_err == file::error::is_silent && !js.isMember( "probes"))
            return;
        else if( probes_err == file::error::is_warning && !js.isMember( "probes"))
        {
            DG_RANK0 std::cerr << "WARNING: probes field not found.  No probes written to file!\n";
            return;
        }
        else
            throw std::runtime_error( "\"probes\" field not found!");

        // test if parameters are file or direct
        auto probes_params = js["probes"];
        std::string type = probes_params["input"].asString();
        if( type == "file")
        {
            std::string path = probes_params["file"].asString();

            probes_params.asJson()["coords"] = dg::file::file2Json( path,
                    dg::file::comments::are_discarded, dg::file::error::is_throw);
        }
        else if( type != "coords")
        {
            throw std::runtime_error( "Error: Unknown magnetic field input '"
                   + type + "'.");
        }

        auto js_probes = probes_params["coords"];

        // read in parameters

        unsigned ndim = js_probes["coords-names"].size();

        std::string first = js_probes["coords-names"][0].asString();
        std::vector< double> scale;
        for( unsigned i=0; i<ndim; i++)
        {
            coords_names[i] = js_probes["coords-names"][i].asString();
            coords[i] = dg::HVec();
            scale[i] = 1.;
            if( type == "file")
                scale[i] = js_probes["scale"][i].asDouble();
        }
        unsigned num_pins = get_coords_sizes();
        format = js_probes["format"].toStyledString();
        probes = (num_pins > 0);

#ifdef MPI_VERSION
        if( rank == 0)
        {
        // only master thread reads probes
#endif  //MPI_VERSION
        for( unsigned i=0; i<ndim; i++)
        {
            coords[i].resize(num_pins);
            for( unsigned k=0; k<num_pins; k++)
                coords[i][k] = js_probes.asJson()[coords_names[i]][k].asDouble()
                    *scale[i];
        }
#ifdef MPI_VERSION
        }
#endif //MPI_VERSION
    }
    unsigned get_coords_sizes( ) const
    {
        unsigned m_num_pins = coords[0].size();
        for( unsigned i=1; i<coords.size(); i++)
        {
            unsigned num_pins = coords[i].size();
            if( m_num_pins != num_pins)
                throw std::runtime_error( "Size of "+coords_names[i] +" probes array ("
                        +std::to_string(num_pins)+") does not match that of "+coords_names[0]+" ("
                        +std::to_string(m_num_pins)+")!");
        }
        return m_num_pins;
    }
};



/**
 * @brief Facilitate output at selected points
 *
 * This class is a high level synthetic diagnostics package
 */
struct Probes
{
    Probes() = default;
    // if coords[i] are empty then all functions simply return immediately only master threads coords count
    template<class Geometry, class ListClass>
    Probes(
        int ncid,
        unsigned itstp,
        const Geometry& grid,
        const ProbesParams& params, // do nothing if probes is false
        const ListClass& probe_list
        )
    {
        m_probes = params.probes;
        if( !params.probes) return;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION

        if ( params.coords.size() != grid.ndim())
            throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in coords!");
        m_num_pins = params.get_coords_sizes();

        m_probe_interpolate = detail::CreateInterpolation<Geometry,
                            Geometry::ndim()>().call( params.coords, grid);
        // Create helper storage probe variables
#ifdef MPI_VERSION
        if(rank==0) m_simple_probes = dg::MHVec(params.coords[0],
                grid.communicator());
#else //MPI_VERSION
        m_simple_probes = dg::HVec(m_num_pins);
#endif
        for( auto& record : probe_list)
            m_simple_probes_intern[record.name] = std::vector<dg::x::HVec>(itstp, m_simple_probes);
        m_time_intern.resize(itstp);
        m_resultD = dg::evaluate( dg::zero, grid);
        m_resultH = dg::evaluate( dg::zero, grid);
        define_nc_variables( ncid, params.format, params.coords, params.coords_names,
                probe_list);
    }

    // record.name, record.long_name, record.function( resultH, ps...)
    template<class ListClass, class ...Params>
    void static_write( const ListClass& diag_static_list, Params&& ... ps)
    {
        if(!m_probes) return;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
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
#ifdef MPI_VERSION
                m_simple_probes.data().data()
#else
                m_simple_probes.data()
#endif
                    );
        }
    }

    // record.name, record.long_name, record.function( resultH, ps...)
    template<class ListClass, class ...Params>
    void write( double time, const ListClass& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        dg::file::NC_Error_Handle err;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        /// Probes FIRST output ///
        size_t probe_count[] = {1, m_num_pins};
        DG_RANK0 err = nc_put_vara_double( m_probe_grp_id, m_probe_timevarID,
                &m_probe_start[0], &probe_count[0], &time);
        for( auto& record : probe_list)
        {
            record.function( m_resultD, std::forward<Params>(ps)...);
            dg::assign( m_resultD, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                    m_probe_id_field.at(record.name), m_probe_start, probe_count,
#ifdef MPI_VERSION
                    m_simple_probes.data().data()
#else
                    m_simple_probes.data()
#endif
            );
        }
         /// End probes output ///

    }
    // is thought to be called itstp times before write
    template<class ListClass, class ...Params>
    void save( double time, unsigned iter, const ListClass& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        for( auto& record : probe_list)
        {
            record.function( m_resultD, std::forward<Params>(ps)...);
            dg::assign( m_resultD, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            m_simple_probes_intern.at(record.name)[iter]=m_simple_probes;
            m_time_intern[iter]=time;
        }
    }
    void write_after_save()
    {
        if( !m_probes) return;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        size_t probe_count[] = {1, m_num_pins};
        //OUTPUT OF PROBES
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
#ifdef MPI_VERSION
                        field.second[j].data().data()
#else
                        field.second[j].data()
#endif
                        );
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

    template<class ListClass>
    void define_nc_variables( int ncid, const std::string& format,
        const std::vector<dg::HVec>& coords,
        const std::vector<std::string> & coords_names,
        const ListClass& probe_list)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        dg::file::NC_Error_Handle err;
        DG_RANK0 err = nc_def_grp(ncid,"probes",&m_probe_grp_id);
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

} //namespace file
}//namespace dg
