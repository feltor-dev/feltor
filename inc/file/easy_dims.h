#pragma once

#include <netcdf.h>
#include "thrust/host_vector.h"

#include "dg/runge_kutta.h" // for dg::is_same
#include "dg/topology/grid.h"
#include "dg/topology/gridX.h"
#include "dg/topology/evaluation.h"
#ifdef MPI_VERSION
#include "dg/topology/mpi_grid.h"
#include "dg/topology/mpi_evaluation.h"
#endif //MPI_VERSION

#include "nc_error.h"
#include "easy_atts.h"

/*!@file
 *
 * The define_dimensions functions
 */

namespace dg
{
namespace file
{

///@cond

namespace detail
{

template<class real_type>
std::vector<dg::RealGrid1d<real_type> > grids( const RealGrid0d<real_type>& ) { return {};}
template<class real_type>
std::vector<dg::RealGrid1d<real_type> > grids( const RealGrid1d<real_type>& g ) { return {g};}
template<class real_type>
std::vector<dg::RealGrid1d<real_type> > grids( const aRealTopology2d<real_type>& g ) { return {g.gy(), g.gx()};}
template<class real_type>
std::vector<dg::RealGrid1d<real_type> > grids( const aRealTopology3d<real_type>& g ) { return {g.gz(), g.gy(), g.gx()};}
template<class real_type>
std::vector<dg::RealGrid1d<real_type> > grids( const aRealTopologyX2d<real_type>& g ) { return grids(g.grid());}
template<class real_type>
std::vector<dg::RealGrid1d<real_type> > grids( const aRealTopologyX3d<real_type>& g ) { return grids(g.grid());;}

template<class real_type>
std::vector<std::string> dim_names( const RealGrid0d<real_type>& ) { return {};}
template<class real_type>
std::vector<std::string> dim_names( const RealGrid1d<real_type>& ) { return {"x"};}
template<class real_type>
std::vector<std::string> dim_names( const aRealTopology2d<real_type>& ) { return {"y", "x"};}
template<class real_type>
std::vector<std::string> dim_names( const aRealTopology3d<real_type>& ) { return {"z", "y", "x"};}
template<class real_type>
std::vector<std::string> dim_names( const aRealTopologyX2d<real_type>& ) { return {"y", "x"};}
template<class real_type>
std::vector<std::string> dim_names( const aRealTopologyX3d<real_type>& ) { return {"z", "y", "x"};}
template<class real_type>
std::vector<std::string> axis_names( const RealGrid0d<real_type>& ) { return {};}
template<class real_type>
std::vector<std::string> axis_names( const RealGrid1d<real_type>& ) { return {"X"};}
template<class real_type>
std::vector<std::string> axis_names( const aRealTopology2d<real_type>& ) { return {"Y", "X"};}
template<class real_type>
std::vector<std::string> axis_names( const aRealTopology3d<real_type>& ) { return {"Z", "Y", "X"};}
template<class real_type>
std::vector<std::string> axis_names( const aRealTopologyX2d<real_type>& ) { return {"Y", "X"};}
template<class real_type>
std::vector<std::string> axis_names( const aRealTopologyX3d<real_type>& ) { return {"Z", "Y", "X"};}


inline void assign_defaults( std::vector<std::string>& name_dims, const std::vector<std::string>& default_names)
{
    if( name_dims.empty())
        name_dims = default_names;
    if( name_dims.size() != default_names.size())
        throw std::runtime_error( "Number of given dimension names "+std::to_string(name_dims.size())+"does not match required number "+std::to_string(default_names.size())+"\n");
}
} // namespace detail
///@endcond

///@addtogroup legacy
///@{

/**
 * @brief DEPRECATED Check if an unlimited dimension exists as if \c define_real_time was called
 *
 * This function returns false if the dimension with the given name does not exist.
 *
 * This function throws \c std::runtime_error if
 *  - The dimension exists but is not unlimited
 *  - The dimension variable has wrong type or dimensions
 *  .
 *  This function throws an \c dg::file::NC_Error if
 *   - The dimension exists but the variable does not
 *   .
 * @note This function does not write anything to the file, only read
 * @param ncid NetCDF file or group ID
 * @param name Name of unlimited dimension and associated variable
 * @param dimID (write-only) time-dimension ID
 * @param tvarID (write-only) time-variable ID (for a time variable of type \c T)
 * @tparam T determine type of dimension variable
 * @return False if dimension with given name does not exist, if no errors are thrown True
 * @attention Dimensions in the parent group are visible in groups, but variables are not, so groups should have separate time dimension https://docs.unidata.ucar.edu/netcdf-c/current/group__groups.html
 */
template<class T>
bool check_real_time( int ncid, const char* name, int* dimID, int* tvarID)
{
    // TODO Axis attribute check is still missing
    file::NC_Error_Handle err;
    // Check that it exists
    int retval = nc_inq_dimid( ncid, name, dimID);
    if( retval)
        return false;
    // Check that it is unlimited
    int num_unlim;
    err = nc_inq_unlimdims( ncid, &num_unlim, NULL);
    std::vector<int> unlim_dims(num_unlim);
    err = nc_inq_unlimdims( ncid, &num_unlim, &unlim_dims[0]);
    if( std::find( unlim_dims.begin(), unlim_dims.end(), *dimID) == std::end( unlim_dims))
        throw std::runtime_error( "Dimension "+std::string(name)+" already defined but not unlimited!\n");
    //// Check the length
    //size_t length;
    //err = nc_inq_dimlen( ncid, *dimID, &length);
    //if( length != 0)
    //    throw std::runtime_error( "Unlimited dimension "+std::string(name)+" already has values!\n");
    // Now check if the dimension variable exists already
    err = nc_inq_varid( ncid, name, tvarID);
    int xtype;
    int ndims;
    int dimids[10];
    err = nc_inq_var( ncid, *tvarID, NULL, &xtype, &ndims, dimids, NULL);
    if( xtype != detail::getNCDataType<T>() || ndims != 1 || dimids[0] != *dimID)
        throw std::runtime_error( "Unlimited variable "+std::string(name)+" has wrong type or wrong dimensions!\n");
    return true;
}

/**
 * @brief DEPRECATED Define an unlimited time dimension and coordinate variable
 *
 * @note By <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">NetCDF conventions</a>
 * a variable with the same name as a dimension is
 * called a coordinate variable.  The CF conventions dictate that the "units"
 * attribute must be defined for a time variable: we give it the value "time
 * since start". Furthermore, we define the "axis" : "T" attribute to mark
 * the time dimension.
 * @param ncid NetCDF file or group ID
 * @param name Name of unlimited dimension and associated variable
 * @param dimID (write-only) time-dimension ID
 * @param tvarID (write-only) time-variable ID (for a time variable of type \c T)
 * @tparam T determine type of dimension variable
 * @param full_check If true, will call \c check_real_time before definition.
 *
 * @return NetCDF error code if any
 */
template<class T>
inline int define_real_time( int ncid, const char* name, int* dimID, int* tvarID, bool full_check = false)
{
    if( full_check)
        if( check_real_time<T>( ncid, name, dimID, tvarID))
            return NC_NOERR;
    int retval;
    if( (retval = nc_def_dim( ncid, name, NC_UNLIMITED, dimID)) ){ return retval;}
    if( (retval = nc_def_var( ncid, name, detail::getNCDataType<T>(), 1, dimID, tvarID))){return retval;}
    std::string t = "time since start"; //needed for paraview to recognize timeaxis
    // Update: Actually paraview also recognizes time from the "T" "axis" without "unit"
    std::string axis = "T";
    if( (retval = nc_put_att_text(ncid, *tvarID, "axis", axis.size(), axis.data())) ){return retval;}
    if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
    return retval;
}

/// DEPRECATED An alias for <tt> define_real_time<double> </tt>
inline int define_time( int ncid, const char* name, int* dimID, int* tvarID)
{
    return define_real_time<double>( ncid, name, dimID, tvarID);
}


/**
 * @brief DEPRECATED Define a limited time dimension and coordinate variable
 *
 * @note By <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">NetCDF conventions</a>
 * a variable with the same name as a dimension is
 * called a coordinate variable.  The CF conventions dictate that the units
 * attribute must be defined for a time variable: we give it the value "time
 * since start". Furthermore, we define the "axis" : "T" attribute to mark
 * the time dimension.
 * @param ncid NetCDF file or group ID
 * @param name Name of the time variable (usually "time")
 * @param size The number of timesteps
 * @param dimID time-dimension ID
 * @param tvarID time-variable ID (for a time variable of type \c NC_DOUBLE)
 *
 * @return NetCDF error code if any
 */
inline int define_limited_time( int ncid, const char* name, int size, int* dimID, int* tvarID)
{
    int retval;
    if( (retval = nc_def_dim( ncid, name, size, dimID)) ){ return retval;}
    if( (retval = nc_def_var( ncid, name, NC_DOUBLE, 1, dimID, tvarID))){return retval;}
    std::string t = "time since start"; //needed for paraview to recognize timeaxis
    std::string axis = "T";
    if( (retval = nc_put_att_text(ncid, *tvarID, "axis", axis.size(), axis.data())) ){return retval;}
    if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
    return retval;
}

/**
 * @brief DEPRECATED Check if a dimension exists as if \c define_dimension was called
 *
 * This function returns false if the dimension with the given name does not exist.
 *
 * This function throws \c std::runtime_error if
 *  - The length of the dimension does not match the grid size
 *  - The dimension variable has wrong type or dimensions
 *  - The dimension variable entries do not match the grid abscissas
 *  .
 *  This function throws an \c dg::file::NC_Error if
 *   - The dimension exists but the variable does not
 *   - The dimension variable has no entries
 *   .
 * @note This function does not write anything to the file, only read
 * @param ncid NetCDF file or group ID
 * @param dimID dimension ID (output)
 * @param g The 1d DG grid from which data points for coordinate variable are generated using \c g.abscissas()
 * @param name_dim Name of dimension and coordinate variable (input)
 * [unnamed string] axis The axis attribute (input) is ignored but kept for now as a future placeholder
 * @tparam T determines the datatype of the dimension variables
 * @return False if dimension with given name does not exist, if no errors are thrown True
 */
template<class T>
bool check_dimension( int ncid, int* dimID, const dg::RealGrid1d<T>& g, std::string name_dim = "x", std::string = "X")
{
    // TODO Axis attribute check is still missing
    file::NC_Error_Handle err;
    // check if the dimension exists already:
    int retval = nc_inq_dimid( ncid, name_dim.data(), dimID);
    if( retval)
        return false;
    size_t length;
    retval = nc_inq_dimlen( ncid, *dimID, &length);
    thrust::host_vector<T> points = g.abscissas();
    if( length != points.size())
        throw std::runtime_error( "Length of dimension "+name_dim+" does not match grid!\n");
    // Now check if the dimension variable exists already
    int varID;
    err = nc_inq_varid( ncid, name_dim.data(), &varID);
    int xtype;
    int ndims;
    int dimids[10];
    err = nc_inq_var( ncid, varID, NULL, &xtype, &ndims, dimids, NULL);
    if( xtype != detail::getNCDataType<T>() || ndims != 1 || dimids[0] != *dimID)
        throw std::runtime_error( "Dimension variable "+name_dim+" has wrong type or wrong dimensions!\n");
    thrust::host_vector<T> data( points);
    err = nc_get_var( ncid, varID, data.data());
    for( unsigned i=0; i<data.size(); i++)
        if( !dg::is_same( data[i], points[i]))
            throw std::runtime_error( "Dimension variable "+name_dim+" has values different from grid!\n");
    return true;
}

/**
 * @brief DEPRECATED Define a 1d dimension and associated coordinate variable
 *
 * @note By <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">NetCDF conventions</a>
 * a variable with the same name as a dimension is
 * called a coordinate variable.
 * @param ncid NetCDF file or group ID
 * @param dimID dimension ID (output)
 * @param g The 1d DG grid from which data points for coordinate variable are generated using \c g.abscissas()
 * @param name_dim Name of dimension and coordinate variable (input)
 * @param axis The axis attribute (input), ("X", "Y" or "Z")
 * @tparam T determines the datatype of the dimension variables
 * @param full_check If true, will call \c check_dimension before definition.
 *
 * @return NetCDF error code if any
 */
template<class T>
inline int define_dimension( int ncid, int* dimID, const dg::RealGrid1d<T>& g, std::string name_dim = "x", std::string axis = "X", bool full_check = false)
{
    if( full_check)
        if( check_dimension( ncid, dimID, g, name_dim, axis))
            return NC_NOERR;
    int retval;
    std::string long_name = name_dim+"-coordinate in Computational coordinate system";
    thrust::host_vector<T> points = g.abscissas();
    if( (retval = nc_def_dim( ncid, name_dim.data(), points.size(), dimID))){ return retval;}
    int varID;
    if( (retval = nc_def_var( ncid, name_dim.data(), detail::getNCDataType<T>(), 1, dimID, &varID))){return retval;}
    if( (retval = nc_put_var( ncid, varID, points.data())) ){ return retval;}
    if( (retval = nc_put_att_text( ncid, varID, "axis", axis.size(), axis.data())) ){return retval;}
    retval = nc_put_att_text( ncid, varID, "long_name", long_name.size(), long_name.data());
    return retval;
}

/**
 * @brief DEPRECATED Define dimensions and associated coordiante variables
 *
 * @note By <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">NetCDF conventions</a>
 * a variable with the same name as a dimension is
 * called a coordinate variable.
 *
 * @param ncid NetCDF file or group ID
 * @param dimsIDs (write - only) dimension IDs, Must be of size <tt> g.ndim() </tt>
 * @param g The dG grid from which data points for coordinate variable are generated using \c g.abscissas() in each dimension
 * @param name_dims Names for the dimension and coordinate variables (Must have size <tt> g.ndim() </tt>)  **in numpy python ordering** e.g. in 3d we have <tt> {"z", "y", "x"}</tt>; If \c name_dims.empty() then default names in <tt> {"z", "y", "x"} </tt> will be used
 * @tparam Topology <tt> typename Topology::value_type </tt> determines the datatype of the dimension variables
 * @note For a 0d grid, the function does nothing
 * @param full_check If true, will call \c check_dimensions before definition.
 * In this case dimensions can already exist in the file and will not trigger a
 * throw (it is also possible for some dimensions to exist while other do not)
 *
 * @return if anything goes wrong, return the NetCDF error code, else NC_NOERR
 */
template<class Topology, std::enable_if_t<dg::is_vector_v<typename Topology::host_vector, dg::SharedVectorTag>,bool > = true>
int define_dimensions( int ncid, int *dimsIDs, const Topology& g, std::vector<std::string> name_dims = {}, bool full_check = false)
{
    int retval = NC_NOERR;
    auto grids = detail::grids( g);
    auto default_names = detail::dim_names(g);
    auto axis_names = detail::axis_names(g);
    detail::assign_defaults( name_dims, default_names);
    for ( unsigned i=0; i<grids.size(); i++)
    {
        retval = define_dimension( ncid, &dimsIDs[i], grids[i], name_dims[i], axis_names[i], full_check);
        if( retval)
            return retval;
    }
    return retval;
}

/**
 * @brief DEPRECATED Define an unlimited time and grid dimensions together with their coordinate variables
 *
 * @note By <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">NetCDF conventions</a>
 * a variable with the same name as a dimension is
 * called a coordinate variable.
 *
 * Semantically equivalent to the following:
 * @code
    retval = define_real_time<typename Topology::value_type>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID);
    if(retval)
        return retval;
    return define_dimensions( ncid, &dimsIDs[1], g, {name_dims.begin()+1, name_dims.end()});
 * @endcode
 *
 * @param ncid NetCDF file or group ID
 * @param dimsIDs (write - only) dimension IDs, Must be of size <tt> g.ndim()+1 </tt>
 * @param tvarID (write - only) time coordinate variable ID (unlimited)
 * @param g The dG grid from which data points for coordinate variable are generated using \c g.abscissas()
 * @param name_dims Names for the dimension and coordinate variables (Must have size <tt> g.ndim()+1 </tt>)  **in numpy python ordering** e.g. in 3d we have <tt> {"time", "z", "y", "x"}</tt>; If \c name_dims.empty() then default names in <tt> {"time", "z", "y", "x"} </tt> will be used
 * @tparam Topology <tt> typename Topology::value_type </tt> determines the datatype of the dimension variables
 * @note For 0d grids only the "time" dimension is defined, no spatial dimension
 * @param full_check If true, will call \c check_dimensions before definition.
 * In this case dimensions can already exist in the file and will not trigger a
 * throw (it is also possible for some dimensions to exist while other do not)
 *
 * @return if anything goes wrong, return the NetCDF error code, else NC_NOERR
 */
template<class Topology, std::enable_if_t<dg::is_vector_v<typename Topology::host_vector, dg::SharedVectorTag>,bool > = true>
int define_dimensions( int ncid, int *dimsIDs, int* tvarID, const Topology& g, std::vector<std::string> name_dims = {}, bool full_check = false)
{
    auto default_names = detail::dim_names(g);
    default_names.insert( default_names.begin(), "time");
    detail::assign_defaults( name_dims, default_names);

    int retval = define_real_time<typename Topology::value_type>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID, full_check);
    if(retval)
        return retval;
    return define_dimensions( ncid, &dimsIDs[1], g, {name_dims.begin()+1, name_dims.end()}, full_check);
}

/**
 * @brief DEPRECATED Check if dimensions exist as if \c define_dimensions was called
 *
 * This function checks if the given file contains dimensions and their associated dimension variables
 * in the same way that the corresponding \c define_dimensions creates them. If anything is amiss, an error
 * will be thrown.
 * @note In order to do this the function will actually read in the coordinate
 * variable and compare to the given grid abscissas
 * @param ncid NetCDF file or group ID
 * @param dimsIDs (write - only) dimension IDs, Must be of size <tt> g.ndim() </tt>
 * @param g The dG grid from which data points for coordinate variable are generated using \c g.abscissas() in each dimension
 * @param name_dims Names for the dimension and coordinate variables (Must have size <tt> g.ndim() </tt>)  **in numpy python ordering** e.g. in 3d we have <tt> {"z", "y", "x"}</tt>; If \c name_dims.empty() then default names in <tt> {"z", "y", "x"} </tt> will be used
 * @tparam Topology <tt> typename Topology::value_type </tt> determines the datatype of the dimension variables
 * @note For a 0d grid, the default dimension name is "i", axis "I" and the dimension will be of size 1
 * @return False if any dimension with given name does not exist, if no errors are thrown True
 * @sa check_dimension
 */
template<class Topology, std::enable_if_t<dg::is_vector_v<typename Topology::host_vector, dg::SharedVectorTag>,bool > = true>
bool check_dimensions( int ncid, int *dimsIDs, const Topology& g, std::vector<std::string> name_dims = {})
{
    auto grids = detail::grids( g);
    auto default_names = detail::dim_names(g);
    auto axis_names = detail::axis_names(g);
    detail::assign_defaults( name_dims, default_names);
    for ( unsigned i=0; i<grids.size(); i++)
    {
        if (!check_dimension( ncid, &dimsIDs[i], grids[i], name_dims[i], axis_names[i]))
            return false;
    }
    return true;
}

/**
 * @brief DEPRECATED Check if dimensions exist as if \c define_dimensions was called
 *
 * Semantically equivalent to the following:
 * @code
    if ( !check_real_time<typename Topology::value_type>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID))
        return false;
    return check_dimensions( ncid, &dimsIDs[1], g, {name_dims.begin()+1, name_dims.end()});
 * @endcode
 * This function checks if the given file contains dimensions and their associated dimension variables
 * in the same way that the corresponding \c define_dimensions creates them. If anything is amiss, an error
 * will be thrown.
 * @note In order to do this the function will actually read in the coordinate
 * variable and compare to the given grid abscissas
 * @param ncid NetCDF file or group ID
 * @param dimsIDs (write - only) dimension IDs, Must be of size <tt> g.ndim()+1 </tt>
 * @param tvarID (write - only) time coordinate variable ID (unlimited)
 * @param g The dG grid from which data points for coordinate variable are generated using \c g.abscissas()
 * @param name_dims Names for the dimension and coordinate variables (Must have size <tt> g.ndim()+1 </tt>)  **in numpy python ordering** e.g. in 3d we have <tt> {"time", "z", "y", "x"}</tt>; If \c name_dims.empty() then default names in <tt> {"time", "z", "y", "x"} </tt> will be used
 * @tparam Topology <tt> typename Topology::value_type </tt> determines the datatype of the dimension variables
 * @return False if any dimension with given name does not exist, if no errors are thrown True
 * @sa check_dimension check_real_time
 */
template<class Topology, std::enable_if_t<dg::is_vector_v<typename Topology::host_vector, dg::SharedVectorTag>,bool > = true>
bool check_dimensions( int ncid, int *dimsIDs, int* tvarID, const Topology& g, std::vector<std::string> name_dims = {})
{
    auto default_names = detail::dim_names(g);
    default_names.insert( default_names.begin(), "time");
    detail::assign_defaults( name_dims, default_names);
    if ( !check_real_time<typename Topology::value_type>( ncid, name_dims[0].data(), &dimsIDs[0], tvarID))
        return false;
    return check_dimensions( ncid, &dimsIDs[1], g, {name_dims.begin()+1, name_dims.end()});
}




/**
 * @brief DEPRECATED Define a limited time and 2 dimensions and associated coordinate variables
 *
 * @note By <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">NetCDF conventions</a>
 * a variable with the same name as a dimension is
 * called a coordinate variable.
 *
 * Semantically equivalent to the following:
 * @code
 * define_limited_time( ncid, name_dims[0], size, &dimsIDs[0], tvarID);
 * define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2]});
 * @endcode
 * Dimensions have attributes of (time, Y, X)
 * @param ncid NetCDF file or group ID
 * @param dimsIDs (write - only) 3D array of dimension IDs (time, Y,X)
 * @param size The size of the time variable
 * @param tvarID (write - only) The ID of the time variable (limited)
 * @param g The 2d DG grid from which data points for coordinate variable are generated using \c g.abscissas()
 * @param name_dims Names for the dimension variables (time, Y, X)
 * @tparam T determines the datatype of the dimension variables
 *
 * @return if anything goes wrong it returns the NetCDF code, else SUCCESS
 * @note File stays in define mode
 */
template<class T>
inline int define_limtime_xy( int ncid, int* dimsIDs, int size, int* tvarID, const dg::aRealTopology2d<T>& g, std::vector<std::string> name_dims = {"time", "y", "x"})
{
    int retval;
    retval = define_limited_time( ncid, name_dims[0].data(), size, &dimsIDs[0], tvarID);
    if(retval)
        return retval;
    return define_dimensions( ncid, &dimsIDs[1], g, {name_dims[1], name_dims[2]});
}


#ifdef MPI_VERSION
/// DEPRECATED All processes may call this but only master process has to and will execute!! Convenience function that just calls the corresponding serial version with the global grid.
template<class MPITopology, std::enable_if_t<dg::is_vector_v<typename MPITopology::host_vector, dg::MPIVectorTag>,bool > = true>
inline int define_dimensions( int ncid, int* dimsIDs, const MPITopology& g, std::vector<std::string> name_dims = {}, bool full_check = false)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank==0) return define_dimensions( ncid, dimsIDs, g.global(), name_dims, full_check);
    else
        return NC_NOERR;
}
/// DEPRECATED All processes may call this but only master process has to and will execute!! Convenience function that just calls the corresponding serial version with the global grid.
template<class MPITopology, std::enable_if_t<dg::is_vector_v<typename MPITopology::host_vector, dg::MPIVectorTag>,bool > = true>
inline int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const MPITopology& g, std::vector<std::string> name_dims = {}, bool full_check = false)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank==0) return define_dimensions( ncid, dimsIDs, tvarID, g.global(), name_dims, full_check);
    else
        return NC_NOERR;
}
/// DEPRECATED All processes may call this and all will execute!! Convenience function that just calls the corresponding serial version with the global grid.
template<class MPITopology, std::enable_if_t<dg::is_vector_v<typename MPITopology::host_vector, dg::MPIVectorTag>,bool > = true>
inline bool check_dimensions( int ncid, int* dimsIDs, const MPITopology& g, std::vector<std::string> name_dims = {})
{
    // all processes can read NetCDF in parallel by default
    return check_dimensions( ncid, dimsIDs, g.global(), name_dims);
}
/// DEPRECATED All processes may call this and all will execute!! Convenience function that just calls the corresponding serial version with the global grid.
template<class MPITopology, std::enable_if_t<dg::is_vector_v<typename MPITopology::host_vector, dg::MPIVectorTag>,bool > = true>
inline bool check_dimensions( int ncid, int* dimsIDs, int* tvarID, const MPITopology& g, std::vector<std::string> name_dims = {})
{
    // all processes can read NetCDF in parallel by default
    return check_dimensions( ncid, dimsIDs, tvarID, g.global(), name_dims);
}
#endif //MPI_VERSION

///@}
} //namespace file
} //namespace dg
