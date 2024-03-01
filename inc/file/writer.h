#pragma once

#include <functional>
#include <map>

#include "../dg/topology/fast_interpolation.h"

#include "easy_dims.h"
#include "easy_output.h"
#include "json_netcdf.h"


namespace dg
{
namespace file
{

///@addtogroup Cpp
///@{
/**
 * @class hide_param_ncid
 * @param ncid A reference to a NetCDF file or group ID. The user is responsible to ensure
 * that the referenced id is valid (i.e. the file is open) any time a member function of \c Writer is called.
 * The reason we need to store a reference is that the ncid of a file can change (e.g. by closing
 * and re-opening) https://docs.unidata.ucar.edu/netcdf-c/current/tutorial_ncids.html
 *
 * @class hide_tparam_topology
 * @tparam Topology Any topology or geometry class.
 * This class constructs and stores a member \c Topology::host_grid
 *
 * @class hide_dim_names_param
 * @param dim_names The dimension names in NetCDF ordering (i.e. reversed
 * compared to the ordering in the dg library). The last dim_name
 * names the \c x dimension in g. If <tt> g.ndim() == dim_names.size()
 * </tt>, all variables managed with the class are time
 * independent.  If <tt> g.ndim() == dim_names.size()+1</tt>
 * all variables are "time"-dependent (they have an unlimited dimension) and
 * the first itmem in \c dim_names names the unlimited dimension.
 * If \c g.ndim()==0, dim_names can be empty, which means that scalar
 * time-independent variables without dimensions are defined.
 * In other cases an error is thrown.
 *
 * @class hide_g_param
 * @param g Used to construct a \c Topology::host_grid. Determines the spatial shape and data type of
 * all variables this class writes to file
 */

/**
 * @brief Write variables of equal type and shape
 *
 * Manage dimensions and variable ids for writing
 *
 * This class provides a lightweight high-level interface to NetCDF
 * writing.
 * The idea is that this class manages variables that all share the same
 * dimensions. For each set of such variables a new instance of the class
 * should be created.
 * @note In an MPI program all processes must construct the class.
 * Also the \c parallel=false parameter is used in the \c dg::file::put_var and
 * \c dg::file::put_vara functions meaning only the master thread will write
 * to file.
 * @copydoc hide_tparam_topology
 * @note A 0d Writer can write the time dimension variable
 */
template<class Topology>
struct Writer
{
    Writer() = default;
    /**
     * @brief Consruct a Writer
     *
     * @copydoc hide_param_ncid
     * @copydoc hide_g_param
     * @copydoc hide_dim_names_param
     */
    Writer( const int& ncid, const Topology& g, std::vector<std::string> dim_names) :
        m_ncid( &ncid), m_grid( g)
    {
        m_dims.resize( dim_names.size());
        if( g.ndim() == dim_names.size())
        {
            dg::file::define_dimensions( ncid, &m_dims[0], m_grid, dim_names, true);
        }
        else if( g.ndim()+1 == dim_names.size())
        {
            int tvarID = 0;
            dg::file::define_dimensions( ncid, &m_dims[0], &tvarID, m_grid, dim_names, true);
            if( g.ndim() == 0) // enable time writing for 0d Writer
            {
                m_varids[dim_names[0]] = tvarID;
            }
        }
        else
            throw std::runtime_error( "Number of dimension names "+std::to_string(dim_names.size())+" must be either the same or one larger than grid dimensions "+std::to_string(g.ndim())+"\n");
    }

    /**
     * @brief Access the underlying shape of the variables
     *
     * @return The grid from the constructor
     */
    const typename Topology::host_grid grid() const{ return m_grid;}

    /**
     * @brief Define a variable named \c name with attributes \c atts
     *
     * Use the dimensions given in the constructor. If the name exists in the file
     * already, the associated dimensions are checked. If they do not match
     * an error is thrown. If they do the variable is managed by class
     * @note This beheviour can be used to open an existing NetCDF file and add
     * data to existing variables
     *
     * @param name the name of the variable to define.
     * @param atts [optional] define attributes together with variable using \c dg::file::json2nc_attrs
     */
    void def( std::string name, const dg::file::JsonType& atts = {})
    {
        // this enables time writing !
        if ( m_varids.find( name) == m_varids.end()) // we did not find it
        {
            do_def( name, atts);
        }
    }
    /**
     * @brief Write data for given variable
     *
     * @param name the name of the variable to write data for (must be
     * registered by a call to \c def prior to this function)
     * @copydoc hide_tparam_host_vector
     * @param data the data to write (must have grid size)
     * @param slice (ignored for time-independent variables). The number of the
     * time-slice to write (first element of the \c startp array in \c
     * nc_put_vara). It is the user's responsibility to keep track of how big
     * variables get and which slice to write.
     * @copydoc hide_comment_slice
     */
    template<class host_vector>
    void put( std::string name, const host_vector& data, unsigned slice=0) const
    {
        static_assert( std::is_same< dg::get_value_type<host_vector>, typename Topology::value_type>::value, "Grid and Host vector must have same value type");
        using tensor_category = dg::get_tensor_category<typename Topology::host_vector>;
        static_assert( dg::is_scalar_or_same_base_category<host_vector, tensor_category>::value, "Data type must have same Tensor category as Topology::host_vector");

        // it is possible to skip writes but not write beyond current_max + 1
        if( m_grid.ndim() == m_dims.size())
            dg::file::put_var( *m_ncid, m_varids.at(name), m_grid, data);
        else
            dg::file::put_vara( *m_ncid, m_varids.at(name), slice, m_grid, data);
    }
    /**
     * @brief Define and write a variable in one go
     *
     * Same as
     * @code
        def( name, atts);
        put( name, data, 0);
     * @endcode
     *
     * @param name the name of the variable to define.
     * @param atts define attributes together with variable using \c dg::file::json2nc_attrs
     * @copydoc hide_tparam_host_vector
     * @param data the data to write (must have grid size)
     */
    template<class host_vector>
    void def_and_put( std::string name, const dg::file::JsonType& atts, const host_vector& data)
    {
        def( name, atts);
        put( name, data, 0);
    }
    private:
#ifdef MPI_VERSION
    // Help SFINAE
    //https://stackoverflow.com/questions/11531989/what-happened-to-my-sfinae-redux-conditional-template-class-members
    template<class T = Topology>
    std::enable_if_t<dg::is_mpi_grid<T>::value,void >
         do_def( std::string name, const dg::file::JsonType& atts)
    {
        m_varids[name] = 0;// all processes are aware that variable exists
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank==0)
        {
            dg::file::NC_Error_Handle err;
            if (!variable_exists(name))
                err = nc_def_var( *m_ncid, name.data(), getNCDataType<typename Topology::value_type>(),
                    m_dims.size(), &m_dims[0], &m_varids.at(name));
            dg::file::json2nc_attrs( atts, *m_ncid, m_varids.at(name));
        }
    }
#endif // MPI_VERSION
    template<class T = Topology>
    std::enable_if_t<dg::is_shared_grid<T>::value,void >
        do_def( std::string name, const dg::file::JsonType& atts)
    {
        dg::file::NC_Error_Handle err;
        if ( !variable_exists( name))
            err = nc_def_var( *m_ncid, name.data(), getNCDataType<typename Topology::value_type>(),
                m_dims.size(), &m_dims[0], &m_varids[name]);
        dg::file::json2nc_attrs( atts, *m_ncid, m_varids.at(name));
    }
    bool variable_exists( std::string name)
    {
        int varid=0;
        int retval = nc_inq_varid( *m_ncid, name.data(), &varid);
        if( retval != NC_NOERR)
            return false; //variable does not exist
        int ndims;
        file::NC_Error_Handle err;
        err = nc_inq_varndims( *m_ncid, varid, &ndims);
        int dimIDs[ndims];
        int xtype;
        err = nc_inq_var( *m_ncid, varid, NULL, &xtype, NULL, dimIDs, NULL);

        if( ndims != (int)m_dims.size() ||
            xtype != getNCDataType<typename Topology::value_type>() )
        {
            throw std::runtime_error( "Variable "+name+" exists but has wrong dimension number or type!");
        }
        else
        {
            for( unsigned i=0; i<m_dims.size(); i++)
                if( m_dims[i] != dimIDs[i])
                    throw std::runtime_error( "Variable "+name+" exists but has wrong dimensions!");
        }
        m_varids[name] = varid;
        return true;
    }

    const int* m_ncid;
    std::vector<int> m_dims;
    std::map<std::string,int> m_varids;
    typename Topology::host_grid m_grid;
};


///@cond
namespace detail
{

template<class Signature>
struct get_first_argument_type;

template<class R, class Arg1, class ...A>
struct get_first_argument_type<R(Arg1, A...)>
{
    using type = Arg1;
};
}//namespace detail
///@endcond
/// If <tt> Signature = R(Arg1, A...)</tt> return \c Arg1
template<class Signature>
using get_first_argument_type_t = std::decay_t<typename detail::get_first_argument_type<Signature>::type>;

/// If <tt> Signature = R(Arg1, A...)</tt> return \c R
template<class Signature>
using get_result_type_t = typename std::function<Signature>::result_type;

/**
 * @brief A realisation of the %Record concept. Helper to generate NetCDF variables.
 *
 * Supposed to be used in connection with a Records writer like \c WriteRecordList
 * @snippet netcdf_t.cpp doxygen
   @tparam SignatureType Signature of the callable function
 */
template<class SignatureType>
struct Record
{
    using Signature = SignatureType; //!< Signature of the \c function
    std::string name; //!< Name of the variable to create
    std::string long_name; //!< Attribute "long_name" of the variable
    std::function<Signature> function; //!< The function to call that generates data for the variable
};

/**
 * @class hide_tparam_listclass
 * @tparam ListClass
 * A Type whose <tt> ListClass::value_type </tt> equals a %Record class (e.g. \c dg::file::Record)
 * The Signature <tt> ListClass::value_type::Signature </tt> must have either \c void as return type
 * or a primitive type. The latter indicates a scalar output and must coincide
 * with <tt> Topology::ndim() == 0</tt>. If the return type is void then the
 * **first argument type** must be a Vector type constructible from \c
 * Topology::host_vector e.g. a \c dg::DVec.
 */

/**
 * @brief A class to generate and write variables from a record list into a netcdf file
 *
 * The idea is that this class manages the \c slice parameter in the \c Write class
 * and essentially generates and writes a goup of variables that share type and shape in each timestep.
 *
 * @note in an MPI program all processes have to create the class and call its methods. The
 * class automatically takes care of which threads write to file.
 * @copydoc hide_tparam_topology
 */
template<class Topology>
struct WriteRecordsList
{
    WriteRecordsList() = default;
    /**
     * @brief Create variables ids
     *
     * For each record in \c records create a variable named \c record.name
     * with attribute \c record.long_name of dimensions \c dim_names with shape
     * given by \c g in group \c ncid
     * @copydoc hide_param_ncid
     * @copydoc hide_g_param
     * @copydoc hide_dim_names_param
     * @copydoc hide_tparam_listclass
     * @param records list of records to put into ncid
     */
    template<class ListClass>
    WriteRecordsList( const int& ncid, const Topology& g, std::vector<std::string> dim_names, const ListClass& records) : m_start(0), m_writer( ncid, g, dim_names)
    {
        for( auto& record : records)
        {
            m_writer.def( record.name, dg::file::long_name( record.long_name));
        }
    }

    /**
     * @brief Write variables created from record list
     *
     * There are two ways the function handles the \c records:
     *  - For each record in \c records call \c record.function( result, ps...)
     *  where \c result is of type <tt>
     *  get_first_argument_type_t<ListClass::value_type::Signature></tt>  of size given
     *  by \c grid. Finally, call \c dg::assign( resultD, resultH)
     *  i.e. assign to a host vector of type \c Topology::host_vector
     *  and write into \c ncid (from constructor).
     *  - If <tt> return_type<ListClass::value_type::Signature> != void </tt>
     *  then call <tt> auto result = record.function( ps...) </tt> and write
     *  directly into \c ncid
     *  .
     * @copydoc hide_tparam_listclass
     * @param records list of records to put into ncid
     * @tparam Params The \c ListClass::value_type::Signature without the first argument
     * @param ps Parameters forwarded to \c record.function( result, ps...) or \c result = record.function( ps...)
     */
    template< class ListClass, class ... Params >
    void write( const ListClass& records, Params&& ...ps)
    {
        do_write( records, std::forward<Params>(ps)...);
    }

    /**
     * @brief Write variables created from record list and transformed using custom operator
     *
     * This works only for vector results in \c records:
     *  - For each record in \c records call \c record.function( result, ps...)
     *  followed by \c dg::apply(func, result, resultD)
     *  where \c resultD is of type
     *  <tt> get_first_argument_type_t<ListClass::value_type::Signature></tt>
     *  of size given by \c grid. Finally, call \c dg::assign( resultD, resultH)
     *  i.e. assign to a host vector of type \c Topology::host_vector
     *  and write into \c ncid (from constructor).
     *  .
     * @tparam MatrixType See \c dg::apply
     * @param func Will be called though \c dg::apply( func, result, resultD)
     * @copydoc hide_tparam_listclass
     * @param records list of records to put into ncid
     * @tparam ContainerType Must equal <tt>get_first_argument_type_t<ListClass::value_type::Signature></tt>
     * @param result The first argument to \c record.function as well as the first argument
     * to func. By providing a workspace
     * the user may avoid the need to allocate space inside the \c record.function
     * @tparam Params The \c ListClass::value_type::Signature without the first argument
     * @param ps Parameters forwarded to \c record.function( result, ps...) or \c result = record.function( ps...)
     */
    template<class MatrixType, class ListClass, class ... Params>
    void transform_write( MatrixType&& func, const ListClass& records, get_first_argument_type_t<typename ListClass::value_type::Signature> result, Params&& ... ps)
    {
        auto resultH = dg::evaluate( dg::zero, m_writer.grid());
        //vector write
        auto transferD =
            dg::construct<get_first_argument_type_t<typename ListClass::value_type::Signature>>(
                resultH);
        for( auto& record : records)
        {
            record.function( result, std::forward<Params>(ps)...);
            dg::apply( std::forward<MatrixType>(func), result, transferD);
            dg::assign( transferD, resultH);
            m_writer.put( record.name, resultH, m_start);
        }
        m_start++;
    }
    /**
     * @brief Same as \c transform_write but the order of \c dg::assign and \c dg::apply is swapped
     *
     * i.e. \c result is first transfered to the host and then \c dg::apply is called.
     * This effectively means that func works on the host
     */
    template<class MatrixType, class ListClass, class ... Params>
    void host_transform_write( MatrixType&& func, const ListClass& records, get_first_argument_type_t<typename ListClass::value_type::Signature> result, Params&& ... ps)
    {
        auto resultH = dg::evaluate( dg::zero, m_writer.grid());
        auto transferH =
            dg::construct<typename Topology::host_vector>( result);
        for( auto& record : records)
        {
            record.function( result, std::forward<Params>(ps)...);
            dg::assign( result, transferH);
            dg::apply( std::forward<MatrixType>(func), transferH, resultH);
            m_writer.put( record.name, resultH, m_start);
        }
        m_start++;
    }
    private:
    template< class ListClass, class ... Params >
    std::enable_if_t<std::is_same<get_result_type_t<typename ListClass::value_type::Signature> ,void>::value >  do_write( const ListClass& records, Params&& ...ps)
    {
        auto resultH = dg::evaluate( dg::zero, m_writer.grid());
        //vector write
        auto resultD =
            dg::construct<get_first_argument_type_t<typename ListClass::value_type::Signature>>(
                resultH);
        for( auto& record : records)
        {
            record.function( resultD, std::forward<Params>(ps)...);
            dg::assign( resultD, resultH);
            m_writer.put( record.name, resultH, m_start);
        }
        m_start++;
    }
    template< class ListClass, class ... Params >
    std::enable_if_t<!std::is_same<get_result_type_t<typename ListClass::value_type::Signature> ,void>::value >  do_write( const ListClass& records, Params&& ...ps)
    {
        // scalar writes
        for( auto& record : records)
        {
            auto result = record.function( std::forward<Params>(ps)...);
            m_writer.put( record.name, result, m_start);
        }
        m_start++;
    }

    size_t m_start;
    Writer<Topology>  m_writer;
};

///@}


}//namespace file
}//namespace dg
