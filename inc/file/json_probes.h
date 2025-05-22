#pragma once

#include "json_wrapper.h"
#include "probes_params.h"
/*!@file
 *
 * json parser for probes
 */

namespace dg
{
namespace file
{

/**
 * @brief Parse probe field in json file for use with Probes class
 *
 * A synthetic diagnostics in FELTOR is equivalent to outputting the
 * computational fields including their first derivatives in time interpolated
 * to any grid location (as if a measurement was done at that location). The
 * output frequency is typically higher than the output frequency of the entire
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
The "coords" field can be read from an external json file alternatively using
@code
"probes" :
{
    "input" : "file",
    "file" : "path/to/file.json", // relative to where the program is executed from
    "scale" : [1000,1000,1] // convert coords from SI units to dimenless units
}
@endcode

 * @note
 *  By default the "probes" input field is optional and can be left away entirely.
 *  No probes will be written into the output file then.  Be
 *  sure not to have any spelling mistakes on "probes" if you do want them though.
 *
 * All measurements from points, lines, surfaces and volumes with different
 * purposes and different diagnostics, must be concatenated and flattened into the
 * one-dimensional coordinate arrays and the measurements are written to file
 * as one-dimensional arrays.  In this way the book-keeping "which point belongs
 * to which diagnostics and is neighbor to which other point" may become
 * challening. This is why the "format" field exists.
 *
 * The format value is a user-defined json value that is ignored by feltor,
 * converted to a styled string and then stored as an attribute to the probes
 * group in the output file.  Its purpose is to hold parsing information for
 * the (flat) \f$ R \f$, \f$ Z \f$, \f$ P \f$ arrays for post-processing. For example
@code
"format" : [
{"name" : "x-probe", "pos" : [0,10], "shape" : [10]},
{"name" : "omp", "pos" : [10,1010], "shape" : [10,10,10]}
]
@endcode
 * interprets the first ten points in the probes array as a linear "x-probe" line,
 * while the remaining 1000 points belong to a 3d measurement volume called "omp".
 * From this information e.g. array views can be easily created in python:
@code
named_arr = dict()
for f in format:
    named_arr[f["name"]] = arr[f["pos"][0]:f["pos"][1]].reshape( f["shape"])
@endcode
 *
 * @param js input json value
 * @param probes_err what to do if "probes" is missing from \c js (overwrites js error mode)
 * If silent, the ProbesParams remain empty if the field is absent
 * @return parsed values
 * @attention In MPI all threads will read in the probes. Only the master thread
 * stores the coordinates in <tt>ProbesParams.coords[i]</tt> the others are empty
 * (and thus also the <tt>ProbesParams.get_coords_sizes()</tt> function will
 * return zero on non-master ranks)
 * @ingroup probes
*/
inline ProbesParams parse_probes( const dg::file::WrappedJsonValue& js, enum error
    probes_err = file::error::is_silent)
{
    ProbesParams out;
    int rank = 0;
#ifdef MPI_VERSION
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif // MPI_VERSION
    if( (probes_err == file::error::is_silent) && !js.isMember( "probes"))
        return out;
    else if( (probes_err == file::error::is_warning) && !js.isMember( "probes"))
    {
        if(rank==0) std::cerr << "WARNING: probes field not found.  No probes written to file!\n";
        return out;
    }
    else if ( !js.isMember("probes"))
        throw std::runtime_error( "\"probes\" field not found!");

    // test if parameters are file or direct
    auto jsprobes = js["probes"];
    std::string type = jsprobes["input"].asString();
    if( type == "file")
    {
        std::string path = jsprobes["file"].asString();
        // everyone reads the file
        jsprobes.asJson()["coords"] = dg::file::file2Json( path,
                dg::file::comments::are_discarded, dg::file::error::is_throw);
    }
    else if( type != "coords")
    {
        throw std::runtime_error( "Error: Unknown coordinates input '"
               + type + "'.");
    }

    auto coords = jsprobes["coords"];

    // read in parameters

    unsigned ndim = coords["coords-names"].size();

    std::string first = coords["coords-names"][0].asString();
    out.coords_names.resize(ndim);
    out.coords.resize(ndim);
    for( unsigned i=0; i<ndim; i++)
    {
        out.coords_names[i] = coords["coords-names"][i].asString();
    }
    unsigned num_pins = coords[out.coords_names[0]].size();
    out.probes = (num_pins > 0);

    if( rank == 0)
    {
    // only master thread reads probes
    for( unsigned i=0; i<ndim; i++)
    {
        unsigned num_pins = coords[out.coords_names[i]].size();
        out.coords[i].resize(num_pins);
        double scale = 1.;
        if( type == "file")
            scale = jsprobes["scale"][i].asDouble();
        for( unsigned k=0; k<num_pins; k++)
            out.coords[i][k] = coords.asJson()[out.coords_names[i]][k]
#ifdef DG_USE_JSONHPP
            .template get<double>()
#else
            .asDouble()
#endif
                *scale;
    }
    }
    // does not check that all probes have same size
    out.format = coords["format"].toStyledString();
    return out;
}



}//namespace file
} //namespace dg
