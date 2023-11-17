#pragma once

#include "json_utilities.h"
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
 * The format value is a user-defined json value that is ignored by feltor and
 * copied "as-is" as a string attribute to the probes group in the output file.
 * Its purpose is to hold parsing information for the (flat) $R$, $Z$, $P$ arrays
 * for post-processing. For example
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
 * @attention In MPI only the master thread will read in the probes the others
 * return empty vectors
 * @ingroup json
*/
ProbesParams parse_probes( const dg::file::WrappedJsonValue& js, enum error
    probes_err = file::error::is_silent)
{
    ProbesParams out;
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif // MPI_VERSION
    if( (probes_err == file::error::is_silent) && !js.isMember( "probes"))
        return out;
    else if( (probes_err == file::error::is_warning) && !js.isMember( "probes"))
    {
        DG_RANK0 std::cerr << "WARNING: probes field not found.  No probes written to file!\n";
        return out;
    }
    else if ( !js.isMember("probes"))
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
    out.coords_names.resize(ndim);
    out.coords.resize(ndim);
    for( unsigned i=0; i<ndim; i++)
    {
        out.coords_names[i] = js_probes["coords-names"][i].asString();
        out.coords[i] = dg::HVec();
    }
    unsigned num_pins = js_probes[out.coords_names[0]].size();
    out.probes = (num_pins > 0);

#ifdef MPI_VERSION
    if( rank == 0)
    {
    // only master thread reads probes
#endif  //MPI_VERSION
    for( unsigned i=0; i<ndim; i++)
    {
        unsigned num_pins = js_probes[out.coords_names[i]].size();
        out.coords[i].resize(num_pins);
        double scale = 1.;
        if( type == "file")
            scale = probes_params["scale"][i].asDouble();
        for( unsigned k=0; k<num_pins; k++)
            out.coords[i][k] = js_probes.asJson()[out.coords_names[i]][k].asDouble()
                *scale;
    }
#ifdef MPI_VERSION
    }
#endif //MPI_VERSION
    // does not check that all probes have same size
    out.format = js_probes["format"].toStyledString();
    return out;
}



}//namespace file
} //namespace dg
