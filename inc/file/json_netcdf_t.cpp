#include <iostream>
#include <ctime>
#include <iomanip>

#define _FILE_INCLUDED_BY_DG_
#include "json_netcdf.h"
#include "nc_utilities.h"


int main(int argc, char* argv[])
{
    dg::file::JsonType inputfile;
    inputfile["physical"]["value"] = 1.2;
    inputfile["physical"]["another"] = "something";
    inputfile["bla"] = "blub";
    dg::file::JsonType att;
    att["text"] = "Hello World!";
    att["number"] = 3e-4;
    att["int"] = -1;
    att["uint"] = 10;
    att["bool"] = true;
#ifdef DG_USE_JSONHPP
    att["realarray"] = {-1.1, 42.3};
    att["intarray"] = {-11, 423};
    att["uintarray"] = {11, 423};
    att["boolarray"] = {true, false};
#else
    att["realarray"].append(-1.1);
    att["realarray"].append(42.3);
    att["intarray"].append(11);
    att["intarray"].append(-423);
    att["uintarray"].append(11);
    att["uintarray"].append(423);
    att["boolarray"].append(true);
    att["boolarray"].append(false);
#endif

    att["title"] = "Output file of feltor/src/toefl/toefl.cpp";
    att["Conventions"] = "CF-1.8";
    ///Get local time and begin file history
    auto ttt = std::time(nullptr);

    std::ostringstream oss;
    ///time string  + program-name + args
    oss << std::put_time(std::localtime(&ttt), "%F %T %Z");
    for( int i=0; i<argc; i++) oss << " "<<argv[i];
    att["history"] = oss.str();
    att["source"] = "FELTOR";
    att["git-hash"] = GIT_HASH;
    // Here we put the inputfile as a string without comments so that it can be read later by another parser
    att["inputfile"] = dg::file::WrappedJsonValue(inputfile).toStyledString();
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "atts.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    std::cout << dg::file::WrappedJsonValue(att).toStyledString()<<"\n";
    dg::file::json2nc_attributes( att, ncid, NC_GLOBAL);
    // TEST EMPTY Value
    att = dg::file::JsonType();
    dg::file::json2nc_attributes( att, ncid, NC_GLOBAL);
    nc_close(ncid);
    std::cout << "\n\nTEST BY USING ncdump -h atts.nc\n\n";


    return 0;
}
