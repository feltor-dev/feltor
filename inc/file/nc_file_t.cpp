#include <iostream>

#include "nc_file.h"

int main()
{
	dg::file::SerialNcFile file;
	try {
		file.open("test.nc", dg::file::nc_clobber);
	}
	catch(dg::file::NC_Error & e)	{
		std::cerr << e.what();
	}
	file.def_grp("subgroup");
	file.set_grp("subgroup");
	file.set_grp("..");
	///////////////////////    Groups
	file.def_grp("test");
	file.set_grp("test");
	file.set_grp(".."); // go to parent group
	file.set_grp("."); // do nothing "." is current group
	file.set_grp(); // go back to root group
	std::vector<std::string> grps = file.get_grps();
	// Traverse all groups
	// probably recursive
	///////////////////////   Attributes
	// set attributes
	file.set_att(".", std::pair{"title", "Hello"} );
	file.set_att(".", std::pair{"same", "thing"} );
	file.set_att(".", std::pair{"title", "Hello world"});
    file.set_att(".", std::make_pair("ttt", 42) );
	std::map<std::string, dg::file::nc_att_t> map_atts = {
		{"anumber", 42},
        {"axis", "X"},
		{"real_vector", std::vector{1.0,2.0,3.0}},
		{"int_vector", std::vector{1,2,3}}
	};
	file.set_atts(".", map_atts);
	file.close();
    file.open( "test.nc", dg::file::nc_nowrite);
	// get attributes
	auto title = file.get_att<std::string>(".", "title"); // get index 0 automatically
    assert( title == "Hello world");
	double t = file.get_att_i<double>(".", "ttt"); // get index 1
    assert( t == 42);
	auto ts = file.get_att_v<double>(".", "real_vector"); // get all indices as vector

    std::vector<double> result = {1.0,2.0,3.0};
    assert( ts == result );
    std::map<std::string, dg::file::nc_att_t> atts =
        file.get_atts<dg::file::nc_att_t>(); // get the map < string, variant>
    std::string hello = "Hello world";
    assert( std::get<std::string>(atts.at("title")) == hello);
    int fortytwo = 42;
    assert( std::get<int>(atts.at("ttt")) == fortytwo);
	file.close();
    file.open( "test.nc", dg::file::nc_write);
    file.rm_att( ".", "same");
    file.rename_att( ".", "ttt", "truth");
    int truth = file.get_att<int>(".", "truth");
    assert( truth == fortytwo);
    std::cout << "PASSED\n";

	//struct NcVariable
	//{
	//	std::string name;
	//	std::map<std::string, dg::file::att_t> atts;
	//	lambda creator(...);
	//};
	//////////////////// Dimensions
	//file.def_dim("x", 42);
	//file.defput_dim("x", atts, abscissas); // Create dim and dimension variable
	//file.def_dim("time");

	//file.dim_size("x");
	//file.get_size("x");
	///////////////////// Variables

	//file.get("variable", data);
	//file.get("name", data2);

	//file.def("name", atts, {"time", "y", "x"});
	//file.def_and_put("name", { "y" }, atts, data2);
	//file.put("name", data);
	//file.put("time", 52, 10);
	//file.stack("time", 52);


	file.close();
	return 0;
}
