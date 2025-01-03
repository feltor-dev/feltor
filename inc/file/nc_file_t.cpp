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
	file.set_att(".", {"title", "Hello"} );
	file.set_att(".", {"same", "thing"} );
	file.set_att(".", {"title", "Hello world"});
    file.set_att(".", {"ttt", 42} );
	std::map<std::string, dg::file::nc_att_t> map_atts = {
		{"anumber", 42},
        {"axis", "X"},
		{"real_vector", std::vector{1.0,2.0,3.0}},
		{"int_vector", std::vector{1,2,3}}
	};
	file.set_atts(".", map_atts);
    file.set_atts(".", {{ "this", "is"}, {"a", 42}, {"map", std::vector{1,2,3}}});
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
    std::cout << "PASSED\n";

    file.open( "test.nc", dg::file::nc_write);
    file.rm_att( ".", "same");
    file.rename_att( ".", "ttt", "truth");
    int truth = file.get_att<int>(".", "truth");
    assert( truth == fortytwo);
    std::cout << "PASSED\n";

	////////////////// Dimensions
	file.def_dim("x", 42);
    std::vector<double> abscissas{ 1.,2.,3.,4.,5.};
	file.defput_dim("y", {{"axis", "Y"}, {"long_name", "y-coordinate"}},
            abscissas); // Create dim and dimension variable
	file.def_dim("time");

	size_t xsize = file.dim_size("x");
    assert( xsize == 42);

	size_t ysize = file.dim_size("y");
    assert( ysize == 5);
    std::cout << "PASSED Dimensions\n";
	/////////////////// Variables put

	file.def_var<double>("variable", {"time", "y","x"});
    std::vector<double> data( xsize*ysize, 7);
	file.put_var("variable", std::vector<size_t>{0, ysize, xsize}, data);

	file.set_att("variable", {"long_name", "blabla"});
    std::vector<int> data2(ysize, 42);
	file.defput_var("name", { "y" }, {{"long_name", "blub"}}, data2);
	//file.put("name", data);
	//file.put_var("time", {52}, 10);
	//file.stack("time", 52);

	/////////////////// Variables get
    file.get_var( "variable", std::vector<size_t>{0, ysize, xsize}, data);
    unsigned size = xsize*ysize;
    assert( data.size() == size);
    assert( data[0] == 7);
    std::cout << "PASSED Getters\n";

	file.close();
	return 0;
}
