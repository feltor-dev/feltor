#include <iostream>

#include "nc_file.h"

int main()
{
	dg::file::NcFile file;
	try {
		file.open("test.nc", dg::file::nc_clobber);
	}
	cath(dg::file::NcError & e)	{
		std::cerr << e.what();
	}
	file.grp_create("subgroup");
	file.grp_set("subgroup");
	file.grp_set("..");
	///////////////////////    Groups
	file.def_grp("test");
	file.set_grp("test");
	file.set_grp(".."); // go to parent group
	file.set_grp("."); // do nothing "." is current group
	file.set_grp(); // go back to root group
	std::vector<std::string> = file.get_grps();
	// Traverse all groups
	// probably recursive
	///////////////////////   Attributes
	// get attributes
	auto title = file.get_att<std::string>(".", "title"); // get index 0 automatically
	double t = file.get_att<double>("variable", "att", 1); // get index 1
	auto ts = file.get_att_v<double>("variable", "att"); // get all indices as vector
	auto atts = file.get_atts(); // get the map < string, variant>
	auto json1 = file.get_atts_as_json("."); // maybe as default "dot is current group"
	auto json2 = file.get_atts_as_json("variable");

	// set attributes
	file.set_att(".", { "title", "Hello world" });
	file.set_att("variable", { "same", "thing" });
	file.set_atts(".", { "title", "Hello world" }, { "ttt", 42 });
	// std::any_cast always compiles, only throws at runtime ...
	// just allow bool, int, unsigned, float, double and std::vector of them
	// or any if you have to account for compound types?
	// somehow register a type? or read from file?
	// Maybe use std::variant<...> and typedef using dg::file::att_t = ...
	std::map<std::string, std::any> atts = {
		{"att0", 42}, {"axis", "X"},
		{"another", std::array{1,2,3}}, // probably disallow
		{"asdl", std::vector{1,2,3}}
	}
	file.set_atts("variable", map_atts);
	file.set_atts_from_json(".", json1);
	struct NcVariable
	{
		std::string name;
		std::map<std::string, dg::file::att_t> atts;
		lambda creator(...);
	};
	////////////////// Dimensions
	file.def_dim("x", 42);
	file.defput_dim("x", atts, abscissas); // Create dim and dimension variable
	file.def_dim("time");

	file.dim_size("x");
	file.get_size("x");
	/////////////////// Variables

	file.get("variable", data);
	file.get("name", data2);

	file.def("name", atts, {"time", "y", "x"});
	file.def_and_put("name", { "y" }, atts, data2);
	file.put("name", data);
	file.put("time", 52, 10);
	file.stack("time", 52);


	file.close();
	return 0;
}
