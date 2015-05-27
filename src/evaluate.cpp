#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "datasets/Chen.hpp"

int main(int argc, char** argv)
{

	/*
	 * Run Chen::quantEval
	 */
	ds::Chen chen;
	chen.quantEval();

	return 0;

}
