#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include "training/Trainer.hpp"

int main(int argc, char** argv)
{
	/*
	 * Argument parsing
	 */
	po::options_description desc("Available options");
	desc.add_options()
	    ("help", "Show this message")
	;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << "Usage: " << argv[0]
			<< " [options]" << std::endl
			<< desc;
		return 1;
	}

	/*
	 * Run trainer
	 */
	Trainer trainer;
	trainer.loadFeatures("Training.yml");
	trainer.train();
	trainer.save("Trained_model.yml");

	return 0;

}
