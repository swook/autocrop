#include <iostream>
#include <regex>

#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

#include "boost/regex.hpp"

#include "file.hpp"

boost::regex fname_regex = boost::regex("^(.*[^\\/]+)(_([a-z]+))\\.([a-z]+)",
                                        boost::regex::extended);

bool isUnprocessedImage(path file)
{
	boost::smatch match;
	const bool success = boost::regex_match(file.string(), match, fname_regex);

	return !success; // If no matches, not preprocessed
}

bool isSaliencyImage(path file)
{
	boost::smatch match;
	const bool success = boost::regex_match(file.string(), match, fname_regex);

	if (success && match.size() < 3) return false;

	return match[2] == "saliency";
}

bool isGradImage(path file)
{
	boost::smatch match;
	const bool success = boost::regex_match(file.string(), match, fname_regex);

	if (success && match.size() < 3) return false;

	return match[2] == "gradient";
}

path setSuffix(path file, std::string suffix)
{
	if (!file.has_filename())
	{
		throw std::runtime_error("Cannot set suffix of directory " + file.string());
	}
	boost::smatch match;
	const bool success = boost::regex_match(file.string(), match, fname_regex);
	if (success)
	{
		std::cout << match << std::endl;
		return path(boost::regex_replace(file.string(), fname_regex,
		            "$2_" + suffix + ".exr"));
	}
	else
	{
		return path(file.parent_path().string() + "/" +
		            file.stem().string() + "_" + suffix + ".exr");
	}
}

paths getUnprocessedImagePaths(path folder)
{
	paths out;
	fs::directory_iterator end;
	for (fs::directory_iterator it(folder); it != end; it++)
	{
		if (fs::is_regular_file(it->status()) &&
		    isUnprocessedImage(*it))
		{
			out.push_back(*it);
		}
	}
	return out;
}

paths getProcessedImagePaths(path folder)
{
	paths out;
	fs::directory_iterator end;
	for (fs::directory_iterator it(folder); it != end; it++)
	{
		if (fs::is_regular_file(it->status()) &&
		    !isUnprocessedImage(*it))
		{
			out.push_back(*it);
		}
	}
	return out;
}



