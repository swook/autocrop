#include <regex>

#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

#include "file.hpp"

auto fname_regex = std::regex("^(.*[^\\/]+)(_([a-z]+))\\.([a-z]+)");

bool isUnprocessedImage(path file)
{
	std::smatch match;
	std::regex_match(file.string(), match, fname_regex);

	return match.size() == 0; // If no matches, not preprocessed
}

bool isSaliencyImage(path file)
{
	std::smatch match;
	std::regex_match(file.string(), match, fname_regex);

	if (match.size() < 3) return false;

	return match[2] == "saliency";
}

bool isGradImage(path file)
{
	std::smatch match;
	std::regex_match(file.string(), match, fname_regex);

	if (match.size() < 3) return false;

	return match[2] == "grad";
}

path setSuffix(path file, std::string suffix)
{
	if (!fs::is_regular_file(file))
	{
		throw std::runtime_error("Cannot set suffix of directory " + file.string());
	}

	std::smatch match;
	std::regex_match(file.string(), match, fname_regex);
	if (match.size())
		return path(std::regex_replace(file.string(), fname_regex,
		            "$2_" + suffix + ".$5"));
	else
	{
		return path(file.parent_path().string() + "/" +
		            file.stem().string() + "_" + suffix +
			    file.extension().string());
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



