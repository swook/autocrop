#pragma once

#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

typedef fs::path              path;
typedef std::vector<fs::path> paths;

/**
 * isUnprocessedImage checks if a given file path points to a filename without a
 * suffix. A suffix is a lower-case string prefixed with _.
 *
 * For example, path/filename.jpg is unprocessed, while
 * path/filename_saliency.jpg is a saliency map output.
 */
bool isUnprocessedImage(path file);

/**
 * setSuffix replaces an existing suffix or adds a suffix to a file path
 *
 * Current suffices include:
 * - _saliency: Saliency Map
 * - _grad:     Gradient Map
 */
path setSuffix(path file, std::string suffix);

/**
 * getUnprocessedImagePaths returns a list of file paths in a given folder which
 * point to unprocessed images. See isUnprocessedImage.
 */
paths getUnprocessedImagePaths(path folder);


