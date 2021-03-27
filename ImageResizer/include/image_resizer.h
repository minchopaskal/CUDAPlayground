#pragma once

#include <cstdlib>
#include <vector>
#include <stack>

using ImageHandle = size_t;
#define InvalidImageHandle ImageHandle(0)

struct ImageResizer {
	
	ImageResizer();
	~ImageResizer();

	/// Resize an image given its path and desired output dimensions.
	/// @param filename Input image file path
	/// @param outputWidth Desired output width
	/// @param outputHeight Desired output height
	/// @param inputImageHandle If not null, returns a handle to the input image for further processing if wished.
	/// @return Handle to the resized image.
	ImageHandle resize(const char *filename, int outputWidth, int outputHeight, ImageHandle *inputImageHandle);

	/// Resize an image given its handle and desired output dimensions.
	/// @param handle Handle of the image we want to resize
	/// @param outputWidth Desired output width
	/// @param outputHeight Desired output height
	/// @return Handle to the resized image.
	ImageHandle resize(ImageHandle handle, int outputWidth, int outputHeight);
	
	/// Given an image handle writes its data to the given outputPath.
	/// @param img Handle to the image we want to output
	/// @param outputPath Output file path
	/// @return false if the handle or the image data is invalid.
	bool writeOutput(ImageHandle img, const char *outputPath) const;

	/// Unloads a saved image given its handle.
	void freeImage(ImageHandle img);

private:
	struct ImageData {
		unsigned char *data; ///< Image data
		int width; ///< Width of the image in pixels
		int height; ///< Height of the image in pixels
		int numComp; ///< Number of 8-bit components per pixel
		bool stbi_loaded; ///< Indicates wheather the image was loaded from file or was a result of some processing
	};

	ImageHandle addImage(ImageData img);
	bool checkImageHandle(ImageHandle handle) const;

private:
	std::vector<ImageData> images;
	std::stack<size_t> freeSlots;
};