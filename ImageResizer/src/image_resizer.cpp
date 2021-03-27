#include <image_resizer.h>

#define STB_IMAGE_IMPLEMENTATION
#include <third_party\stb_image.h>

ImageResizer::ImageResizer() {
	// Push a sentinel value since 0 is reserved for InvalidImageHandle
	images.push_back(ImageData{});
}

void ImageResizer::freeImage(ImageHandle imgHandle) {
	if (!checkImageHandle(imgHandle)) {
		return;
	}

	ImageData img = images[imgHandle];
	if (!img.stbi_loaded) {
		return;
	}

	stbi_image_free(img.data);
	freeSlots.push(imgHandle);
}

ImageHandle ImageResizer::addImage(ImageData img) {
	ImageHandle result;

	if (freeSlots.empty()) {
		images.push_back(img);
		result = images.size() - 1;
	} else {
		result = freeSlots.top();
		freeSlots.pop();
	}
	
	return result;
}

ImageResizer::~ImageResizer() {
	for (int i = 0; i < images.size(); ++i) {
		freeImage(i);
	}
}

ImageHandle ImageResizer::resize(const char *filename, int outputWidth, int outputHeight, ImageHandle *inputImageHandle) {
	ImageData inputImg;
	inputImg.data = stbi_load(filename, &inputImg.width, &inputImg.height, &inputImg.numComp, 0);
	if (inputImg.data == nullptr) {
		if (inputImageHandle) {
			*inputImageHandle = InvalidImageHandle;
		}
		return InvalidImageHandle;
	}

	inputImg.stbi_loaded = true;

	ImageHandle _inputImageHandle = addImage(inputImg);

	ImageHandle outputImageHandle = resize(_inputImageHandle, outputWidth, outputHeight);

	if (inputImageHandle) {
		*inputImageHandle = _inputImageHandle;
	}

	return outputImageHandle;
}

ImageHandle ImageResizer::resize(ImageHandle handle, int outputWidth, int outputHeight) {
	if (!checkImageHandle(handle)) {
		return InvalidImageHandle;
	}
	// TODO: resize the image
	return InvalidImageHandle;
}

bool ImageResizer::writeOutput(ImageHandle handle, const char * outputPath) const {
	if (!checkImageHandle(handle)) {
		return false;
	}

	// TODO: writing to output
	return false;
}

bool ImageResizer::checkImageHandle(ImageHandle handle) const {
	return !(handle == InvalidImageHandle || handle >= images.size() || images[handle].data == nullptr);
}