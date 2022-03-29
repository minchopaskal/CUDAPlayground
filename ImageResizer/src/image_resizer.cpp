#include <image_resizer.h>

#define STBI_NO_HDR // TODO
#define STB_IMAGE_IMPLEMENTATION
#include <third_party/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <third_party/stb_image_write.h>

#include <cuda_buffer.h>

ImageResizer::ImageResizer() {
	// Push a sentinel value since index 0 is reserved for InvalidImageHandle
	images.push_back(ImageData{});

	// Choose device for resizing. Choose the one with maximum total mem.
	CUDAManager &cudaman = getCUDAManager();
	const std::vector<CUDADevice>& devices = cudaman.getDevices();
	SizeType maxTotalMem = 0;
	for (int i = 0; i < devices.size(); ++i) {
		SizeType totalMem;
		CUDAError err = devices[i].getTotalMemory(totalMem);
		if (err.hasError()) {
			continue;
		}
		if (totalMem > maxTotalMem) {
			maxTotalMem = totalMem;
			device = &devices[i];
		}
	}

	// No device is suitable. Just fail
	// We could fall back to CPU resizing but there is no point since this whole program
	// is just a CUDA exercise.
	if (maxTotalMem == 0) {
		Logger::log(LogLevel::Error, "No CUDA device suitable for resizing the image. Exitting...");
		exit(1);
	}

	resizeKernel.initialize(device->getModule(), "resize");
}

ImageHandle ImageResizer::openImage(const char *filename) {
	ImageData inputImg;
	inputImg.data = stbi_load(filename, &inputImg.width, &inputImg.height, &inputImg.numComp, 0);
	if (inputImg.data == nullptr) {
		Logger::log(LogLevel::Warning, "Image %s not found!", filename);
		return InvalidImageHandle;
	}

	inputImg.stbi_loaded = true;

	return addImage(inputImg);
}

void ImageResizer::freeImage(ImageHandle imgHandle) {
	if (!checkImageHandle(imgHandle)) {
		return;
	}

	ImageData &img = images[imgHandle];
	if (img.stbi_loaded) {
		stbi_image_free(img.data);
	} else {
		free(img.data);
	}

	img.data = nullptr;

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

ImageHandle ImageResizer::resize(const char *filename, int outputWidth, int outputHeight, ResizeAlgorithm resizingAlgorithm, ImageHandle *inputImageHandle) {
	ImageHandle _inputImageHandle = openImage(filename);
	if (inputImageHandle) {
		*inputImageHandle = _inputImageHandle;
	}

	if (_inputImageHandle == InvalidImageHandle) {
		return InvalidImageHandle;
	}

	ImageHandle outputImageHandle = resize(_inputImageHandle, outputWidth, outputHeight, resizingAlgorithm);

	if (inputImageHandle == nullptr) {
		freeImage(_inputImageHandle);
	}

	return outputImageHandle;
}

ImageHandle ImageResizer::resize(ImageHandle handle, int outputWidth, int outputHeight, ResizeAlgorithm resizingAlgorithm) {
	if (!checkImageHandle(handle)) {
		return InvalidImageHandle;
	}

	ImageData inputImage = images[handle];

	CUDAManager &cudaman = getCUDAManager();
	CUDADefaultBuffer deviceInputImage;
	CUDADefaultBuffer deviceOutputImage;

	device->use();
	
	const SizeType inputImageSize = SizeType(inputImage.width) * inputImage.height * inputImage.numComp;
	CUDAError err = deviceInputImage.initialize(inputImageSize);
	// TODO: handle out of mem errors with breaking up the image in parts
	if (err.hasError()) {
		return InvalidImageHandle;
	}

	err = deviceInputImage.upload(inputImage.data);
	if (err.hasError()) {
		return InvalidImageHandle;
	}

	const SizeType outputImagePixels = SizeType(outputWidth) * outputHeight;
	const SizeType outputImageSize = outputImagePixels * inputImage.numComp;
	err = deviceOutputImage.initialize(outputImageSize);
	if (err.hasError()) {
		return InvalidImageHandle;
	}

	const int numResizeKernelParams = 7;
	if (resizeKernel.getNumParams() > 0) {
		massert(resizeKernel.getNumParams() == numResizeKernelParams);
		resizeKernel.clearParams();
	}

	err = resizeKernel.addParams(
		deviceInputImage.handle(),
		inputImage.width,
		inputImage.height,
		inputImage.numComp,
		outputWidth,
		outputHeight,
		static_cast<int>(resizingAlgorithm),
		deviceOutputImage.handle()
	);
	if (err.hasError()) {
		return InvalidImageHandle;
	}

	err = resizeKernel.launchSync(
		outputImagePixels,
		device->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)
	);
	if (err.hasError()) {
		return InvalidImageHandle;
	}

	ImageData outputImage = {
		nullptr,
		outputWidth,
		outputHeight,
		inputImage.numComp,
		false
	};
	outputImage.data = (unsigned char*)malloc(outputImageSize);
	
	err = deviceOutputImage.download(outputImage.data);
	if (err.hasError()) {
		return InvalidImageHandle;
	}

	return addImage(outputImage);
}

bool ImageResizer::writeOutput(ImageHandle handle, ImageFormat format, const char *outputPath) const {
	if (!checkImageHandle(handle)) {
		return false;
	}

	Logger::log(LogLevel::Info, "Writing output to: %s", outputPath);

	ImageData img = images[handle];

	switch (format) {
	case ImageFormat::PNG:
		return stbi_write_png(outputPath, img.width, img.height, img.numComp, img.data, 0);
		break;
	case ImageFormat::BMP:
		return stbi_write_bmp(outputPath, img.width, img.height, img.numComp, img.data);
		break;
	case ImageFormat::TGA:
		return stbi_write_tga(outputPath, img.width, img.height, img.numComp, img.data);
		break;
	case ImageFormat::JPG:
		return stbi_write_jpg(outputPath, img.width, img.height, img.numComp, img.data, 100);
		break;
	case ImageFormat::HDR:
		// TODO
		return false;
	default:
		return false;
	}
	return false;
}

bool ImageResizer::checkImageHandle(ImageHandle handle) const {
	return !(handle == InvalidImageHandle || handle >= images.size() || images[handle].data == nullptr);
}