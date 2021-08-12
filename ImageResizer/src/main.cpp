#include <cuda_manager.h>
#include <image_resizer.h>

void testSystem() {
	CUDAManager &cudaman = getCUDAManager();
	CUDAError err = cudaman.testSystem();
	if (err.hasError()) {
		LOG_CUDA_ERROR(err, LogLevel::Error);
	}
}

void printUsage(const char *appName) {
	Logger::log(
		LogLevel::InfoFancy, 
		"Usage:\n"
		"\t%s -i|-input input_img_file_path MANDATORY\n"
		"\t-o|-output output_img_file_path OPTIONAL\n"
		"\t-ow output_width (0-inf] MANDATORY\n"
		"\t-oh output_height (0-inf] MANDATORY\n"
		"\t-h prints this usage message and exits OPTIONAL",
		appName
	);
}

int main(int argc, char **argv) {
	if (argc < 7) {
		printUsage(argv[0]);
		return 1;
	}

	initializeCUDAManager(std::vector<std::string>{"data\\resize_kernel.ptx"}, false);

	//testSystem();

	const char *imgFilePath = nullptr;
	const char *imgOutputPath = nullptr;
	int outputWidth = -1;
	int outputHeight = -1;
	// TODO: also add which resizing algorithm to use

	for (int i = 1; i < argc; ) {
		if (strncmp(argv[i], "-h", 2) == 0) {
			printUsage(argv[0]);
			return 0;
		}

		if (strncmp(argv[i], "-input", 6) == 0 || strncmp(argv[i], "-i", 2) == 0) {
			imgFilePath = argv[i+1];
			i+=2;
			continue;
		}

		if (strncmp(argv[i], "-ow", 3) == 0) {
			outputWidth = atoi(argv[i + 1]);
			i += 2;
			continue;
		}

		if (strncmp(argv[i], "-oh", 3) == 0) {
			outputHeight = atoi(argv[i + 1]);
			i += 2;
			continue;
		}

		if (strncmp(argv[i], "-output", 6) == 0 || strncmp(argv[i], "-o", 2) == 0) {
			imgOutputPath = argv[i+1];
			i+=2;
			continue;
		}

		++i;
	}

	if (imgFilePath == nullptr || outputWidth <= 0 || outputHeight <= 0) {
		Logger::log(LogLevel::Error, "Invalid arguments! Please refer to help:");
		printUsage(argv[0]);
		return 1;
	}
	
	ImageResizer imgResizer;
	ImageHandle outImgHandle = imgResizer.resize(imgFilePath, outputWidth, outputHeight, nullptr);

	std::string outName = imgFilePath;
	if (imgOutputPath == nullptr) {
		const char *outExt = "_OUT.jpg";
		SizeType lastDotIdx = outName.find_last_of('.');
		if (lastDotIdx != std::string::npos) {
			outName.erase(lastDotIdx);
		}

		outName += outExt;
		imgOutputPath = outName.c_str();
	}

	if (!imgResizer.writeOutput(outImgHandle, ImageFormat::JPG, imgOutputPath)) {
		Logger::log(LogLevel::Debug, "Writing output image failed!");
	}

	deinitializeCUDAManager();

	return 0;
}