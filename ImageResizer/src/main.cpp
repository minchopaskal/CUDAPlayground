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
		"\t%s\n\t-i|-input input_img_file_path MANDATORY\n"
		"\t-o|-output output_img_file_path OPTIONAL\n"
		"\t-ow output_width (0-inf] MANDATORY\n"
		"\t-oh output_height (0-inf] MANDATORY\n"
		"\t-a|-algorithm which algorithm to use for resizing [0-1] OPTIONAL DEFAULT: Lancsoz\n"
		"\t-h prints this usage message and exits OPTIONAL\n"
		"\n"
		"\tSupported resizing algorithms: 0(Nearest neighbour); 1(Lancsoz).\n",
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
	int resizingAlgorithm = 1;

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

		if (strncmp(argv[i], "-output", 7) == 0 || strncmp(argv[i], "-o", 2) == 0) {
			imgOutputPath = argv[i+1];
			i += 2;
			continue;
		}

		if (strncmp(argv[i], "-algorithm", 10) == 0 || strncmp(argv[i], "-a", 2) == 0) {
			resizingAlgorithm = atoi(argv[i + 1]);
			if (resizingAlgorithm < 0 || resizingAlgorithm > static_cast<int>(ResizeAlgorithm::Count)) {
				printUsage(argv[0]);
				return 0;
			}
			i += 2;
			continue;
		}

		++i;
	}

	if (imgFilePath == nullptr || outputWidth <= 0 || outputHeight <= 0) {
		Logger::log(LogLevel::Error, "Invalid arguments! Please refer to help:");
		printUsage(argv[0]);
		return 1;
	}
	
	ResizeAlgorithm algo;
	switch (resizingAlgorithm) {
	case 0:
		algo = ResizeAlgorithm::Nearest;
		break;
	case 1:
		algo = ResizeAlgorithm::Lancsoz;
		break;
	default:
		algo = ResizeAlgorithm::Lancsoz;
		break;
	}

	ImageResizer imgResizer;
	ImageHandle outImgHandle = imgResizer.resize(imgFilePath, outputWidth, outputHeight, algo, nullptr);

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

	std::string ext;
	SizeType lastDotIdx = outName.find_last_of('.');
	if (lastDotIdx != std::string::npos) {
		ext = outName.substr(lastDotIdx + 1);
	}

	ImageFormat outputFormat;
	if (ext == "jpg" || ext == "jpeg") {
		outputFormat = ImageFormat::JPG;
	}
	if (ext == "tga") {
		outputFormat = ImageFormat::TGA;
	}
	if (ext == "png") {
		outputFormat = ImageFormat::PNG;
	}
	if (ext == "bmp") {
		outputFormat = ImageFormat::BMP;
	}

	if (!imgResizer.writeOutput(outImgHandle, outputFormat, imgOutputPath)) {
		Logger::log(LogLevel::Debug, "Writing output image failed!");
	}

	deinitializeCUDAManager();

	return 0;
}