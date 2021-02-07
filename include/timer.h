#pragma once

#include <Windows.h>

struct Timer {
	Timer() {
		LARGE_INTEGER temp;
		QueryPerformanceFrequency(&temp);
		frequency = static_cast<double>(temp.QuadPart) / 1000.0;
	}

	void start() {
		QueryPerformanceCounter(&startTime);
	}

	void restart() {
		QueryPerformanceCounter(&startTime);
	}
	
	float time() {
		LARGE_INTEGER endTime;
		QueryPerformanceCounter(&endTime);
		double elapsedTime = static_cast<double>(endTime.QuadPart) - static_cast<double>(startTime.QuadPart);

		return static_cast<float>(elapsedTime / frequency);
	}

private:
	LARGE_INTEGER startTime;
	double frequency;
};