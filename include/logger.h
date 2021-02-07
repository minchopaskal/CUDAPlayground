#pragma once

#include <cstdio>

#define OUT_STREAM stdout
#define IN_STREAM stdin
#define ERR_STREAM stderr

enum class LogLevel {
	Info,
	Warning,
	Debug,
	Error
};

struct Logger {
	Logger() = delete;

	static void log(LogLevel lvl, const char *fmt, ...) {
#ifndef CUDA_DEBUG
		if (lvl == LogLevel::Debug) {
			return;
		}
#endif

		// TODO: take into consideration the log level
		va_list args;
		__va_start(&args, fmt);
		vfprintf(OUT_STREAM, fmt, args);
		__crt_va_end(args);
		fprintf(OUT_STREAM, "\n");
	}
};