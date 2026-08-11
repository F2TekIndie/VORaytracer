#include "Core/Log.h"

#include <chrono>
#include <format>
#include <iostream>
#include <mutex>

namespace vor
{
void log(LogLevel level, std::string_view message)
{
    static std::mutex mutex;
    const char* label = "INFO";
    if (level == LogLevel::Warning)
        label = "WARN";
    else if (level == LogLevel::Error)
        label = "ERROR";

    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::scoped_lock lock(mutex);
    std::clog << std::format("[{:%H:%M:%S}] {:5} {}\n", now, label, message);
}
} // namespace vor

