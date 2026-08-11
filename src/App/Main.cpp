#include "App/Application.h"

#include "Core/Log.h"

#include <exception>

int main()
{
    try
    {
        vor::Application application;
        return application.run();
    }
    catch (const std::exception& error)
    {
        vor::log(vor::LogLevel::Error, error.what());
        return 1;
    }
}

