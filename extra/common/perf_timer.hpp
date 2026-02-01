#pragma once

#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace flexi
{

    struct perf_timer
    {
        using clock = std::chrono::high_resolution_clock;
        clock::time_point start_;
        perf_timer() : start_(clock::now()) {}
        void reset() { start_ = clock::now(); }
        double elapsed_ms() const {  return std::chrono::duration<double, std::milli>(clock::now() - start_).count(); }
    };

    class scoped_timer
    {
    public:

        scoped_timer(std::string label, std::ostream & os, bool enabled = true, std::string prefix = "")
            : label_(std::move(label)),
              prefix_(std::move(prefix)),
              os_(&os),
              enabled_(enabled),
              stopped_(false),
              elapsed_ms_(0.0) {}

        ~scoped_timer() { stop(); }

        double stop()
        {
            if (!enabled_ || stopped_) return elapsed_ms_;
            elapsed_ms_ = timer_.elapsed_ms();
            stopped_    = true;
            if (os_)
            {
                std::ostringstream line;
                line << prefix_ << label_ << ": " << std::fixed << std::setprecision(3)
                     << elapsed_ms_ << " ms\n";
                (*os_) << line.str();
            }
            return elapsed_ms_;
        }

        double elapsed_ms() const { return stopped_ ? elapsed_ms_ : timer_.elapsed_ms(); }

    private:

        perf_timer timer_;
        std::string label_;
        std::string prefix_;
        std::ostream * os_;
        bool enabled_;
        bool stopped_;
        double elapsed_ms_;
    };

}  // namespace flexi
