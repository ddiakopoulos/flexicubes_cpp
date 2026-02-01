#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <common/perf_timer.hpp>

#include <cstring>
#include <iostream>
#include <memory>

bool g_run_slow_tests = false;

struct TimingListener : doctest::IReporter
{
    const doctest::ContextOptions & opts;
    const doctest::TestCaseData * current = nullptr;
    std::unique_ptr<flexi::scoped_timer> current_timer;

    TimingListener(const doctest::ContextOptions & in)
        : opts(in) {}

    void report_query(const doctest::QueryData &) override {}
    void test_run_start() override {}
    void test_run_end(const doctest::TestRunStats &) override {}

    void test_case_start(const doctest::TestCaseData & in) override
    {
        current            = &in;
        std::ostream & out = opts.cout ? *opts.cout : std::cout;
        const char * name  = current && current->m_name ? current->m_name : "<unknown>";
        current_timer      = std::make_unique<flexi::scoped_timer>(name, out, true, "[doctest][timing] ");
    }

    void test_case_reenter(const doctest::TestCaseData & in) override
    {
        current            = &in;
        std::ostream & out = opts.cout ? *opts.cout : std::cout;
        const char * name  = current && current->m_name ? current->m_name : "<unknown>";
        current_timer      = std::make_unique<flexi::scoped_timer>(name, out, true, "[doctest][timing] ");
    }

    void test_case_end(const doctest::CurrentTestCaseStats &) override
    {
        current_timer.reset();
    }

    void test_case_exception(const doctest::TestCaseException &) override {}
    void subcase_start(const doctest::SubcaseSignature &) override {}
    void subcase_end() override {}
    void log_assert(const doctest::AssertData &) override {}
    void log_message(const doctest::MessageData &) override {}
    void test_case_skipped(const doctest::TestCaseData &) override {}
};

DOCTEST_REGISTER_LISTENER("timing", 0, TimingListener);

int main(int argc, char ** argv)
{
    int write = 1;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--run-slow-tests") == 0 || std::strcmp(argv[i], "--slow") == 0)
        {
            g_run_slow_tests = true;
            continue;
        }
        argv[write++] = argv[i];
    }
    argc = write;

    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int result = context.run();
    if (context.shouldExit())
    {
        return result;
    }
    return result;
}
