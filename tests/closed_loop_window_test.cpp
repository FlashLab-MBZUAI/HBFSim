#include "physical/hybrid/composition_common.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using hbfsim::physical::hybrid::ClosedLoopWindow;

void expect_time(double actual, double expected, const std::string& context) {
    if (std::abs(actual - expected) > 1e-12) {
        throw std::runtime_error(
            context + ": expected " + std::to_string(expected) +
            " ns, got " + std::to_string(actual) + " ns");
    }
}

void test_out_of_order_completion_returns_credit() {
    ClosedLoopWindow window{2};
    expect_time(window.admit(0.0), 0.0, "first admission");
    window.complete(100.0);
    expect_time(window.admit(0.0), 0.0, "second admission");
    window.complete(10.0);

    // This is the essential regression: request 1 completes before request 0,
    // so generic max-outstanding semantics must admit at 10 ns, not 100 ns.
    expect_time(window.admit(0.0), 10.0, "out-of-order credit return");
    window.complete(20.0);
    expect_time(window.admit(0.0), 20.0, "next earliest completion");
}

void test_completion_ties_return_every_credit() {
    ClosedLoopWindow window{3};
    expect_time(window.admit(0.0), 0.0, "tie admission 0");
    window.complete(5.0);
    expect_time(window.admit(0.0), 0.0, "tie admission 1");
    window.complete(100.0);
    expect_time(window.admit(0.0), 0.0, "tie admission 2");
    window.complete(5.0);

    expect_time(window.admit(0.0), 5.0, "first tied credit");
    window.complete(200.0);
    // Both 5 ns completions were retired at the same causal frontier, leaving
    // another credit available without waiting for 100 ns.
    expect_time(window.admit(5.0), 5.0, "second tied credit");
}

void test_nominal_time_retires_completed_requests() {
    ClosedLoopWindow window{2};
    expect_time(window.admit(0.0), 0.0, "nominal admission 0");
    window.complete(5.0);
    expect_time(window.admit(0.0), 0.0, "nominal admission 1");
    window.complete(10.0);

    expect_time(window.admit(7.0), 7.0, "nominal frontier retires old completion");
    window.complete(9.0);
    expect_time(window.admit(8.0), 9.0, "full window waits for earliest completion");
}

void test_window_edge_modes() {
    ClosedLoopWindow open_loop{0};
    expect_time(open_loop.admit(3.0), 3.0, "open-loop admission");
    open_loop.complete(1000.0);
    expect_time(open_loop.admit(4.0), 4.0, "open-loop ignores completion");

    ClosedLoopWindow serial{1};
    expect_time(serial.admit(0.0), 0.0, "serial admission 0");
    serial.complete(11.0);
    expect_time(serial.admit(0.0), 11.0, "serial completion barrier");
}

void test_independent_tier_windows_do_not_cross_block() {
    ClosedLoopWindow hbm{1};
    ClosedLoopWindow hbf{1};
    expect_time(hbm.admit(0.0), 0.0, "HBM first admission");
    hbm.complete(100.0);
    expect_time(hbf.admit(0.0), 0.0, "HBF independent admission");
    hbf.complete(10.0);
    expect_time(hbf.admit(0.0), 10.0, "HBF returns its own credit");
    hbf.complete(20.0);
    expect_time(hbm.admit(0.0), 100.0, "HBM retains its own long credit");
}

}  // namespace

int main() {
    try {
        test_out_of_order_completion_returns_credit();
        test_completion_ties_return_every_credit();
        test_nominal_time_retires_completed_requests();
        test_window_edge_modes();
        test_independent_tier_windows_do_not_cross_block();
        std::cout << "closed-loop completion-order credit regression passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "closed-loop window regression failed: " << error.what() << '\n';
        return 1;
    }
}
