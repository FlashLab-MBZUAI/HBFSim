#pragma once
#include "host/hbf_controller.hpp"
#include <memory>

namespace hbfsim::verification {
// Component-test fixture with an explicit HBM backing device. This is not
// the zero-HBM system baseline: production compositions attach their existing
// HBM device and subtract its controller reservation from application capacity.
class HbfWithHbm : public host::HbfController {
public:
    explicit HbfWithHbm(host::HbfConfig config,
        physical::AddressHeatmap* heatmap = nullptr,
        physical::hbm::HbmConfig hbm_config = {})
        : host::HbfController(std::move(config), heatmap),
          hbm_(std::make_unique<physical::hbm::HbmDevice>(hbm_config)) {
        attach_hbm_buffer(*hbm_);
    }
    HbfWithHbm(HbfWithHbm&&) noexcept = default;
    HbfWithHbm& operator=(HbfWithHbm&&) noexcept = default;
    physical::hbm::HbmDevice& buffer_memory() { return *hbm_; }
private:
    std::unique_ptr<physical::hbm::HbmDevice> hbm_;
};
}
