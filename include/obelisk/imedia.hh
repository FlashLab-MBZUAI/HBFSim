#pragma once

#include "obelisk/imodule.hh"

namespace obelisk {

class IMedia : public IModule {
public:
    virtual MediaType media_type() const = 0;
    virtual uint64_t capacity_bytes() const = 0;
    virtual uint32_t min_access_granularity() const = 0;
    virtual double peak_bandwidth_gbps() const = 0;
};

}  // namespace obelisk
