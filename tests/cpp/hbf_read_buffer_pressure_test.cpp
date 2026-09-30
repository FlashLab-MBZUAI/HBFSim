#include "physical/hbf/hbf_device.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// Independent serial LRU oracle: each bank has two decoded pages. Exercise
// thousands of fills, hits, invalidations, and causal-frontier retirements.
int main() {
    using namespace hbfsim::physical::hbf;
    HbfDeviceConfig c; c.channels_per_stack=1; c.planes_per_die=4;
    c.blocks_per_plane=16; c.pages_per_block=16;
    HbfDevice d(c);
    std::array<std::vector<std::uint64_t>,4> oracle;
    std::mt19937_64 rng(20260911);
    const auto stride = c.blocks_per_plane*c.pages_per_block;
    for (unsigned n=1;n<=20000;++n) {
        const auto bank=rng()%4, page=bank*stride+rng()%8;
        auto& resident=oracle[bank];
        const auto found=std::find(resident.begin(),resident.end(),page);
        const bool expected=found!=resident.end();
        const auto action=rng()%5;
        if (action==0) {
            d.cache_purge_page(page); if(expected) resident.erase(found);
        } else {
            if (d.cache_hit(page,n)!=expected) throw std::runtime_error("bank-local LRU differs from oracle");
            if(expected) resident.erase(found); else d.cache_fill(page,n);
            resident.push_back(page);
            if(resident.size()>2) resident.erase(resident.begin());
        }
        d.advance_cache(n);
    }
    std::cout << "20,000 bank-local cache operations match the independent LRU oracle\n";
}
