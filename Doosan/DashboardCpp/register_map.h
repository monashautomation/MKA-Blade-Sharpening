// Register map mirroring Doosan/Dashboard/server.py.
// Adding a new register: add a row here AND a row in the index.html render() call.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct RegMeta {
    int          addr;
    int          scale;     // stored = real * scale
    bool         is_signed; // true -> two's complement on 16 bits
    bool         editable;  // false -> dashboard renders read-only unless override
    std::string  group;     // "config" | "detection" | "toggle" | "derived" | "handshake"
};

inline const std::vector<std::pair<std::string, RegMeta>>& registers_in_order() {
    static const std::vector<std::pair<std::string, RegMeta>> R = {
        // config
        {"BAY_ID",         {128, 1,   false, true,  "config"}},
        {"GRINDER_ID",     {129, 1,   false, true,  "config"}},
        {"ANGLE",          {130, 10,  false, true,  "config"}},
        {"DEPTH",          {131, 100, false, true,  "config"}},
        {"LENGTH",         {132, 1,   false, true,  "config"}},
        {"CONFIG_VERSION", {133, 1,   false, true,  "config"}},
        {"BLADE_COUNT",    {144, 1,   false, true,  "config"}},

        // detection
        {"DETECTION_X",    {134, 100, true,  true,  "detection"}},
        {"DETECTION_Y",    {135, 100, true,  true,  "detection"}},

        // toggles
        {"FEAT_HOMING",    {145, 1, false, true, "toggle"}},
        {"FEAT_PICKUP",    {146, 1, false, true, "toggle"}},
        {"FEAT_LENGTH",    {147, 1, false, true, "toggle"}},
        {"FEAT_GRIND",     {148, 1, false, true, "toggle"}},
        {"FEAT_PUTDOWN",   {149, 1, false, true, "toggle"}},
        {"MODE",           {150, 1, false, true, "toggle"}},

        // derived
        {"BLADE_DEPTH",    {160, 100, true,  true, "derived"}},
        {"BLADE_RIGHT_Y",  {161, 10,  true,  true, "derived"}},
        {"BLADE_LENGTH",   {162, 10,  false, true, "derived"}},
        {"BLADE_LEFT_Y",   {163, 10,  true,  true, "derived"}},

        // handshake
        {"STATUS",             {136, 1,  false, false, "handshake"}},
        {"START",              {137, 1,  false, false, "handshake"}},
        {"GRIND_READY",        {138, 1,  false, false, "handshake"}},
        {"GRIND",              {139, 1,  false, false, "handshake"}},
        {"EMERGENCY",          {140, 1,  false, false, "handshake"}},
        {"CALIBRATION_READY",  {141, 1,  false, false, "handshake"}},
        {"CALIBRATION_ANGLE",  {142, 10, false, false, "handshake"}},
    };
    return R;
}

inline const std::unordered_map<std::string, RegMeta>& registers_by_name() {
    static std::unordered_map<std::string, RegMeta> M = [] {
        std::unordered_map<std::string, RegMeta> m;
        for (const auto& [k, v] : registers_in_order()) m.emplace(k, v);
        return m;
    }();
    return M;
}

inline double decode(uint16_t raw, const RegMeta& m) {
    int v = raw;
    if (m.is_signed && v >= 32768) v -= 65536;
    return m.scale != 1 ? static_cast<double>(v) / m.scale : static_cast<double>(v);
}

inline uint16_t encode(double real, const RegMeta& m) {
    long v = static_cast<long>(std::round(real * m.scale));
    if (v > 32767)  v = 32767;
    if (v < -32768) v = -32768;
    if (m.is_signed && v < 0) v += 65536;
    return static_cast<uint16_t>(v & 0xFFFF);
}
