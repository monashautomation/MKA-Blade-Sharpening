// DashboardCpp - C++ replacement for Doosan/Dashboard/server.py.
//
// Today: matches Flask version exactly. Browser polls /api/registers, this
// process talks Modbus TCP (port 502) to the robot's slave registers.
//
// Tomorrow (per-file stages, see notes in chat):
//   - link DRFL, hold a CDRFLEx connection alongside the libmodbus one
//   - add POST /api/run/<stage>  -> reads stages/<stage>.drl, prepends
//                                   _header.drl, calls robot.drl_start(...)
//   - add POST /api/drl/stop     -> robot.drl_stop()
//   - hook DRFL state callback so /api/registers can return current state
//     (STANDBY / MOVING / etc.) without polling
//
// Build (Linux):
//   sudo apt-get install libmodbus-dev libpoco-dev
//   cmake -B build && cmake --build build -j
//   ./build/dashboard_cpp                 # serves http://0.0.0.0:5050
//
// Env: ROBOT_IP (default 172.24.89.89), ROBOT_PORT (502), HTTP_PORT (5050).

#define CPPHTTPLIB_OPENSSL_SUPPORT 0

#include "register_map.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <modbus/modbus.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

using json = nlohmann::json;

// Single Modbus connection, serialized by mutex. libmodbus contexts are not
// thread-safe; cpp-httplib calls handlers from its thread pool.
struct ModbusCtx {
    modbus_t*  ctx = nullptr;
    std::mutex mu;
    bool       connected = false;
};

static std::string env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return v ? v : fallback;
}

static bool ensure_connected(ModbusCtx& mb, const std::string& ip, int port) {
    if (mb.connected) return true;
    if (!mb.ctx) {
        mb.ctx = modbus_new_tcp(ip.c_str(), port);
        modbus_set_response_timeout(mb.ctx, 2, 0);
    }
    if (modbus_connect(mb.ctx) == -1) {
        std::cerr << "modbus_connect failed: " << modbus_strerror(errno) << "\n";
        return false;
    }
    mb.connected = true;
    return true;
}

static void disconnect(ModbusCtx& mb) {
    if (mb.ctx && mb.connected) {
        modbus_close(mb.ctx);
        mb.connected = false;
    }
}

static json meta_to_json(const RegMeta& m) {
    return json{
        {"addr", m.addr}, {"scale", m.scale}, {"signed", m.is_signed},
        {"editable", m.editable}, {"group", m.group},
    };
}

int main() {
    const std::string ROBOT_IP   = env_or("ROBOT_IP",   "172.24.89.89");
    const int         ROBOT_PORT = std::stoi(env_or("ROBOT_PORT", "502"));
    const int         HTTP_PORT  = std::stoi(env_or("HTTP_PORT",  "5050"));

    ModbusCtx mb;

    httplib::Server srv;
    srv.set_mount_point("/", ".");  // serves index.html and any siblings

    srv.Get("/api/registers", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(mb.mu);
        if (!ensure_connected(mb, ROBOT_IP, ROBOT_PORT)) {
            res.status = 502;
            res.set_content(json{{"ok", false},
                {"error", "can't reach " + ROBOT_IP + ":" + std::to_string(ROBOT_PORT)}}.dump(),
                "application/json");
            return;
        }
        json out_regs = json::object();
        for (const auto& [name, meta] : registers_in_order()) {
            uint16_t raw = 0;
            int rc = modbus_read_registers(mb.ctx, meta.addr, 1, &raw);
            if (rc == -1) {
                disconnect(mb);
                out_regs[name] = {{"value", nullptr},
                                  {"error", modbus_strerror(errno)},
                                  {"meta", meta_to_json(meta)}};
            } else {
                out_regs[name] = {{"value", decode(raw, meta)},
                                  {"raw", raw},
                                  {"meta", meta_to_json(meta)}};
            }
        }
        res.set_content(json{{"ok", true},
            {"robot_ip", ROBOT_IP + ":" + std::to_string(ROBOT_PORT)},
            {"registers", out_regs}}.dump(),
            "application/json");
    });

    srv.Post("/api/registers", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(json{{"ok", false}, {"error", "invalid json"}}.dump(),
                            "application/json");
            return;
        }

        std::lock_guard<std::mutex> lk(mb.mu);
        if (!ensure_connected(mb, ROBOT_IP, ROBOT_PORT)) {
            res.status = 502;
            res.set_content(json{{"ok", false},
                {"error", "can't reach " + ROBOT_IP + ":" + std::to_string(ROBOT_PORT)}}.dump(),
                "application/json");
            return;
        }

        json results = json::object();
        const auto& by_name = registers_by_name();
        for (auto it = body.begin(); it != body.end(); ++it) {
            const auto& name = it.key();
            auto m_it = by_name.find(name);
            if (m_it == by_name.end()) {
                results[name] = {{"ok", false}, {"error", "unknown register"}};
                continue;
            }
            double real;
            try {
                if (it.value().is_string()) real = std::stod(it.value().get<std::string>());
                else                        real = it.value().get<double>();
            } catch (...) {
                results[name] = {{"ok", false}, {"error", "bad value"}};
                continue;
            }
            uint16_t raw = encode(real, m_it->second);
            int rc = modbus_write_register(mb.ctx, m_it->second.addr, raw);
            if (rc == -1) {
                results[name] = {{"ok", false}, {"error", modbus_strerror(errno)}};
                disconnect(mb);
            } else {
                results[name] = {{"ok", true}, {"raw", raw}};
            }
        }
        res.set_content(json{{"ok", true}, {"results", results}}.dump(),
                        "application/json");
    });

    std::cout << "Dashboard on http://0.0.0.0:" << HTTP_PORT
              << "  ->  robot " << ROBOT_IP << ":" << ROBOT_PORT << std::endl;

    srv.listen("0.0.0.0", HTTP_PORT);

    if (mb.ctx) modbus_free(mb.ctx);
    return 0;
}
