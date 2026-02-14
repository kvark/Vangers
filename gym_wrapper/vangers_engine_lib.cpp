#include "vangers_engine_lib.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <dlfcn.h>
#include <cstdint>

/*
 * Use the engine's headers directly for declarations. Do not duplicate
 * the wrapper header include or provide redundant extern "C" declarations
 * here — rely on the engine headers to provide the correct C++ linkage
 * and declarations. This keeps the translation unit consistent with the
 * rest of the engine build.
 */
#include "../src/common.h"
#include "../lib/xgraph/xgraph.h"
#include "../src/actint/item_api.h"
#include "../src/units/uvsapi.h"
#include "../src/network.h"
#include "../src/sqexp.h"
#include "../src/units/mechos.h"
#include "../src/3d/3dobject.h"
 
extern XGR_Screen XGR_Obj;
extern int xgrScreenSizeX;
extern int xgrScreenSizeY;


// NOTE:
// The gym wrapper must avoid tightly coupling to engine-internal headers here.
// To reduce the risk of include-order / dependency problems we treat engine
// types opaquely in this translation unit and only use forward declarations
// for the symbols we call. The full engine core is linked into the shared
// library at link time, so these externs will resolve then.
//
// Avoid including heavy engine headers in this file to keep the wrapper
// changes isolated to the `gym_wrapper` directory (per project policy).

struct uvsVanger;
struct VangerUnit;
struct iGameMap;

// Minimal forward declarations for functions we call into the engine core.
// Implementations are provided by the engine core when linked into the
// shared `vangers_engine` target.
extern uvsVanger* FindFreeVanger();
extern VangerUnit* addVanger(uvsVanger* p, int x, int y, int Human);

// Core engine entrypoints we call directly (real engine in same repo exposes these)
extern void gameQuant();
extern void FIRE_ALL_WEAPONS();
extern void vMapPrepare(const char* name, int nWorld);
extern void vMapInit(void);
extern void MLload(void);

// The main game may export a current game map pointer; declare it opaque here.
extern iGameMap* curGMap;

// Minimal game integration - avoid complex header dependencies for now
// This provides a working gym interface that can be extended later
 
// NOTE: Wrapper-introduced state has been moved into the `GameSimulation`
// and `InstanceManager` classes to avoid process-wide statics. The original
// game's own globals (from the Vangers engine) are still used when necessary,
// but wrapper state (player_state, deterministic seeds, init flags, and gym-mode)
// are instance-managed to allow safer testing and potential multi-instance use.

using namespace vangers_engine;

// Global state for gym integration is now managed by InstanceManager and
// per-GameSimulation instance members. See InstanceManager::set_gym_mode/get_gym_mode
// and GameSimulation::{set,is}_game_initialized / access_internal_player_state.

// Event queue implementation
void EventQueue::push_event(int type, float value, float x, float y) {
    std::lock_guard<std::mutex> lock(mutex_);
    GameEvent event;
    event.type = type;
    event.value = value;
    event.x = x;
    event.y = y;
    event.timestamp = timestamp_counter_++;
    events_.push(event);

    // Limit queue size to prevent memory bloat
    while (events_.size() > 100) {
        events_.pop();
    }
}

int EventQueue::get_events(GameEvent* events, int max_events) {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    while (!events_.empty() && count < max_events) {
        events[count] = events_.front();
        events_.pop();
        count++;
    }
    return count;
}

void EventQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!events_.empty()) {
        events_.pop();
    }
}

// Game simulation wrapper - manages external Vangers process
GameSimulation::GameSimulation(int width, int height)
    : screen_width_(width), screen_height_(height) {
    std::cout << "Creating game simulation " << width << "x" << height << std::endl;
}

GameSimulation::~GameSimulation() {
    // Only perform shutdown if the simulation is still initialized. This avoids
    // duplicate shutdown messages when the owning GameEngineInstance already
    // invoked shutdown() prior to destruction.
    if (is_game_initialized()) {
        shutdown();
    }
}

bool GameSimulation::initialize() {
    std::cout << "Initializing game simulation" << std::endl;

    // Set up deterministic mode
    setup_deterministic_mode();

    // Initialize the full game engine subsystems (if not already)
    if (!is_game_initialized()) {
        std::cout << "Initializing full game systems..." << std::endl;
 
        // Seed legacy RNGs used by game subsystems (moved into instance state)
        this->legacy_rndval_ = 83;
        this->legacy_real_rndval_ = 12345;
         
        // Initialize core 3D engine tables and draw buffers
        std::cout << "Calling graph3d_init()..." << std::endl;
        graph3d_init();
        std::cout << "graph3d_init() done." << std::endl;

        // Initialize offscreen graphics if headless mode is requested or SDL driver is set.
        bool want_headless = headless_mode_;
        const char* sdl_driver = std::getenv("SDL_VIDEODRIVER");
        if (!want_headless && sdl_driver && sdl_driver[0] != '\0') {
            want_headless = true;
        }
        if (want_headless) {
            if (headless_mode_) {
                // Ensure SDL uses dummy drivers before any SDL initialization occurs.
                setenv("SDL_VIDEODRIVER", "dummy", 0);
                setenv("SDL_AUDIODRIVER", "dummy", 0);
            }
            init_graphics_headless();
        } else {
            std::cout << "SDL_VIDEODRIVER not set; skipping XGR init (may be unstable for rendering)." << std::endl;
            xgrScreenSizeX = screen_width_;
            xgrScreenSizeY = screen_height_;
        }
         
        // Prepare UVS (worlds, escaves, vangers, items)
        std::cout << "Calling uniVangPrepare()..." << std::endl;
        uniVangPrepare();
        std::cout << "uniVangPrepare() done." << std::endl;

        // Initialize general object systems, dispatchers, palettes, resources, etc.
        std::cout << "Calling GeneralSystemInit()..." << std::endl;
        GeneralSystemInit();
        std::cout << "GeneralSystemInit() done." << std::endl;

        // Create a temporary world list file for gym usage
        {
            std::ofstream world_lst("gym_world.lst");
            if (world_lst.is_open()) {
                world_lst << "1\n";
                world_lst << "Fostral thechain/fostral/world.ini\n";
                world_lst.close();
            } else {
                std::cerr << "Failed to create gym_world.lst" << std::endl;
            }
        }

        // Load default map to ensure path variables are set for GeneralSystemOpen
        std::cout << "Calling vMapPrepare()..." << std::endl;
        ::vMapPrepare("gym_world.lst", 0);
        ::vMapInit();
        std::cout << "vMapPrepare() done." << std::endl;
        std::cout << "Calling MLload()..." << std::endl;
        ::MLload();
        std::cout << "MLload() done." << std::endl;
         
        // Finalize/open system resources into active structures
        std::cout << "Calling GeneralSystemOpen()..." << std::endl;
        GeneralSystemOpen();
        std::cout << "GeneralSystemOpen() done." << std::endl;
         
        // Try to spawn a player vanger from UVS into the active units and hook it up.
        // This attempts to connect the wrapper's `player_unit_` to a real engine VangerUnit.
        uvsVanger* pv = nullptr;
        try {
            pv = FindFreeVanger();
        } catch (...) {
            pv = nullptr;
        }
        if (pv) {
            VangerUnit* vv = nullptr;
            try {
                vv = addVanger(pv, xgrScreenSizeX / 2, xgrScreenSizeY / 2, 1);
            } catch (...) {
                vv = nullptr;
            }
            if (vv) {
                player_unit_ = vv;
                std::cout << "Player vanger spawned and connected to engine" << std::endl;
            } else {
                std::cout << "Warning: addVanger returned null (player not created)" << std::endl;
            }
        } else {
            std::cout << "Warning: no UVS vanger available to spawn (FindFreeVanger returned null)" << std::endl;
        }
         
        set_game_initialized(true);
        std::cout << "Full game systems initialized" << std::endl;
    }

    // Initialize physics and AI systems (deterministic)
    init_physics_deterministic();
    init_ai_systems();

    // Setup event hooks
    setup_event_hooks();

    // Require the engine to provide a real player unit - do not create a placeholder.
    if (!player_unit_) {
        create_player_unit();
        if (!player_unit_) {
            std::cerr << "Error: GameSimulation failed to obtain a player unit from engine. Aborting initialization." << std::endl;
            return false;
        }
    }

    // Ensure we have a real game map available; do not rely on minimal stubs.
    if (!game_map_) {
        create_game_map();
        if (!game_map_) {
            std::cerr << "Error: GameSimulation failed to attach to an engine iGameMap (curGMap). Aborting initialization." << std::endl;
            return false;
        }
    }

    frame_count_ = 0;
    last_physics_time_ = 0;

    std::cout << "Game simulation initialized successfully" << std::endl;
    return true;
}

void GameSimulation::shutdown() {
    // Idempotent shutdown: if already not initialized, do nothing.
    if (!is_game_initialized()) {
        return;
    }

    std::cout << "Shutting down game simulation" << std::endl;

    // Clean up resources
    player_unit_ = nullptr;
    game_map_ = nullptr;

    if (headless_mode_) {
        XGR_Finit();
    }

    // Mark this simulation as not initialized (instance-local)
    set_game_initialized(false);
}

void GameSimulation::reset() {
    std::cout << "Resetting game simulation" << std::endl;
     
    frame_count_ = 0;
    last_physics_time_ = 0;
     
    std::cout << "Resetting player to spawn position" << std::endl;
     
    // Reset minimal player state (instance-local)
    player_state_.pos_x = 256.0f * 256.0f;
    player_state_.pos_y = 256.0f * 256.0f;
    player_state_.pos_z = 0.0f;
    player_state_.vel_x = 0.0f;
    player_state_.vel_y = 0.0f;
    player_state_.vel_z = 0.0f;
    player_state_.angle = 0.0f;
    player_state_.armor = player_state_.armor_max;
    player_state_.energy = player_state_.energy_max;
    player_state_.speed = 0;
    player_state_.alive = true;
    player_state_.on_ground = true;
    player_state_.in_water = false;
    player_state_.step_count = 0;
     
    std::cout << "Player reset complete" << std::endl;
}

void GameSimulation::step(int num_steps) {
    for (int i = 0; i < num_steps; i++) {
        frame_count_++;
        player_state_.step_count++;
        
        // Execute one quantum of the game engine
        if (curGMap) {
            gameQuant();
        }
         
        last_physics_time_ = frame_count_;
         
        // Small delay to prevent excessive CPU usage (instance-managed gym-mode)
        if (!InstanceManager::instance().get_gym_mode()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void GameSimulation::set_time_scale(float scale) {
    time_scale_ = scale;
    std::cout << "Setting time scale to: " << scale << std::endl;
}

void GameSimulation::apply_action(const Action& action) {
    // TODO: Map action to engine inputs (VangerUnit control flags or key injection)
    // For now, just logging or relying on AI if enabled.
    (void)action;
}

void GameSimulation::render_frame(unsigned char* buffer, int width, int height) {
    // Prefer using the software rasterizer (XGR_Obj) if available.
    // Fall back to the animated test pattern when the renderer isn't ready
    // or sizes mismatch.
    if (width != screen_width_ || height != screen_height_) {
        std::cout << "Warning: frame size mismatch" << std::endl;
    }

    // Attempt to obtain the internal XGR screen buffers.
    uint8_t* screen_indexes = XGR_Obj.get_default_render_buffer();
    uint32_t* screen2d_rgba = XGR_Obj.get_2d_rgba_render_buffer();
    uint8_t* screen2d_indexes = XGR_Obj.get_2d_render_buffer();

    // If XGR is initialized and matches requested size, use it.
    if (screen_indexes && xgrScreenSizeX == width && xgrScreenSizeY == height) {
        // Temporary RGBA buffer (32-bit values) filled by blitRgba
        std::vector<uint32_t> tmp_rgba;
        tmp_rgba.resize(width * height);

        // Ask XGR to produce an RGBA image into our temporary buffer.
        // blitRgba will use palette cache / 2D overlays as the original engine does.
        XGR_Obj.blitRgba(tmp_rgba.data(), screen_indexes, screen2d_rgba, screen2d_indexes);

        // Convert 32-bit pixels to triplet RGB bytes in buffer (R,G,B).
        // XGR32_PaletteCache / SDL surface mapping uses layout where
        // red is in bits 16-23, green 8-15, blue 0-7 (0x00RRGGBB or 0xAARRGGBB).
        const int pixel_count = width * height;
        for (int i = 0; i < pixel_count; ++i) {
            uint32_t px = tmp_rgba[i];
            buffer[i * 3 + 0] = static_cast<unsigned char>((px >> 16) & 0xFF); // R
            buffer[i * 3 + 1] = static_cast<unsigned char>((px >> 8) & 0xFF);  // G
            buffer[i * 3 + 2] = static_cast<unsigned char>(px & 0xFF);         // B
        }
        return;
    }

    // Fallback: generate animated test pattern (previous behavior)
    float phase = frame_count_ * 0.01f;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 3;

            // Create animated test pattern
            float fx = x / float(width);
            float fy = y / float(height);

            float r = 0.5f + 0.3f * sin(fx * 10 + phase);
            float g = 0.5f + 0.3f * sin(fy * 10 + phase * 1.1f);
            float b = 0.5f + 0.3f * sin((fx + fy) * 8 + phase * 0.9f);

            buffer[index] = (unsigned char)(r * 255);
            buffer[index + 1] = (unsigned char)(g * 255);
            buffer[index + 2] = (unsigned char)(b * 255);
        }
    }
}



void GameSimulation::create_player_unit() {
    std::cout << "Creating player unit" << std::endl;
     
    // Attempt to hook into the real engine and spawn/attach a real VangerUnit.
    // Keep logic conservative: do not reference optional globals like `Gamer`
    // which may not be present in every build configuration. Prefer to query
    // FindFreeVanger() and call addVanger() if available.
    player_unit_ = nullptr;
    VangerUnit* created = nullptr;

    try {
        uvsVanger* freev = nullptr;
        try {
            freev = FindFreeVanger();
        } catch (...) {
            freev = nullptr;
        }

        if (freev) {
            // Use a sensible spawn location (screen center) when engine map coords
            // are not safely available without including heavy headers.
            int sx = screen_width_ / 2;
            int sy = screen_height_ / 2;
            try {
                created = addVanger(freev, sx, sy, 1);
            } catch (...) {
                created = nullptr;
            }

            if (created) {
                player_unit_ = created;
                std::cout << "Spawned player from free UVS vanger" << std::endl;
                return;
            }
        }
    } catch (...) {
        created = nullptr;
    }

    // Do not create a minimal placeholder. Fail fast and let the caller handle initialization failure.
    std::cerr << "Error: Failed to obtain a real engine VangerUnit. No placeholder will be created." << std::endl;
    player_unit_ = nullptr;
}

void GameSimulation::create_game_map() {
    std::cout << "Creating game map" << std::endl;

    // If the real engine has already created the current game map, attach to it.
    // Do not dereference curGMap fields here to avoid requiring engine internals.
    if (curGMap) {
        game_map_ = curGMap;
        std::cout << "Attached to engine game map (curGMap)" << std::endl;
        return;
    }

    // Create a fallback iGameMap if the engine did not create one during boot.
    // This mirrors the core engine's initialization path and is sufficient for headless RL.
    int cx = xgrScreenSizeX > 0 ? xgrScreenSizeX / 2 : XGR_MAXX / 2;
    int cy = xgrScreenSizeY > 0 ? xgrScreenSizeY / 2 : XGR_MAXY / 2;
    int xside = std::max(1, cx);
    int yside = std::max(1, cy);

    try {
        curGMap = new iGameMap(cx, cy, xside, yside);
        game_map_ = curGMap;
        std::cout << "Created fallback game map (curGMap) for headless mode" << std::endl;
        return;
    } catch (...) {
        // Fall through to failure.
    }

    game_map_ = nullptr;
    std::cerr << "Error: Failed to create fallback game map (curGMap)." << std::endl;
}

void GameSimulation::setup_deterministic_mode() {
    std::cout << "Setting up deterministic mode" << std::endl;
    deterministic_mode_ = true;
    util::rng.set_seed(12345);
 
    // Also initialize legacy RNG values used elsewhere in the minimal systems.
    legacy_rndval_ = 83;
    legacy_real_rndval_ = 12345;
 
    // Ensure stable starting times
    frame_count_ = 0;
    last_physics_time_ = 0;
}

void GameSimulation::disable_real_time_features() {
    std::cout << "Disabling real-time features" << std::endl;
 
    // When running in deterministic/headless mode we avoid sleeping in step loops
    // and disable any OS-timed effects. Mark both the instance and the global manager.
    set_gym_mode_enabled(true);
    InstanceManager::instance().set_gym_mode(true);
}

void GameSimulation::init_graphics_headless() {
    std::cout << "Initializing headless graphics" << std::endl;

    // Try to create XGR software surfaces at the requested simulation resolution.
    // Use public API that encapsulates internal surface creation.
    try {
        // If XGR_Obj has an init function, call it with default flags (safe no-op if already initialized).
        // Note: XGR_Init is a macro wrapper for XGR_Obj.init(...)
#ifdef XGR_Init
        XGR_Init(XGR_INIT);
#endif

        // Set the resolution which triggers creation of internal surfaces.
        XGR_Obj.set_resolution(screen_width_, screen_height_);

        // Clear 2D overlays and ensure default buffer is active before our first render.
        XGR_Obj.clear_2d_surface();
        XGR_Obj.set_default_render_buffer();

        std::cout << "Headless software rasterizer initialized: " << screen_width_ << "x" << screen_height_ << std::endl;
    } catch (...) {
        // If anything goes wrong, fallback behavior will still produce frames via the test pattern.
        std::cout << "Warning: failed to initialize headless graphics, falling back to test pattern" << std::endl;
    }
}

void GameSimulation::init_physics_deterministic() {
    std::cout << "Initializing deterministic physics" << std::endl;

    // Configure simulation timing for deterministic stepping.
    // Use time_scale_ to control the effective dt used in step().
    time_scale_ = 1.0f;

    // Ensure physics substep bookkeeping is initialized.
    last_physics_time_ = 0;

    // In deterministic mode disable any time-based randomness or real-time waits.
    disable_real_time_features();
}

void GameSimulation::init_ai_systems() {
    std::cout << "Initializing AI systems" << std::endl;

    // Ensure deterministic RNG seed is set for AI consumers
    util::rng.set_seed(util::rng.get_seed());

    // When running with the full engine (in-repo), VangerUnit provides its own AI
    // initialization method. If we already have a player unit attached, call its
    // InitAI() directly to initialize per-unit AI subsystems deterministically.
    if (player_unit_) {
        try {
            player_unit_->InitAI();
            std::cout << "Player unit AI initialized via VangerUnit::InitAI()" << std::endl;
        } catch (...) {
            std::cerr << "Warning: VangerUnit::InitAI() threw an exception; continuing." << std::endl;
        }
    } else {
        // No player unit yet - defer AI initialization until unit is created.
        std::cout << "No player unit present; AI init deferred until player spawn." << std::endl;
    }
}

void GameSimulation::setup_event_hooks() {
    std::cout << "Setting up event hooks" << std::endl;

    // The Vangers engine exposes internal event systems. For this wrapper we rely
    // on instance-level EventQueue for delivering events into the Python layer.
    // If the engine has public callback registration points with a known ABI,
    // they can be wired here. At present, we do not perform runtime symbol
    // discovery for callbacks; instead we record engine events via GameEngineInstance
    // and EventQueue::push_event where appropriate.
}

// Game engine instance implementation
GameEngineInstance::GameEngineInstance(int width, int height)
    : screen_width_(width), screen_height_(height), render_mode_(1) {
    instance_id_ = reinterpret_cast<void*>(this);
    frame_buffer_.resize(width * height * 3, 128); // Gray background
    simulation_ = std::make_unique<GameSimulation>(width, height);
}

void GameEngineInstance::set_headless_mode(bool headless) {
    headless_mode_ = headless;
    if (simulation_) {
        simulation_->set_headless_mode(headless);
    }
}

GameEngineInstance::~GameEngineInstance() {
    shutdown();
}

bool GameEngineInstance::initialize() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (initialized_) {
        return true;
    }

    std::cout << "Initializing game engine instance " << screen_width_ << "x" << screen_height_ << std::endl;

    // Ensure default render mode is RGB (1) unless explicitly changed later
    render_mode_ = 1;

    if (!simulation_->initialize()) {
        std::cerr << "Failed to initialize simulation" << std::endl;
        return false;
    }

    // Warm up frame buffer by rendering an initial frame so clients immediately
    // receive non-empty data. update_frame_buffer() is safe to call here.
    try {
        update_frame_buffer();
    } catch (...) {
        std::cout << "Warning: failed to warm up frame buffer during initialization" << std::endl;
    }

    // Set as current instance (managed by InstanceManager)
    InstanceManager::instance().set_current_instance(this);
     
    initialized_ = true;
    return true;
}

void GameEngineInstance::shutdown() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!initialized_) {
        return;
    }

    std::cout << "Shutting down game engine instance" << std::endl;

    if (simulation_) {
        // Ensure the simulation is cleanly shut down and then release ownership so
        // the simulation destructor is not invoked again from elsewhere and cause
        // duplicate shutdown activity / messages.
        simulation_->shutdown();
        simulation_.reset();
    }

    // Clear current instance pointer via InstanceManager if it points to us.
    if (InstanceManager::instance().get_current_instance() == this) {
        InstanceManager::instance().set_current_instance(nullptr);
    }

    initialized_ = false;
}

int GameEngineInstance::reset() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!initialized_) {
        std::cerr << "Cannot reset uninitialized instance" << std::endl;
        return 0;
    }

    std::cout << "Resetting game engine instance" << std::endl;

    simulation_->reset();
    event_queue_.clear();
    simulation_step_count_ = 0;
    current_action_ = Action(); // Reset to default

    return 1;
}

int GameEngineInstance::step_simulation(int num_steps) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!initialized_ || paused_) {
        return 0;
    }

    if (num_steps <= 0) {
        // Nothing to do - treat as success
        return 1;
    }

    VangerUnit* real_unit = nullptr;
    if (simulation_) {
        real_unit = simulation_->get_player_unit();
    }

    // Simple profiling / timing for this step invocation
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    try {
        // Capture current action deterministically (locks action mutex internally)
        Action action = get_current_action();

        // Honor configured physics substeps: number of internal physics steps per logical frame
        int substeps = std::max(1, physics_substeps_);

        // Perform deterministic stepping: for each logical frame, apply the action and advance physics
        for (int f = 0; f < num_steps; ++f) {
            // If the engine provided a real VangerUnit and a real map, drive the real engine.
            if (real_unit) {
                // Map our wrapper Action to engine controls. Use the engine's CONTROLS enum.
                // Movement
                if (action.movement == Action::FORWARD) {
                    real_unit->controls(CONTROLS::TRACTION_INCREASE);
                } else if (action.movement == Action::BACKWARD) {
                    real_unit->controls(CONTROLS::TRACTION_DECREASE);
                }
                // Steering
                if (action.steering == Action::LEFT) {
                    real_unit->controls(CONTROLS::STEER_LEFT);
                } else if (action.steering == Action::RIGHT) {
                    real_unit->controls(CONTROLS::STEER_RIGHT);
                }
                // Fire - use high-level fire entrypoint used by the engine
                if (action.fire == Action::FIRE) {
                    FIRE_ALL_WEAPONS();
                }
                // Specials mapped to virtual up/down controls (example mapping)
                if (action.special1 == Action::USE_SPECIAL) {
                    real_unit->controls(CONTROLS::VIRTUAL_UP);
                }
                if (action.special2 == Action::USE_SPECIAL) {
                    real_unit->controls(CONTROLS::VIRTUAL_DOWN);
                }

                // Advance the engine's main quant/step routine deterministically.
                for (int s = 0; s < substeps; ++s) {
                    gameQuant();
                }

            } else if (simulation_) {
                // Defensive fallback (should not be taken in "full engine" mode)
                simulation_->apply_action(action);
                simulation_->step(substeps);
            } else {
                std::cerr << "No engine simulation or player unit available to step" << std::endl;
                return 0;
            }

            // Increment the simulation step counter in terms of internal physics steps
            simulation_step_count_ += static_cast<uint64_t>(substeps);
        }

        // After stepping, update visual/frame state once
        update_frame_buffer();

        // Handle any generated events
        handle_game_events();

        auto t1 = clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // Log light-weight profiling info. Keep it low-volume to avoid spamming in tight loops.
        if (ms > 0) {
            std::cout << "step_simulation: frames=" << num_steps
                      << " substeps=" << substeps
                      << " internal_steps=" << (num_steps * substeps)
                      << " time_ms=" << ms << std::endl;
        } else {
            // Very fast: still provide concise feedback occasionally (every call)
            std::cout << "step_simulation: frames=" << num_steps
                      << " substeps=" << substeps
                      << " time_ms=" << ms << std::endl;
        }

        return 1;

    } catch (const std::exception& e) {
        std::cerr << "Error in simulation step: " << e.what() << std::endl;
        return 0;
    }
}

void GameEngineInstance::set_action(const Action& action) {
    std::lock_guard<std::mutex> lock(action_mutex_);
    current_action_ = action;
}

Action GameEngineInstance::get_current_action() const {
    std::lock_guard<std::mutex> lock(action_mutex_);
    return current_action_;
}

bool GameEngineInstance::get_player_state(PlayerState& state) const {
    std::lock_guard<std::mutex> lock(state_mutex_);

    if (!initialized_) {
        return false;
    }

    // If we have a real engine player unit attached, prefer its live state
    VangerUnit* real_unit = nullptr;
    if (simulation_) {
        real_unit = simulation_->get_player_unit();
    }

    if (!real_unit && player_unit_) {
        // If instance-level player_unit_ is set, prefer that
        real_unit = player_unit_;
    }

    if (real_unit) {
        // Extract from the real engine unit via util helper
        util::extract_player_state(real_unit, state);

        // Fill small additional defaults
        state.pitch = 0.0f;
        state.roll = 0.0f;

    } else if (simulation_) {
        // Fallback to instance-local simulated state (should be rare in full-engine mode)
        const auto& ps = simulation_->get_internal_player_state();

        state.position.x = ps.pos_x;
        state.position.y = ps.pos_y;
        state.position.z = ps.pos_z;

        state.velocity.x = ps.vel_x;
        state.velocity.y = ps.vel_y;
        state.velocity.z = ps.vel_z;

        state.angle = ps.angle;
        state.pitch = 0.0f;
        state.roll = 0.0f;

        state.armor = ps.armor;
        state.energy = ps.energy;
        state.speed = ps.speed;
        state.max_speed = ps.max_speed;

        state.alive = ps.alive;
        state.on_ground = ps.on_ground;
        state.in_water = ps.in_water;
    } else {
        // Conservative defaults if no simulation or engine unit exists
        state.position.x = state.position.y = state.position.z = 0.0f;
        state.velocity.x = state.velocity.y = state.velocity.z = 0.0f;
        state.angle = state.pitch = state.roll = 0.0f;

        state.armor = 100;
        state.energy = 100;
        state.speed = 0;
        state.max_speed = 200;

        state.alive = true;
        state.on_ground = true;
        state.in_water = false;
    }

    state.volume = 0;
    state.max_volume = 100;
    state.money = 1000;

    return true;
}

bool GameEngineInstance::get_frame_buffer(unsigned char* buffer, int buffer_size) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    int required_size = screen_width_ * screen_height_ * 3;
    if (buffer_size < required_size) {
        std::cerr << "Frame buffer too small: " << buffer_size << " < " << required_size << std::endl;
        return false;
    }

    if (frame_buffer_.size() >= static_cast<size_t>(required_size)) {
        memcpy(buffer, frame_buffer_.data(), required_size);
        return true;
    }

    return false;
}

int GameEngineInstance::get_events(GameEvent* events, int max_events) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    return event_queue_.get_events(events, max_events);
}

void GameEngineInstance::clear_events() {
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.clear();
}

VangerUnit* GameEngineInstance::get_player_unit() const {
    if (simulation_) {
        return simulation_->get_player_unit();
    }
    return nullptr;
}

iGameMap* GameEngineInstance::get_game_map() const {
    if (simulation_) {
        return simulation_->get_game_map();
    }
    return nullptr;
}

void GameEngineInstance::set_map(const std::string& map_name) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::cout << "Setting map: " << map_name << std::endl;
    // TODO: Implement map loading
}

void GameEngineInstance::update_frame_buffer() {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    if (simulation_ && render_mode_ > 0) {
        simulation_->render_frame(frame_buffer_.data(), screen_width_, screen_height_);
    }
}

void GameEngineInstance::update_player_state() {
    // Player state is updated in get_player_state()
}

void GameEngineInstance::process_action() {
    // Actions are processed immediately when set
}

void GameEngineInstance::advance_physics(int steps) {
    if (simulation_) {
        simulation_->step(steps);
    }
}

void GameEngineInstance::handle_game_events() {
    // Generate mock events for demonstration
    event_counter_++;

    // Generate collision event every 100 steps
    if (event_counter_ % 100 == 0) {
        event_queue_.push_event(0, 10.0f, 100.0f, 200.0f); // Collision
    }

    // Generate item pickup every 150 steps
    if (event_counter_ % 150 == 0) {
        event_queue_.push_event(1, 1.0f, 150.0f, 250.0f); // Item collected
    }

    // Generate damage event every 200 steps
    if (event_counter_ % 200 == 0) {
        event_queue_.push_event(3, 5.0f, 0.0f, 0.0f); // Damage taken
    }
}

void GameEngineInstance::on_collision(VangerUnit* /*unit*/, GeneralObject* /*obj*/, float damage) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.push_event(0, damage, 0.0f, 0.0f);
    std::cout << "Collision event: damage=" << damage << std::endl;
}

void GameEngineInstance::on_item_collected(VangerUnit* /*unit*/, StuffObject* /*item*/) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.push_event(1, 1.0f, 0.0f, 0.0f);
    std::cout << "Item collected event" << std::endl;
}

void GameEngineInstance::on_objective_completed() {
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.push_event(2, 100.0f, 0.0f, 0.0f);
    std::cout << "Objective completed event" << std::endl;
}

void GameEngineInstance::on_player_damage(VangerUnit* /*unit*/, float damage) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    event_queue_.push_event(3, damage, 0.0f, 0.0f);
    std::cout << "Player damage event: " << damage << std::endl;
}

// Instance manager implementation
InstanceManager& InstanceManager::instance() {
    static InstanceManager manager;
    return manager;
}

InstanceManager::~InstanceManager() {
    cleanup_all();
}

void* InstanceManager::create_instance(int width, int height) {
    return create_instance_with_options(width, height, false);
}

void* InstanceManager::create_instance_with_options(int width, int height, bool headless) {
    void* id = reinterpret_cast<void*>(next_id_.fetch_add(1));
    auto instance = std::make_unique<GameEngineInstance>(width, height);
    instance->set_headless_mode(headless);

    if (instance->initialize()) {
        {
            std::lock_guard<std::mutex> lock(instances_mutex_);
            instances_[id] = std::move(instance);
        }
        std::cout << "Created gym instance: " << id << " (" << width << "x" << height << ")" << std::endl;
        return id;
    } else {
        std::cerr << "Failed to initialize gym instance" << std::endl;
        return nullptr;
    }
}

void InstanceManager::destroy_instance(void* handle) {
    std::unique_ptr<GameEngineInstance> instance;
    {
        std::lock_guard<std::mutex> lock(instances_mutex_);
        auto it = instances_.find(handle);
        if (it == instances_.end()) {
            std::cerr << "Attempted to destroy non-existent instance: " << handle << std::endl;
            return;
        }
        instance = std::move(it->second);
        instances_.erase(it);
    }

    std::cout << "Destroying gym instance: " << handle << std::endl;
    // Ensure the instance is cleanly shut down after releasing the manager lock.
    try {
        if (instance) {
            instance->shutdown();
        }
    } catch (...) {
        // Best-effort: ignore exceptions during shutdown to avoid throwing from destructor paths.
    }
}

GameEngineInstance* InstanceManager::get_instance(void* handle) {
    std::lock_guard<std::mutex> lock(instances_mutex_);

    auto it = instances_.find(handle);
    if (it != instances_.end()) {
        return it->second.get();
    }

    return nullptr;
}

void InstanceManager::cleanup_all() {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    std::cout << "Cleaning up all gym instances (" << instances_.size() << ")" << std::endl;
    instances_.clear();
}

size_t InstanceManager::get_num_instances() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(instances_mutex_));
    return instances_.size();
}

// Utility functions
namespace vangers_engine {
namespace util {

Vector3 normalize_position(const Vector3& pos) {
    Vector3 result;
    result.x = std::max(-1.0f, std::min(1.0f, pos.x / 1000.0f));
    result.y = std::max(-1.0f, std::min(1.0f, pos.y / 1000.0f));
    result.z = std::max(-1.0f, std::min(1.0f, pos.z / 100.0f));
    return result;
}

Vector3 normalize_velocity(const Vector3& vel) {
    Vector3 result;
    result.x = std::max(-1.0f, std::min(1.0f, vel.x / 100.0f));
    result.y = std::max(-1.0f, std::min(1.0f, vel.y / 100.0f));
    result.z = std::max(-1.0f, std::min(1.0f, vel.z / 50.0f));
    return result;
}

float normalize_angle(float angle) {
    // Normalize to [-π, π] then to [-1, 1]
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle / M_PI;
}

Action convert_action(int movement, int steering, int fire, int special1, int special2) {
    Action action;
    action.movement = static_cast<Action::Movement>(movement);
    action.steering = static_cast<Action::Steering>(steering);
    action.fire = static_cast<Action::Fire>(fire);
    action.special1 = static_cast<Action::Special>(special1);
    action.special2 = static_cast<Action::Special>(special2);
    return action;
}

void extract_player_state(const VangerUnit* unit, PlayerState& state) {
    // Prefer extracting directly from the engine's VangerUnit when available.
    // Initialize to safe defaults first.
    state = PlayerState{};
    if (!unit) {
        return;
    }

    // Position (engine stores current coords in R_curr)
    state.position.x = static_cast<float>(unit->R_curr.x);
    state.position.y = static_cast<float>(unit->R_curr.y);
    state.position.z = static_cast<float>(unit->R_curr.z);

    // Angle and speed (ActionUnit / TrackUnit provide Angle and Speed)
    // Use defensively in case fields are not present in some build variants.
    float angle = 0.0f;
    int speed = 0;
    try {
        angle = static_cast<float>(unit->Angle);
        speed = static_cast<int>(unit->Speed);
    } catch (...) {
        angle = 0.0f;
        speed = 0;
    }
    state.angle = angle;

    // Approximate velocity from angle+speed if explicit velocity vectors are not exposed.
    state.velocity.x = cosf(angle) * static_cast<float>(speed);
    state.velocity.y = sinf(angle) * static_cast<float>(speed);
    state.velocity.z = 0.0f;

    // Armor / energy fields are provided by uvsUnitType base
    try {
        state.armor = static_cast<int>(unit->Armor);
        state.energy = static_cast<int>(unit->Energy);
    } catch (...) {
        state.armor = 0;
        state.energy = 0;
    }

    state.speed = speed;

    // Attempt to read uvsMaxSpeed if available
    try {
        state.max_speed = static_cast<int>(unit->uvsMaxSpeed);
    } catch (...) {
        state.max_speed = state.speed > 0 ? state.speed : 200;
    }

    // Basic flags
    state.alive = (unit->PlayerDestroyFlag == 0);
    state.on_ground = true;  // not directly available here; assume true
    state.in_water = false;  // not directly available here; assume false

    // Other fields - engine exposes numerous per-unit properties; fill sensible defaults
    state.volume = 0;
    state.max_volume = 100;
    state.money = 0;
}

void convert_frame_format(const unsigned char* src, unsigned char* dst,
                         int width, int height, int src_format, int dst_format) {
    // Simple format conversion
    int size = width * height * 3;

    if (src_format == dst_format || !src || !dst) {
        if (src && dst) {
            memcpy(dst, src, size);
        }
        return;
    }

    // TODO: Implement proper format conversion between different pixel formats
    memcpy(dst, src, size);
}

thread_local DeterministicRNG rng;

uint32_t DeterministicRNG::next() {
    state_ = (state_ * 1664525 + 1013904223);
    return state_;
}

float DeterministicRNG::next_float() {
    return next() / float(0xFFFFFFFFU);
}

int DeterministicRNG::next_int(int min_val, int max_val) {
    if (min_val >= max_val) return min_val;
    return min_val + (next() % (max_val - min_val));
}

} // namespace util
} // namespace vangers_engine

// Global state moved to InstanceManager (engine_initialized_)

// C API implementation
extern "C" {

int vangers_engine_init_with_path(const char* resource_path) {
    if (InstanceManager::instance().get_gym_mode()) {
        return 1;
    }

    // Use provided resource path or fall back to the known working path.
    std::string path = resource_path ? std::string(resource_path) : std::string("/x/Work/VangersData");

    // Verify the path exists. If it does not, log a warning and continue using the
    // current working directory so the library can still function in minimal/test setups.
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        std::cerr << "Warning: resource path not found: " << path << " (continuing with current working directory)" << std::endl;
    } else {
        // If the path does exist, try to chdir into it to make subsequent file loads relative to it.
        if (chdir(path.c_str()) != 0) {
            std::cerr << "Warning: failed to chdir to resource path: " << path << " (continuing)" << std::endl;
        }
    }

    std::cout << "Initializing Vangers gym engine library v1.0.0 (resources: " << path << ")" << std::endl;
    InstanceManager::instance().set_gym_mode(true);
    return 1;
}

int vangers_engine_init() {
    // Default initialization uses the known data path
    return vangers_engine_init_with_path("/x/Work/VangersData");
}

void vangers_engine_cleanup() {
    if (!InstanceManager::instance().get_gym_mode()) {
        return;
    }

    std::cout << "Cleaning up Vangers gym engine library" << std::endl;

    InstanceManager::instance().cleanup_all();
    InstanceManager::instance().set_gym_mode(false);
    XGR_Finit();
}

void* vangers_create_instance(int width, int height, const VangersCreateOptions* options) {
    if (!InstanceManager::instance().get_gym_mode()) {
        std::cerr << "Engine not initialized - call vangers_engine_init() first" << std::endl;
        return nullptr;
    }

    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        std::cerr << "Invalid dimensions: " << width << "x" << height << std::endl;
        return nullptr;
    }

    bool headless = false;
    if (options && options->size >= sizeof(VangersCreateOptions)) {
        headless = (options->headless != 0);
    }

    return InstanceManager::instance().create_instance_with_options(width, height, headless);
}

void vangers_destroy_instance(void* instance) {
    if (!instance) {
        std::cerr << "Attempted to destroy null instance" << std::endl;
        return;
    }

    InstanceManager::instance().destroy_instance(instance);
}

int vangers_reset_instance(void* instance) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (!eng) {
        std::cerr << "Invalid instance handle in reset" << std::endl;
        return 0;
    }

    return eng->reset();
}

int vangers_step_simulation(void* instance, int num_steps) {
    if (num_steps <= 0) {
        return 1; // No-op for zero or negative steps
    }

    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (!eng) {
        std::cerr << "Invalid instance handle in step" << std::endl;
        return 0;
    }

    return eng->step_simulation(num_steps);
}

void vangers_set_time_scale(void* instance, float scale) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (eng) {
        eng->set_time_scale(std::max(0.1f, std::min(10.0f, scale))); // Clamp scale
    }
}

void vangers_pause_simulation(void* instance, bool paused) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (eng) {
        eng->pause_simulation(paused);
    }
}

void vangers_set_action(void* instance, int movement, int steering, int fire, int special1, int special2) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (!eng) {
        return;
    }

    // Validate action parameters
    if (movement < 0 || movement > 2 || steering < 0 || steering > 2 ||
        fire < 0 || fire > 1 || special1 < 0 || special1 > 1 || special2 < 0 || special2 > 1) {
        std::cerr << "Invalid action parameters" << std::endl;
        return;
    }

    Action action = util::convert_action(movement, steering, fire, special1, special2);
    eng->set_action(action);
}

int vangers_get_player_state(void* instance, PlayerState* state) {
    if (!state) {
        std::cerr << "Null state pointer in get_player_state" << std::endl;
        return 0;
    }

    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (!eng) {
        std::cerr << "Invalid instance handle in get_player_state" << std::endl;
        return 0;
    }

    return eng->get_player_state(*state) ? 1 : 0;
}

int vangers_get_frame_buffer(void* instance, unsigned char* buffer, int buffer_size) {
    if (!buffer || buffer_size <= 0) {
        std::cerr << "Invalid buffer parameters in get_frame_buffer" << std::endl;
        return 0;
    }

    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (!eng) {
        std::cerr << "Invalid instance handle in get_frame_buffer" << std::endl;
        return 0;
    }

    return eng->get_frame_buffer(buffer, buffer_size) ? 1 : 0;
}

int vangers_get_events(void* instance, GameEvent* events, int max_events) {
    if (!events || max_events <= 0) {
        return 0;
    }

    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (!eng) {
        return 0;
    }

    return eng->get_events(events, max_events);
}

void vangers_clear_events(void* instance) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (eng) {
        eng->clear_events();
    }
}

void vangers_set_map(void* instance, const char* map_name) {
    if (!map_name) {
        std::cerr << "Null map name in set_map" << std::endl;
        return;
    }

    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (eng) {
        eng->set_map(map_name);
    }
}

void vangers_set_render_mode(void* instance, int mode) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (eng) {
        eng->set_render_mode(std::max(0, std::min(2, mode))); // Clamp mode
    }
}

void vangers_set_physics_substeps(void* instance, int substeps) {
    GameEngineInstance* eng = InstanceManager::instance().get_instance(instance);
    if (eng) {
        eng->set_physics_substeps(std::max(1, std::min(10, substeps))); // Clamp substeps
    }
}

void vangers_set_debug_mode(bool enabled) {
    std::cout << "Vangers gym debug mode: " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
}

const char* vangers_get_version() {
    return "1.0.0-gym-bridge";
}

int vangers_get_num_instances() {
    return static_cast<int>(InstanceManager::instance().get_num_instances());
}

} // extern "C"
