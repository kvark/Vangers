#include "vangers_engine_lib.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <cmath>

// Minimal game integration - avoid complex header dependencies for now
// This provides a working gym interface that can be extended later

// Simplified game state management
static bool g_GameInitialized = false;
static unsigned g_RNDVAL = 83;
static unsigned g_realRNDVAL = 12345;

// Minimal game state structures
struct MinimalPlayerState {
    float pos_x = 256.0f * 256.0f;
    float pos_y = 256.0f * 256.0f;
    float pos_z = 0.0f;
    float vel_x = 0.0f;
    float vel_y = 0.0f;
    float vel_z = 0.0f;
    float angle = 0.0f;
    int armor = 100;
    int armor_max = 100;
    int energy = 100;
    int energy_max = 100;
    int speed = 0;
    int max_speed = 200;
    bool alive = true;
    bool on_ground = true;
    bool in_water = false;
    int step_count = 0;
};

static MinimalPlayerState g_player_state;

using namespace vangers_engine;

// Global state for gym integration
static bool g_gym_mode_enabled = false;
static GameEngineInstance* g_current_instance = nullptr;

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
    shutdown();
}

bool GameSimulation::initialize() {
    std::cout << "Initializing game simulation" << std::endl;
    
    // Set up deterministic mode
    setup_deterministic_mode();
    
    // Initialize minimal game systems
    if (!g_GameInitialized) {
        std::cout << "Initializing minimal game systems..." << std::endl;
        
        // Initialize random number generator with fixed seed
        g_RNDVAL = 83;
        g_realRNDVAL = 12345;
        
        // Initialize minimal systems
        init_graphics_headless();
        
        g_GameInitialized = true;
        std::cout << "Minimal game systems initialized" << std::endl;
    }
    
    // Initialize physics and AI systems
    init_physics_deterministic();
    init_ai_systems();
    
    // Setup event hooks
    setup_event_hooks();
    
    // Create actual player unit and game map
    create_player_unit();
    create_game_map();
    
    frame_count_ = 0;
    last_physics_time_ = 0;
    
    std::cout << "Game simulation initialized successfully" << std::endl;
    return true;
}

void GameSimulation::shutdown() {
    std::cout << "Shutting down game simulation" << std::endl;
    
    // Clean up resources
    player_unit_ = nullptr;
    game_map_ = nullptr;
}

void GameSimulation::reset() {
    std::cout << "Resetting game simulation" << std::endl;
    
    frame_count_ = 0;
    last_physics_time_ = 0;
    
    std::cout << "Resetting player to spawn position" << std::endl;
    
    // Reset minimal player state
    g_player_state.pos_x = 256.0f * 256.0f;
    g_player_state.pos_y = 256.0f * 256.0f;
    g_player_state.pos_z = 0.0f;
    g_player_state.vel_x = 0.0f;
    g_player_state.vel_y = 0.0f;
    g_player_state.vel_z = 0.0f;
    g_player_state.angle = 0.0f;
    g_player_state.armor = g_player_state.armor_max;
    g_player_state.energy = g_player_state.energy_max;
    g_player_state.speed = 0;
    g_player_state.alive = true;
    g_player_state.on_ground = true;
    g_player_state.in_water = false;
    g_player_state.step_count = 0;
    
    std::cout << "Player reset complete" << std::endl;
}

void GameSimulation::step(int num_steps) {
    for (int i = 0; i < num_steps; i++) {
        frame_count_++;
        g_player_state.step_count++;
        
        // Apply basic physics simulation
        float dt = time_scale_ / 50.0f; // Assume 50fps target
        
        // Apply drag/friction
        g_player_state.vel_x *= 0.95f;
        g_player_state.vel_y *= 0.95f;
        g_player_state.vel_z *= 0.95f;
        
        // Update position based on velocity
        g_player_state.pos_x += g_player_state.vel_x * dt;
        g_player_state.pos_y += g_player_state.vel_y * dt;
        g_player_state.pos_z += g_player_state.vel_z * dt;
        
        // Keep player on ground (simple terrain following)
        if (g_player_state.pos_z < 0) {
            g_player_state.pos_z = 0;
            g_player_state.vel_z = 0;
            g_player_state.on_ground = true;
        }
        
        // Update speed based on velocity magnitude
        g_player_state.speed = (int)sqrt(g_player_state.vel_x * g_player_state.vel_x + 
                                        g_player_state.vel_y * g_player_state.vel_y);
        
        // Simulate energy consumption and regeneration
        if (g_player_state.speed > 0) {
            g_player_state.energy = std::max(0, g_player_state.energy - 1);
        } else if (g_player_state.energy < g_player_state.energy_max) {
            g_player_state.energy = std::min(g_player_state.energy_max, g_player_state.energy + 1);
        }
        
        last_physics_time_ = frame_count_;
        
        // Small delay to prevent excessive CPU usage
        if (!g_gym_mode_enabled) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void GameSimulation::set_time_scale(float scale) {
    time_scale_ = scale;
    std::cout << "Setting time scale to: " << scale << std::endl;
}

void GameSimulation::apply_action(const Action& action) {
    // Apply action to minimal player state simulation
    float acceleration = 50.0f; // Base acceleration
    
    // Movement
    if (action.movement == Action::FORWARD && g_player_state.energy > 0) {
        float cos_angle = cos(g_player_state.angle);
        float sin_angle = sin(g_player_state.angle);
        g_player_state.vel_x += cos_angle * acceleration;
        g_player_state.vel_y += sin_angle * acceleration;
    } else if (action.movement == Action::BACKWARD && g_player_state.energy > 0) {
        float cos_angle = cos(g_player_state.angle);
        float sin_angle = sin(g_player_state.angle);
        g_player_state.vel_x -= cos_angle * acceleration * 0.5f; // Reverse is slower
        g_player_state.vel_y -= sin_angle * acceleration * 0.5f;
    }
    
    // Steering (only when moving)
    if (g_player_state.speed > 5) {
        if (action.steering == Action::LEFT) {
            g_player_state.angle -= 0.1f;
        } else if (action.steering == Action::RIGHT) {
            g_player_state.angle += 0.1f;
        }
    }
    
    // Normalize angle to [0, 2*PI)
    while (g_player_state.angle < 0) g_player_state.angle += 2.0f * M_PI;
    while (g_player_state.angle >= 2.0f * M_PI) g_player_state.angle -= 2.0f * M_PI;
    
    // Cap maximum velocity
    float max_vel = g_player_state.max_speed;
    float vel_mag = sqrt(g_player_state.vel_x * g_player_state.vel_x + 
                        g_player_state.vel_y * g_player_state.vel_y);
    if (vel_mag > max_vel) {
        g_player_state.vel_x = (g_player_state.vel_x / vel_mag) * max_vel;
        g_player_state.vel_y = (g_player_state.vel_y / vel_mag) * max_vel;
    }
    
    // Fire action (simple implementation - just consume energy)
    if (action.fire == Action::FIRE && g_player_state.energy >= 10) {
        g_player_state.energy -= 10;
        
        std::cout << "Applying action: move=" << action.movement 
                  << " steer=" << action.steering 
                  << " fire=" << action.fire
                  << " sp1=" << action.special1
                  << " sp2=" << action.special2 << std::endl;
    }
}

void GameSimulation::render_frame(unsigned char* buffer, int width, int height) {
    // Generate mock frame data
    // In real implementation, this would capture actual game rendering
    
    if (width != screen_width_ || height != screen_height_) {
        std::cout << "Warning: frame size mismatch" << std::endl;
    }
    
    // Generate test pattern based on simulation state
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
    
    // For now create a minimal implementation that can provide basic state
    // In a full implementation this would properly initialize a VangerUnit
    player_unit_ = nullptr; // Will be properly implemented when we have full game integration
    
    std::cout << "Player unit created (minimal implementation)" << std::endl;
}

void GameSimulation::create_game_map() {
    std::cout << "Creating game map" << std::endl;
    
    // For now create a minimal implementation
    // In a full implementation this would properly load a Vangers world
    game_map_ = nullptr; // Will be properly implemented when we have full game integration
    
    std::cout << "Game map created (minimal implementation)" << std::endl;
}

void GameSimulation::setup_deterministic_mode() {
    std::cout << "Setting up deterministic mode" << std::endl;
    deterministic_mode_ = true;
    util::rng.set_seed(12345);
}

void GameSimulation::disable_real_time_features() {
    std::cout << "Disabling real-time features" << std::endl;
}

void GameSimulation::init_graphics_headless() {
    std::cout << "Initializing headless graphics" << std::endl;
}

void GameSimulation::init_physics_deterministic() {
    std::cout << "Initializing deterministic physics" << std::endl;
}

void GameSimulation::init_ai_systems() {
    std::cout << "Initializing AI systems" << std::endl;
}

void GameSimulation::setup_event_hooks() {
    std::cout << "Setting up event hooks" << std::endl;
}

// Game engine instance implementation
GameEngineInstance::GameEngineInstance(int width, int height) 
    : screen_width_(width), screen_height_(height) {
    instance_id_ = reinterpret_cast<void*>(this);
    frame_buffer_.resize(width * height * 3, 128); // Gray background
    simulation_ = std::make_unique<GameSimulation>(width, height);
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
    
    if (!simulation_->initialize()) {
        std::cerr << "Failed to initialize simulation" << std::endl;
        return false;
    }
    
    // Set as current instance
    g_current_instance = this;
    
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
        simulation_->shutdown();
    }
    
    if (g_current_instance == this) {
        g_current_instance = nullptr;
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
    
    try {
        // Apply current action
        if (simulation_) {
            simulation_->apply_action(current_action_);
            simulation_->step(num_steps);
        }
        
        simulation_step_count_ += num_steps;
        
        // Update frame buffer
        update_frame_buffer();
        
        // Generate some mock events periodically
        handle_game_events();
        
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
    
    // Use the simulated minimal player state
    state.position.x = g_player_state.pos_x;
    state.position.y = g_player_state.pos_y;
    state.position.z = g_player_state.pos_z;
    
    state.velocity.x = g_player_state.vel_x;
    state.velocity.y = g_player_state.vel_y;
    state.velocity.z = g_player_state.vel_z;
    
    state.angle = g_player_state.angle;
    state.pitch = 0.0f;  // Not simulated yet
    state.roll = 0.0f;   // Not simulated yet
    
    state.armor = g_player_state.armor;
    state.energy = g_player_state.energy;
    state.speed = g_player_state.speed;
    state.max_speed = g_player_state.max_speed;
    
    state.volume = 0;     // Not simulated yet
    state.max_volume = 100;
    state.money = 1000;   // Fixed for now
    
    state.alive = g_player_state.alive;
    state.on_ground = g_player_state.on_ground;
    state.in_water = g_player_state.in_water;
    
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
    static int event_counter = 0;
    event_counter++;
    
    // Generate collision event every 100 steps
    if (event_counter % 100 == 0) {
        event_queue_.push_event(0, 10.0f, 100.0f, 200.0f); // Collision
    }
    
    // Generate item pickup every 150 steps
    if (event_counter % 150 == 0) {
        event_queue_.push_event(1, 1.0f, 150.0f, 250.0f); // Item collected
    }
    
    // Generate damage event every 200 steps
    if (event_counter % 200 == 0) {
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
    std::lock_guard<std::mutex> lock(instances_mutex_);
    
    void* id = reinterpret_cast<void*>(next_id_.fetch_add(1));
    auto instance = std::make_unique<GameEngineInstance>(width, height);
    
    if (instance->initialize()) {
        instances_[id] = std::move(instance);
        std::cout << "Created gym instance: " << id << " (" << width << "x" << height << ")" << std::endl;
        return id;
    } else {
        std::cerr << "Failed to initialize gym instance" << std::endl;
        return nullptr;
    }
}

void InstanceManager::destroy_instance(void* handle) {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    
    auto it = instances_.find(handle);
    if (it != instances_.end()) {
        std::cout << "Destroying gym instance: " << handle << std::endl;
        instances_.erase(it);
    } else {
        std::cerr << "Attempted to destroy non-existent instance: " << handle << std::endl;
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
    // This is a mock implementation
    // In real implementation, this would extract from actual VangerUnit
    // Initialize state to zero
    state = PlayerState{};
    
    if (unit) {
        // Mock values - replace with actual unit data extraction
        state.position = Vector3(0, 0, 0);
        state.velocity = Vector3(0, 0, 0);
        state.angle = 0.0f;
        state.armor = 100;
        state.energy = 100;
        state.alive = true;
        state.on_ground = true;
    }
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

// Global state
static bool g_engine_initialized = false;

// C API implementation
extern "C" {

int vangers_engine_init() {
    if (g_engine_initialized) {
        return 1;
    }
    
    std::cout << "Initializing Vangers gym engine library v1.0.0" << std::endl;
    g_gym_mode_enabled = true;
    g_engine_initialized = true;
    return 1;
}

void vangers_engine_cleanup() {
    if (!g_engine_initialized) {
        return;
    }
    
    std::cout << "Cleaning up Vangers gym engine library" << std::endl;
    
    InstanceManager::instance().cleanup_all();
    g_gym_mode_enabled = false;
    g_engine_initialized = false;
}

void* vangers_create_instance(int width, int height) {
    if (!g_engine_initialized) {
        std::cerr << "Engine not initialized - call vangers_engine_init() first" << std::endl;
        return nullptr;
    }
    
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        std::cerr << "Invalid dimensions: " << width << "x" << height << std::endl;
        return nullptr;
    }
    
    return InstanceManager::instance().create_instance(width, height);
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