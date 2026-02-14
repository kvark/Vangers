#ifndef VANGERS_ENGINE_LIB_H
#define VANGERS_ENGINE_LIB_H

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <queue>
#include <thread>
#include <functional>
#include <unordered_map>
#include <stdint.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations of game classes
struct VangerUnit;
struct iGameMap;
struct ActionUnit;
struct GeneralObject;
struct StuffObject;

// C-compatible structures for Python interface
extern "C" {
    
    // Vector3 structure for positions and velocities
    struct Vector3 {
        float x, y, z;
        
        Vector3() : x(0), y(0), z(0) {}
        Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    };
    
    // Player state structure - directly accessible from Python
    struct PlayerState {
        Vector3 position;     // World position
        Vector3 velocity;     // Current velocity
        float angle;          // Heading angle (radians)
        float pitch;          // Pitch angle (radians)
        float roll;           // Roll angle (radians)
        int armor;            // Current armor
        int energy;           // Current energy
        int speed;            // Current speed
        int max_speed;        // Maximum speed
        int volume;           // Current cargo volume
        int max_volume;       // Maximum cargo volume
        int money;            // Current money
        bool alive;           // Is player alive
        bool on_ground;       // Is on ground surface
        bool in_water;        // Is in water
    };
    
    // Game event structure
    struct GameEvent {
        int type;             // Event type: 0=collision, 1=item, 2=objective, 3=damage
        float value;          // Event-specific value (damage amount, reward, etc.)
        float x, y;           // Event world position
        uint64_t timestamp;   // Game timestamp
    };

    struct VangersCreateOptions {
        uint32_t size;        // sizeof(VangersCreateOptions)
        int headless;         // 0 = normal, 1 = SDL dummy driver
    };
    
    // C API functions
    void* vangers_create_instance(int width, int height, const VangersCreateOptions* options);
    void vangers_destroy_instance(void* instance);
    int vangers_reset_instance(void* instance);
    
    // Deterministic simulation control
    int vangers_step_simulation(void* instance, int num_steps);
    void vangers_set_time_scale(void* instance, float scale);
    void vangers_pause_simulation(void* instance, bool paused);
    
    // Action interface
    void vangers_set_action(void* instance, int movement, int steering, int fire, int special1, int special2);
    
    // State query
    int vangers_get_player_state(void* instance, PlayerState* state);
    int vangers_get_frame_buffer(void* instance, unsigned char* buffer, int buffer_size);
    
    // Event system
    int vangers_get_events(void* instance, GameEvent* events, int max_events);
    void vangers_clear_events(void* instance);
    
    // Environment configuration
    void vangers_set_map(void* instance, const char* map_name);
    void vangers_set_render_mode(void* instance, int mode); // 0=none, 1=rgb, 2=depth
    void vangers_set_physics_substeps(void* instance, int substeps);
}

// C++ implementation classes
//
// The engine initialization functions are declared in the global namespace so
// linkage matches the engine core implementation. They are intentionally placed
// here (before the vangers_engine namespace) to avoid conflicting declarations
// and to ensure the wrapper calls the same global symbols the engine provides.
extern void graph3d_init();
extern void GeneralSystemInit();
extern void uniVangPrepare();
extern void GeneralSystemOpen();

namespace vangers_engine {

// (engine init declarations moved to global scope to match engine core linkage)

// Internal game state (moved from module statics into instance state)
// Moved to namespace scope so the GameSimulation class can reference it
// without being forced to carry the full nested definition in the header.
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

// Action representation
struct Action {
    enum Movement { STOP = 0, FORWARD = 1, BACKWARD = 2 };
    enum Steering { STRAIGHT = 0, LEFT = 1, RIGHT = 2 };
    enum Fire { NO_FIRE = 0, FIRE = 1 };
    enum Special { NO_SPECIAL = 0, USE_SPECIAL = 1 };
    
    Movement movement;
    Steering steering;
    Fire fire;
    Special special1;
    Special special2;
    
    Action() : movement(STOP), steering(STRAIGHT), fire(NO_FIRE), 
               special1(NO_SPECIAL), special2(NO_SPECIAL) {}
};

// Event queue for tracking game events
class EventQueue {
public:
    void push_event(int type, float value, float x = 0, float y = 0);
    int get_events(GameEvent* events, int max_events);
    void clear();
    
private:
    std::queue<GameEvent> events_;
    std::mutex mutex_;
    uint64_t timestamp_counter_ = 0;
};

// Deterministic game engine instance
class GameEngineInstance {
public:
    GameEngineInstance(int width, int height);
    ~GameEngineInstance();
    
    // Instance management
    bool initialize();
    void shutdown();
    int reset();

    void set_headless_mode(bool headless);
    bool get_headless_mode() const { return headless_mode_; }
    
    // Simulation control (deterministic)
    int step_simulation(int num_steps = 1);
    void set_time_scale(float scale) { 
        std::lock_guard<std::mutex> lock(state_mutex_);
        time_scale_ = scale; 
    }
    void pause_simulation(bool paused) { 
        std::lock_guard<std::mutex> lock(state_mutex_);
        paused_ = paused; 
    }
    
    // Action interface
    void set_action(const Action& action);
    Action get_current_action() const;
    
    // State access
    bool get_player_state(PlayerState& state) const;
    bool get_frame_buffer(unsigned char* buffer, int buffer_size) const;
    
    // Event system
    int get_events(GameEvent* events, int max_events);
    void clear_events();
    
    // Environment configuration
    void set_map(const std::string& map_name);
    void set_render_mode(int mode) { 
        std::lock_guard<std::mutex> lock(state_mutex_);
        render_mode_ = mode; 
    }
    void set_physics_substeps(int substeps) { 
        std::lock_guard<std::mutex> lock(state_mutex_);
        physics_substeps_ = substeps; 
    }
    
    // Game object access (thread-safe)
    VangerUnit* get_player_unit() const;
    iGameMap* get_game_map() const;
    
    // Instance identification
    void* get_instance_id() const { return instance_id_; }
    
private:
    // Instance identification
    void* instance_id_;
    
    // Configuration (protected by state_mutex_)
    int screen_width_, screen_height_;
    int render_mode_ = 0;  // 0=none, 1=rgb, 2=depth
    int physics_substeps_ = 1;
    float time_scale_ = 1.0f;
    bool paused_ = false;
    
    // Game state (each instance has its own)
    std::unique_ptr<class GameSimulation> simulation_;
    VangerUnit* player_unit_ = nullptr;
    iGameMap* game_map_ = nullptr;
    
    // Action state (thread-safe)
    Action current_action_;
    mutable std::mutex action_mutex_;
    
    // Event tracking (instance-specific)
    EventQueue event_queue_;
    mutable std::mutex event_mutex_;
    
    // Frame buffer (instance-specific)
    std::vector<unsigned char> frame_buffer_;
    mutable std::mutex frame_mutex_;
    
    // Simulation state (protected by state_mutex_)
    uint64_t simulation_step_count_ = 0;
    uint64_t last_update_time_ = 0;
    bool initialized_ = false;
    bool headless_mode_ = false;
    
    // Master mutex for state consistency
    mutable std::mutex state_mutex_;
    
    // Internal methods
    void update_frame_buffer();
    void update_player_state();
    void process_action();
    void advance_physics(int steps);
    void handle_game_events();
    // Event counter moved from function-local static to instance member to avoid module-level statics
    int event_counter_ = 0;
    
    // Event callbacks (called by game engine)
    void on_collision(VangerUnit* unit, GeneralObject* obj, float damage);
    void on_item_collected(VangerUnit* unit, StuffObject* item);
    void on_objective_completed();
    void on_player_damage(VangerUnit* unit, float damage);
    
    // Disable copy/move to prevent accidental sharing
    GameEngineInstance(const GameEngineInstance&) = delete;
    GameEngineInstance& operator=(const GameEngineInstance&) = delete;
    GameEngineInstance(GameEngineInstance&&) = delete;
    GameEngineInstance& operator=(GameEngineInstance&&) = delete;
};

// Game simulation wrapper that isolates game engine
class GameSimulation {
public:
    GameSimulation(int width, int height);
    ~GameSimulation();

    // Accessors for internal player state (exposed publicly for wrapper)
    const MinimalPlayerState& get_internal_player_state() const { return player_state_; }
    MinimalPlayerState& access_internal_player_state() { return player_state_; }

    bool initialize();
    void shutdown();
    void reset();
    
    void step(int num_steps);
    void set_time_scale(float scale);
    
    VangerUnit* get_player_unit() const { return player_unit_; }
    iGameMap* get_game_map() const { return game_map_; }
    
    void apply_action(const Action& action);
    void render_frame(unsigned char* buffer, int width, int height);

    void set_headless_mode(bool headless) { headless_mode_ = headless; }
    bool is_headless_mode() const { return headless_mode_; }
    
    // Event callbacks (disabled for minimal implementation)
    void set_collision_callback(std::function<void(void*, void*, float)>) {
        // Callback system disabled in minimal implementation
    }
    void set_item_callback(std::function<void(void*, void*)>) {
        // Callback system disabled in minimal implementation
    }
    void set_objective_callback(std::function<void()>) {
        // Callback system disabled in minimal implementation
    }
    void set_damage_callback(std::function<void(void*, float)>) {
        // Callback system disabled in minimal implementation
    }
    
private:
    // Game engine components
    VangerUnit* player_unit_ = nullptr;
    iGameMap* game_map_ = nullptr;
    
    // Simulation parameters
    int screen_width_, screen_height_;
    float time_scale_ = 1.0f;
    bool deterministic_mode_ = true;
    bool headless_mode_ = false;
    
    // Internal game state (moved from module statics into instance state)
    // The struct definition was moved to namespace scope above; keep the
    // per-instance storage here as a simple member.
    MinimalPlayerState player_state_;              // replaces previous `g_player_state`
    uint64_t frame_count_ = 0;
    uint64_t last_physics_time_ = 0;
    
    // Legacy RNG seeds used by wrapper-level deterministic features
    unsigned rng_seed_ = 12345;
    unsigned legacy_rndval_ = 83;
    unsigned legacy_real_rndval_ = 12345;
    
    // Engine / wrapper initialization flags moved into instance state
    bool game_initialized_ = false;                // replaces previous `g_GameInitialized`
    bool gym_mode_enabled_ = false;                // replaces previous `g_gym_mode_enabled`
    
    // Reference to owning engine instance manager (optional convenience)
    // Note: this is not an owning pointer; InstanceManager remains responsible for instances.
    void* owning_instance_id_ = nullptr;
    
    // Callbacks (disabled for minimal implementation)
    // std::function<void(VangerUnit*, GeneralObject*, float)> collision_callback_;
    // std::function<void(VangerUnit*, StuffObject*)> item_callback_;
    // std::function<void()> objective_callback_;
    // std::function<void(VangerUnit*, float)> damage_callback_;
    
    // Private methods
    void create_player_unit();
    void create_game_map();
    void setup_deterministic_mode();
    void disable_real_time_features();
    
    // Accessors for the moved state were relocated to the public section above.
    void set_game_initialized(bool v) { game_initialized_ = v; }
    bool is_game_initialized() const { return game_initialized_; }
    void set_gym_mode_enabled(bool v) { gym_mode_enabled_ = v; }
    bool is_gym_mode_enabled() const { return gym_mode_enabled_; }
    
    // Game engine integration
    void init_graphics_headless();
    void init_physics_deterministic();
    void init_ai_systems();
    void setup_event_hooks();
};

// Global instance manager for multiple environments
class InstanceManager {
public:
    static InstanceManager& instance();
    
    void* create_instance(int width, int height);
    void* create_instance_with_options(int width, int height, bool headless);
    void destroy_instance(void* handle);
    GameEngineInstance* get_instance(void* handle);
    
    void cleanup_all();
    size_t get_num_instances() const;
    
    // Engine-wide flags / state that were previously module-level statics.
    // These are now managed by the instance manager which has process-lifetime scope.
    void set_gym_mode(bool enabled) {
        std::lock_guard<std::mutex> lock(instances_mutex_);
        gym_mode_enabled_ = enabled;
    }
    bool get_gym_mode() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(instances_mutex_));
        return gym_mode_enabled_;
    }
    
    // Track the current active instance (if any). This replaces the previous global pointer.
    void set_current_instance(GameEngineInstance* instance) {
        std::lock_guard<std::mutex> lock(instances_mutex_);
        current_instance_ = instance;
    }
    GameEngineInstance* get_current_instance() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(instances_mutex_));
        return current_instance_;
    }
    
private:
    InstanceManager() = default;
    ~InstanceManager();
    
    std::unordered_map<void*, std::unique_ptr<GameEngineInstance>> instances_;
    mutable std::mutex instances_mutex_;
    std::atomic<uintptr_t> next_id_{1};
    
    // Moved globals (now per-process state managed via InstanceManager)
    bool gym_mode_enabled_ = false;               // replaces previous `g_gym_mode_enabled`
    GameEngineInstance* current_instance_ = nullptr; // replaces previous `g_current_instance`
    // Engine initialization flag for the wrapper/core, moved here from module-level static
    bool engine_initialized_ = false;
};

// Utility functions
namespace util {
    // Convert game coordinates to normalized values
    Vector3 normalize_position(const Vector3& pos);
    Vector3 normalize_velocity(const Vector3& vel);
    float normalize_angle(float angle);
    
    // Convert between game and library action formats
    Action convert_action(int movement, int steering, int fire, int special1, int special2);
    void extract_player_state(const VangerUnit* unit, PlayerState& state);
    
    // Frame buffer utilities
    void convert_frame_format(const unsigned char* src, unsigned char* dst, 
                             int width, int height, int src_format, int dst_format);
    
    // Deterministic random number generation
    class DeterministicRNG {
    public:
        DeterministicRNG(uint32_t seed = 12345) : state_(seed) {}
        
        uint32_t next();
        float next_float();  // [0, 1)
        int next_int(int min_val, int max_val);
        
        void set_seed(uint32_t seed) { state_ = seed; }
        uint32_t get_seed() const { return state_; }
        
    private:
        uint32_t state_;
    };
    
    extern thread_local DeterministicRNG rng;
}

} // namespace vangers_engine

// Global initialization (called once when library loads)
extern "C" {
    int vangers_engine_init();
    int vangers_engine_init_with_path(const char* resource_path);
    void vangers_engine_cleanup();
    
    // Debugging and diagnostics
    void vangers_set_debug_mode(bool enabled);
    const char* vangers_get_version();
    int vangers_get_num_instances();
}

#endif // VANGERS_ENGINE_LIB_H
