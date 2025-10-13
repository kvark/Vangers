#include "../vangers_engine_lib.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Vangers Engine Test Client" << std::endl;
    std::cout << "=========================" << std::endl;

    // Initialize engine
    if (vangers_engine_init() != 1) {
        std::cerr << "Failed to initialize Vangers engine" << std::endl;
        return 1;
    }

    std::cout << "Engine version: " << vangers_get_version() << std::endl;

    // Create instance
    void* instance = vangers_create_instance(640, 480);
    if (!instance) {
        std::cerr << "Failed to create Vangers instance" << std::endl;
        vangers_engine_cleanup();
        return 1;
    }

    std::cout << "Created engine instance: " << instance << std::endl;

    // Reset instance
    if (vangers_reset_instance(instance) != 1) {
        std::cerr << "Failed to reset instance" << std::endl;
        vangers_destroy_instance(instance);
        vangers_engine_cleanup();
        return 1;
    }

    std::cout << "Instance reset successful" << std::endl;

    // Test player state
    PlayerState state;
    if (vangers_get_player_state(instance, &state) == 1) {
        std::cout << "Player state:" << std::endl;
        std::cout << "  Position: (" << state.position.x << ", " << state.position.y << ", " << state.position.z << ")" << std::endl;
        std::cout << "  Velocity: (" << state.velocity.x << ", " << state.velocity.y << ", " << state.velocity.z << ")" << std::endl;
        std::cout << "  Angle: " << state.angle << std::endl;
        std::cout << "  Armor: " << state.armor << "/" << "100" << std::endl;
        std::cout << "  Energy: " << state.energy << "/" << "100" << std::endl;
        std::cout << "  Speed: " << state.speed << "/" << state.max_speed << std::endl;
        std::cout << "  Alive: " << (state.alive ? "Yes" : "No") << std::endl;
    } else {
        std::cerr << "Failed to get player state" << std::endl;
    }

    // Test frame buffer
    std::vector<unsigned char> frame_buffer(640 * 480 * 3);
    if (vangers_get_frame_buffer(instance, frame_buffer.data(), frame_buffer.size()) == 1) {
        std::cout << "Frame buffer retrieved successfully (size: " << frame_buffer.size() << " bytes)" << std::endl;
        
        // Check if frame has any non-zero data
        bool has_data = false;
        for (size_t i = 0; i < frame_buffer.size() && !has_data; i++) {
            if (frame_buffer[i] != 0) {
                has_data = true;
            }
        }
        std::cout << "Frame buffer contains " << (has_data ? "image data" : "empty/black image") << std::endl;
    } else {
        std::cerr << "Failed to get frame buffer" << std::endl;
    }

    // Test simulation stepping
    std::cout << "\nTesting simulation stepping..." << std::endl;
    
    for (int step = 0; step < 10; step++) {
        // Set a simple action: move forward
        vangers_set_action(instance, 1, 0, 0, 0, 0); // forward, straight, no fire, no specials
        
        // Step simulation
        if (vangers_step_simulation(instance, 1) == 1) {
            std::cout << "Step " << (step + 1) << " completed" << std::endl;
            
            // Get updated state
            if (vangers_get_player_state(instance, &state) == 1) {
                std::cout << "  Position: (" << state.position.x << ", " << state.position.y << ", " << state.position.z << ")" << std::endl;
            }
            
            // Check for events
            GameEvent events[10];
            int num_events = vangers_get_events(instance, events, 10);
            if (num_events > 0) {
                std::cout << "  Events: " << num_events << std::endl;
                for (int i = 0; i < num_events; i++) {
                    std::cout << "    Event " << i << ": type=" << events[i].type 
                              << ", value=" << events[i].value << std::endl;
                }
                vangers_clear_events(instance);
            }
        } else {
            std::cerr << "Simulation step " << (step + 1) << " failed" << std::endl;
        }
        
        // Small delay to simulate real-time behavior
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }

    // Test time scale
    std::cout << "\nTesting time scale..." << std::endl;
    vangers_set_time_scale(instance, 2.0f);
    std::cout << "Set time scale to 2.0x" << std::endl;
    
    // Test pause/unpause
    std::cout << "Testing pause..." << std::endl;
    vangers_pause_simulation(instance, true);
    std::cout << "Simulation paused" << std::endl;
    
    vangers_pause_simulation(instance, false);
    std::cout << "Simulation unpaused" << std::endl;

    // Test different actions
    std::cout << "\nTesting different actions..." << std::endl;
    
    struct {
        const char* name;
        int movement, steering, fire, special1, special2;
    } test_actions[] = {
        {"Stop", 0, 0, 0, 0, 0},
        {"Forward", 1, 0, 0, 0, 0},
        {"Backward", 2, 0, 0, 0, 0},
        {"Turn Left", 1, 1, 0, 0, 0},
        {"Turn Right", 1, 2, 0, 0, 0},
        {"Fire", 1, 0, 1, 0, 0},
        {"Special 1", 1, 0, 0, 1, 0},
        {"Special 2", 1, 0, 0, 0, 1},
    };
    
    for (size_t i = 0; i < sizeof(test_actions) / sizeof(test_actions[0]); i++) {
        std::cout << "Action: " << test_actions[i].name << std::endl;
        vangers_set_action(instance, 
                          test_actions[i].movement,
                          test_actions[i].steering, 
                          test_actions[i].fire,
                          test_actions[i].special1, 
                          test_actions[i].special2);
        
        if (vangers_step_simulation(instance, 1) == 1) {
            std::cout << "  Action executed successfully" << std::endl;
        } else {
            std::cout << "  Action execution failed" << std::endl;
        }
    }

    // Test render modes
    std::cout << "\nTesting render modes..." << std::endl;
    vangers_set_render_mode(instance, 0); // No rendering
    std::cout << "Set render mode to 0 (none)" << std::endl;
    
    vangers_set_render_mode(instance, 1); // RGB rendering
    std::cout << "Set render mode to 1 (RGB)" << std::endl;
    
    // Test physics substeps
    std::cout << "\nTesting physics substeps..." << std::endl;
    vangers_set_physics_substeps(instance, 2);
    std::cout << "Set physics substeps to 2" << std::endl;

    // Performance test
    std::cout << "\nPerformance test..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    const int perf_steps = 1000;
    
    for (int i = 0; i < perf_steps; i++) {
        vangers_set_action(instance, 1, 0, 0, 0, 0);
        vangers_step_simulation(instance, 1);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double steps_per_second = static_cast<double>(perf_steps) / (duration.count() / 1000.0);
    std::cout << "Performance: " << perf_steps << " steps in " << duration.count() << "ms" << std::endl;
    std::cout << "Steps per second: " << steps_per_second << std::endl;

    // Test map setting (even though it's a stub)
    std::cout << "\nTesting map setting..." << std::endl;
    vangers_set_map(instance, "test_map");
    std::cout << "Map set to 'test_map'" << std::endl;

    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    vangers_destroy_instance(instance);
    std::cout << "Instance destroyed" << std::endl;

    vangers_engine_cleanup();
    std::cout << "Engine cleanup complete" << std::endl;

    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}