// Unified Test Runner for LuminaChat Rework Phases 1-4
// This file allows testing individual phases or all phases together
// 
// Usage:
//   test_phases.exe        - Run all phases
//   test_phases.exe 1      - Run only phase 1
//   test_phases.exe 2      - Run only phase 2
//   test_phases.exe 3      - Run only phase 3
//   test_phases.exe 4      - Run only phase 4

#include <iostream>
#include <string>
#include <vector>

// Forward declarations for test functions
int TestPhase1Foundation();
int TestPhase2Model();
int TestPhase3Template();
int TestPhase4Context();

int main(int argc, char* argv[]) {
    std::cout << "=== LuminaChat Rework Test Suite ===" << std::endl;
    std::cout << "Testing Phases 1-4 Implementation" << std::endl;
    std::cout << std::endl;

    std::vector<int> phases_to_run;
    
    if (argc == 1) {
        // No arguments - run all phases
        phases_to_run = {1, 2, 3, 4};
        std::cout << "Running all phases (1-4)" << std::endl;
    } else {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            try {
                int phase = std::stoi(argv[i]);
                if (phase >= 1 && phase <= 4) {
                    phases_to_run.push_back(phase);
                } else {
                    std::cerr << "Invalid phase number: " << phase << " (must be 1-4)" << std::endl;
                    return 1;
                }
            } catch (const std::exception&) {
                std::cerr << "Invalid argument: " << argv[i] << " (must be a number 1-4)" << std::endl;
                return 1;
            }
        }
        
        std::cout << "Running phases: ";
        for (size_t i = 0; i < phases_to_run.size(); ++i) {
            std::cout << phases_to_run[i];
            if (i < phases_to_run.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    
    int total_failures = 0;
    
    for (int phase : phases_to_run) {
        std::cout << "===============================================" << std::endl;
        
        int result = 0;
        switch (phase) {
            case 1:
                std::cout << "PHASE 1: Foundation Components" << std::endl;
                std::cout << "===============================================" << std::endl;
                result = TestPhase1Foundation();
                break;
                
            case 2:
                std::cout << "PHASE 2: Model Layer" << std::endl;
                std::cout << "===============================================" << std::endl;
                result = TestPhase2Model();
                break;
                
            case 3:
                std::cout << "PHASE 3: Template Layer" << std::endl;
                std::cout << "===============================================" << std::endl;
                result = TestPhase3Template();
                break;
                
            case 4:
                std::cout << "PHASE 4: Context Layer" << std::endl;
                std::cout << "===============================================" << std::endl;
                result = TestPhase4Context();
                break;
        }
        
        if (result == 0) {
            std::cout << "✓ PHASE " << phase << " PASSED" << std::endl;
        } else {
            std::cout << "✗ PHASE " << phase << " FAILED" << std::endl;
            total_failures++;
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "===============================================" << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    if (total_failures == 0) {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        std::cout << "Phases 1-4 are ready for integration!" << std::endl;
    } else {
        std::cout << "❌ " << total_failures << " phase(s) failed" << std::endl;
        std::cout << "Please fix the failing tests before proceeding" << std::endl;
    }
    
    return total_failures;
}

// Include the actual test implementations
#include "test_phase1_impl.hpp"
#include "test_phase2_impl.hpp"
#include "test_phase3_impl.hpp"
#include "test_phase4_impl.hpp"
