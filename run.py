#!/usr/bin/env python3
"""
Wrapper to fix multiprocessing spawn issues
Sets mp.set_start_method BEFORE any torch imports
"""
import sys
import os

# Configure Qt BEFORE any imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # CRITICAL: Use torch.multiprocessing and set spawn BEFORE any torch/cuda imports
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Now import and run main (this will load torch/cuda modules)
    from main import main, main_debug, main_hybrid_mac
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "debug":
            main_debug()
        elif mode == "test":
            # Quick test mode: 50 frames only
            import os
            os.environ["DEBUG_MAX_FRAMES"] = "50"
            main_debug()
        elif mode == "benchmark":
            # Benchmark mode: 200 frames for stable metrics
            import os
            os.environ["DEBUG_MAX_FRAMES"] = "200"
            main_debug()
        elif mode == "hybrid":
            main_hybrid_mac()
        else:
            print(f"❌ Modo '{mode}' no reconocido")
            print("💡 Modos disponibles: debug, test (50 frames), benchmark (200 frames), hybrid")
            sys.exit(1)
    else:
        main()
