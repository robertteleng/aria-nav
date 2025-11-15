#!/usr/bin/env python3
"""
Professional profiling tool for aria-nav
Usa cProfile + snakeviz para visualizar cuellos de botella
"""

import cProfile
import pstats
import sys
import os
from pathlib import Path

def profile_main(duration_seconds: int = 60):
    """Profile main.py con cProfile"""
    
    # Setup environment
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    
    # Import después de env vars
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Import y run main
        from main import main
        import signal
        
        # Setup timeout
        def timeout_handler(signum, frame):
            raise KeyboardInterrupt("Profiling timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration_seconds)
        
        print(f"🔍 Profiling for {duration_seconds}s...")
        main()
        
    except KeyboardInterrupt:
        print("\n✅ Profiling complete")
    finally:
        profiler.disable()
        
        # Save results
        output_dir = Path(__file__).parent.parent / "logs" / "profiling"
        output_dir.mkdir(exist_ok=True)
        
        profile_file = output_dir / "latest.prof"
        stats_file = output_dir / "latest_stats.txt"
        
        # Dump raw profile
        profiler.dump_stats(str(profile_file))
        print(f"📊 Profile saved: {profile_file}")
        
        # Generate text report
        with open(stats_file, 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs()
            ps.sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("TOP 20 BY TIME\n")
            f.write("="*80 + "\n")
            ps.sort_stats('time')
            ps.print_stats(20)
        
        print(f"📄 Stats saved: {stats_file}")
        print(f"\n💡 Visualize with: snakeviz {profile_file}")
        print(f"💡 Or install: pip install snakeviz && snakeviz {profile_file}")

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    profile_main(duration)
