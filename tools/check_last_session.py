"""Check last session logs to see if detections are happening"""

import json
from pathlib import Path
from datetime import datetime

# Find most recent session
logs_dir = Path("logs")
sessions = sorted([d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("session_")])

if not sessions:
    print("❌ No sessions found in logs/")
    exit(1)

last_session = sessions[-1]
print(f"📁 Checking session: {last_session.name}\n")

# Check detections.log
detections_log = last_session / "detections.log"
if not detections_log.exists():
    print(f"❌ No detections.log found")
    exit(1)

print("=" * 60)
print("DETECTIONS ANALYSIS")
print("=" * 60)

detections = []
with open(detections_log) as f:
    for line in f:
        if not line.strip():
            continue
        # Format: timestamp - level - json
        parts = line.split(" - ", 2)
        if len(parts) >= 3:
            try:
                data = json.loads(parts[2])
                detections.append(data)
            except:
                pass

if not detections:
    print("❌ No detections found in log")
    print("\n⚠️  PROBLEM: YOLO is not detecting anything")
    print("   Check if models are loaded correctly")
    exit(1)

print(f"✅ Found {len(detections)} detections\n")

# Analyze by source
rgb_count = sum(1 for d in detections if d.get('source') == 'rgb')
slam_count = sum(1 for d in detections if d.get('source') in ['slam', 'slam1', 'slam2'])

print(f"📊 Detections by source:")
print(f"   RGB:  {rgb_count}")
print(f"   SLAM: {slam_count}")

# Show sample detections
print(f"\n📋 Sample detections (last 5):")
for det in detections[-5:]:
    obj = det.get('object_class', 'unknown')
    conf = det.get('confidence', 0)
    source = det.get('source', 'unknown')
    dist = det.get('distance_meters', 'N/A')
    print(f"   - {obj} ({conf:.2f}) from {source} at {dist}m")

# Check audio_events.log
print("\n" + "=" * 60)
print("AUDIO EVENTS ANALYSIS")
print("=" * 60)

audio_log = last_session / "audio_events.log"
if not audio_log.exists():
    print(f"❌ No audio_events.log found")
else:
    audio_events = []
    with open(audio_log) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split(" - ", 2)
            if len(parts) >= 3:
                try:
                    data = json.loads(parts[2])
                    audio_events.append(data)
                except:
                    pass
    
    if not audio_events:
        print("❌ No audio events logged")
        print("\n⚠️  PROBLEM: Audio router is not processing detections")
        print("   Check NavigationAudioRouter in logs/")
    else:
        print(f"✅ Found {audio_events} audio events\n")
        
        # Count by action
        announced = sum(1 for e in audio_events if e.get('action') == 'announced')
        skipped = sum(1 for e in audio_events if e.get('action') == 'skipped')
        dropped = sum(1 for e in audio_events if e.get('action') == 'dropped')
        
        print(f"📊 Audio events:")
        print(f"   Announced: {announced}")
        print(f"   Skipped:   {skipped}")
        print(f"   Dropped:   {dropped}")
        
        # Show sample
        print(f"\n📋 Sample audio events (last 5):")
        for evt in audio_events[-5:]:
            action = evt.get('action', 'unknown')
            msg = evt.get('message', 'N/A')
            reason = evt.get('reason', '')
            if reason:
                print(f"   - {action}: {msg} (reason: {reason})")
            else:
                print(f"   - {action}: {msg}")

# Check audio_routing.log for more details
print("\n" + "=" * 60)
print("AUDIO ROUTING DEBUG")
print("=" * 60)

routing_log = last_session / "audio_routing.log"
if not routing_log.exists():
    print(f"❌ No audio_routing.log found")
    print("\n⚠️  NavigationAudioRouter logs are missing")
else:
    with open(routing_log) as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ audio_routing.log is empty")
        print("\n⚠️  NavigationAudioRouter is not logging anything")
    else:
        print(f"✅ Found {len(lines)} routing log entries\n")
        print("Last 10 lines:")
        for line in lines[-10:]:
            print(f"   {line.rstrip()}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if detections and rgb_count > 0:
    print("✅ RGB detections are happening")
else:
    print("❌ No RGB detections")

if audio_events:
    print("✅ Audio events are being logged")
else:
    print("❌ No audio events - check NavigationAudioRouter")
