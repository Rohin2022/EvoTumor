import os
import time

WATCH_DIR = "/scratch/rpinise1/MultiTumorSynthesis/AutoencoderCache"

def get_files_state(path):
    return {f: os.path.getmtime(os.path.join(path, f))
            for f in os.listdir(path)}

def main():
    print(f"Watching: {WATCH_DIR}")

    prev_times = {}
    last_event_time = None

    while True:
        try:
            current = get_files_state(WATCH_DIR)

            # detect new files
            new_files = set(current.keys()) - set(prev_times.keys())

            for f in sorted(new_files, key=lambda x: current[x]):
                now = current[f]

                if last_event_time is not None:
                    delta = now - last_event_time
                    print(f"New file: {f} | Δ {delta:.4f} sec")
                else:
                    print(f"First file: {f}")

                last_event_time = now

            prev_times = current

            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\nStopped.")
            break

if __name__ == "__main__":
    main()