from benchmark_models import benchmark

def generate_report(num_frames=30):
    results = benchmark(num_frames=num_frames)

    print("\n========== MODEL PERFORMANCE REPORT ==========")
    print(f"😃 Emotion:")
    print(f"   Avg Inference Time: {results['avg_emotion_time_ms']:.2f} ms")
    print(f"   Min Inference Time: {results['min_emotion_time_ms']:.2f} ms")
    print(f"   Max Inference Time: {results['max_emotion_time_ms']:.2f} ms")
    print(f"   Avg Confidence    : {results['avg_emotion_conf']:.2f}")
    print(f"   FPS               : {results['fps_emotion']:.2f}")

    print(f"\n🎂 Age:")
    print(f"   Avg Inference Time: {results['avg_age_time_ms']:.2f} ms")
    print(f"   Min Inference Time: {results['min_age_time_ms']:.2f} ms")
    print(f"   Max Inference Time: {results['max_age_time_ms']:.2f} ms")
    print(f"   Avg Confidence    : {results['avg_age_conf']:.2f}")
    print(f"   FPS               : {results['fps_age']:.2f}")

    print(f"\n🚻 Gender:")
    print(f"   Avg Inference Time: {results['avg_gender_time_ms']:.2f} ms")
    print(f"   Min Inference Time: {results['min_gender_time_ms']:.2f} ms")
    print(f"   Max Inference Time: {results['max_gender_time_ms']:.2f} ms")
    print(f"   Avg Confidence    : {results['avg_gender_conf']:.2f}")
    print(f"   FPS               : {results['fps_gender']:.2f}")

    print(f"\n🖥️ System Usage:")
    print(f"   Avg CPU Usage    : {results['cpu_usage_avg']:.2f}%")
    print(f"   Avg Memory Usage : {results['memory_usage_avg']:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    generate_report()