from benchmark_models import benchmark

def generate_report():
    results = benchmark(num_frames=50)

    print("\n========== MODEL PERFORMANCE REPORT ==========")
    print(f"✅ Avg Emotion Inference Time : {results['avg_emotion_time_ms']:.2f} ms")
    print(f"⚡ Min Emotion Inference Time : {results['min_emotion_time_ms']:.2f} ms")
    print(f"🐢 Max Emotion Inference Time : {results['max_emotion_time_ms']:.2f} ms")
    print(f"😃 Avg Emotion Score          : {results['avg_emotion_score']:.2f}")
    print(f"🎯 FPS                       : {results['fps']:.2f}")
    print(f"🖥️  Avg CPU Usage             : {results['cpu_usage_avg']:.2f}%")
    print(f"💾 Avg Memory Usage           : {results['memory_usage_avg']:.2f}%")
    print("==============================================")

if __name__ == "__main__":
    generate_report()