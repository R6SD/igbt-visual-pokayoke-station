#pragma once

#include <igbt/common.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace igbt {

struct BufferedFrame {
    int64_t frame_id{0};
    double capture_monotonic_s{0.0};
    cv::Mat bgr;
};

/**
 * Fixed-capacity pre-trigger raw frame ring buffer (e.g., 25 slots @ 5 FPS = 5s).
 * Reuses OpenCV Mat backing buffers to avoid runtime heap churn and skips JPEG
 * encoding until an anomaly burst actually occurs.
 */
class SkipFrameRingBuffer {
public:
    explicit SkipFrameRingBuffer(std::size_t capacity = 25, int stride = 5)
        : capacity_(std::max<std::size_t>(1, capacity)),
          stride_(std::max(1, stride)),
          slots_(capacity_) {}

    bool push_if_due(int64_t frame_id, double capture_ts, const cv::Mat& bgr) {
        if (bgr.empty() || (frame_id % stride_) != 0) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        BufferedFrame& slot = slots_[write_idx_];
        slot.frame_id = frame_id;
        slot.capture_monotonic_s = capture_ts;
        bgr.copyTo(slot.bgr);
        write_idx_ = (write_idx_ + 1) % capacity_;
        if (count_ < capacity_) {
            ++count_;
        }
        return true;
    }

    std::vector<BufferedFrame> snapshot_ordered() const {
        std::lock_guard<std::mutex> lk(mtx_);
        std::vector<BufferedFrame> out;
        out.reserve(count_);
        if (count_ == 0) return out;
        const std::size_t oldest = (count_ < capacity_) ? 0 : write_idx_;
        for (std::size_t i = 0; i < count_; ++i) {
            const BufferedFrame& s = slots_[(oldest + i) % capacity_];
            if (!s.bgr.empty()) {
                out.push_back(BufferedFrame{s.frame_id, s.capture_monotonic_s, s.bgr.clone()});
            }
        }
        return out;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        write_idx_ = 0;
        count_ = 0;
        for (auto& s : slots_) {
            s.frame_id = 0;
            s.capture_monotonic_s = 0.0;
        }
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return count_;
    }

    std::size_t capacity() const noexcept { return capacity_; }
    int stride() const noexcept { return stride_; }

private:
    const std::size_t capacity_;
    const int stride_;
    mutable std::mutex mtx_;
    std::vector<BufferedFrame> slots_;
    std::size_t write_idx_{0};
    std::size_t count_{0};
};

/**
 * Low-priority asynchronous JPEG burst writer with Drop-Incoming backpressure
 * and coalescing window to keep the 25 FPS capture loop free of disk stalls.
 */
class AsyncEvidenceWriter {
public:
    explicit AsyncEvidenceWriter(
        std::filesystem::path evidence_root = {},
        std::size_t max_queue_size = 8,
        double coalesce_window_s = 10.0
    )
        : evidence_root_(std::move(evidence_root)),
          max_queue_size_(std::max<std::size_t>(1, max_queue_size)),
          coalesce_window_s_(std::max(0.0, coalesce_window_s)) {}

    ~AsyncEvidenceWriter() {
        stop_and_drain();
    }

    AsyncEvidenceWriter(const AsyncEvidenceWriter&) = delete;
    AsyncEvidenceWriter& operator=(const AsyncEvidenceWriter&) = delete;

    void set_evidence_root(std::filesystem::path root) {
        std::lock_guard<std::mutex> lk(mtx_);
        evidence_root_ = std::move(root);
    }

    void set_coalesce_window_s(double seconds) {
        std::lock_guard<std::mutex> lk(mtx_);
        coalesce_window_s_ = std::max(0.0, seconds);
    }

    bool enqueue_burst(
        const std::string& cycle_id,
        const std::string& reason,
        std::vector<BufferedFrame> frames,
        double now_mono
    ) {
        if (frames.empty() || stop_requested_.load()) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        if (evidence_root_.empty() || stop_requested_.load()) {
            return false;
        }
        if (coalesce_window_s_ > 0.0 && last_burst_at_mono_ >= 0.0 &&
            (now_mono - last_burst_at_mono_) < coalesce_window_s_) {
            ++coalesced_bursts_;
            return false;
        }
        if (queue_.size() >= max_queue_size_) {
            ++dropped_bursts_;
            return false;
        }

        last_burst_at_mono_ = now_mono;
        BurstJob job;
        job.cycle_id = sanitize_id(cycle_id);
        job.reason = reason.empty() ? "ALARM" : reason;
        job.created_utc = utc_now_iso();
        job.frames = std::move(frames);
        queue_.push_back(std::move(job));
        ensure_started_locked();
        cv_.notify_one();
        return true;
    }

    void stop_and_drain() {
        std::thread to_join;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_requested_.store(true);
            cv_.notify_all();
            if (worker_.joinable()) {
                to_join = std::move(worker_);
            }
        }
        if (to_join.joinable()) {
            to_join.join();
        }
    }

    json stats_json() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return json{
            {"queued_bursts", queue_.size()},
            {"written_bursts", written_bursts_},
            {"written_frames", written_frames_},
            {"dropped_bursts", dropped_bursts_},
            {"coalesced_bursts", coalesced_bursts_}
        };
    }

private:
    struct BurstJob {
        std::string cycle_id;
        std::string reason;
        std::string created_utc;
        std::vector<BufferedFrame> frames;
    };

    static std::string sanitize_id(const std::string& raw) {
        std::string clean;
        clean.reserve(raw.size());
        for (unsigned char ch : raw) {
            if (std::isalnum(ch) || ch == '_' || ch == '-') {
                clean.push_back(static_cast<char>(ch));
            }
        }
        if (clean.empty()) return "LIVE_BURST";
        if (clean.size() > 64) clean.resize(64);
        return clean;
    }

    void ensure_started_locked() {
        if (worker_.joinable() || stop_requested_.load()) return;
        worker_ = std::thread([this]() { worker_loop(); });
    }

    void worker_loop() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        while (true) {
            BurstJob job;
            std::filesystem::path root_copy;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this]() { return !queue_.empty() || stop_requested_.load(); });
                if (queue_.empty() && stop_requested_.load()) {
                    break;
                }
                if (queue_.empty()) continue;
                job = std::move(queue_.front());
                queue_.pop_front();
                root_copy = evidence_root_;
            }

            if (root_copy.empty()) continue;
            const auto bursts_base = root_copy / "bursts";
            const auto burst_dir = bursts_base / path_from_utf8(job.cycle_id);
            if (!is_path_within(root_copy, burst_dir)) continue;

            std::error_code ec;
            std::filesystem::create_directories(burst_dir, ec);
            const std::string burst_id = "burst_" + random_token_hex(6);
            json frame_entries = json::array();
            uint64_t saved_now = 0;

            for (std::size_t i = 0; i < job.frames.size(); ++i) {
                const auto& f = job.frames[i];
                if (f.bgr.empty()) continue;
                std::ostringstream fn;
                fn << burst_id << "_" << std::setw(3) << std::setfill('0') << i << ".jpg";
                const std::string filename = fn.str();
                if (imwrite_unicode(burst_dir / path_from_utf8(filename), f.bgr, 82)) {
                    ++saved_now;
                    frame_entries.push_back(json{
                        {"file", filename},
                        {"frame_id", f.frame_id},
                        {"capture_monotonic_s", f.capture_monotonic_s}
                    });
                }
            }

            const json manifest{
                {"burst_id", burst_id},
                {"cycle_id", job.cycle_id},
                {"reason", job.reason},
                {"created_at_utc", job.created_utc},
                {"frame_count", saved_now},
                {"frames", frame_entries}
            };
            const std::string dumped = manifest.dump(2);
            write_file_bytes_atomic(
                burst_dir / path_from_utf8(burst_id + "_manifest.json"),
                dumped.data(),
                dumped.size()
            );

            {
                std::lock_guard<std::mutex> lk(mtx_);
                ++written_bursts_;
                written_frames_ += saved_now;
            }
        }
    }

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::filesystem::path evidence_root_;
    const std::size_t max_queue_size_;
    double coalesce_window_s_;
    double last_burst_at_mono_{-1.0};
    std::deque<BurstJob> queue_;
    std::thread worker_;
    std::atomic<bool> stop_requested_{false};
    uint64_t written_bursts_{0};
    uint64_t written_frames_{0};
    uint64_t dropped_bursts_{0};
    uint64_t coalesced_bursts_{0};
};

}  // namespace igbt
