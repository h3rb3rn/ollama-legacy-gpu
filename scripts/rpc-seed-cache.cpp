// rpc-seed-cache — fills the tensor cache of an llama.cpp `rpc-server` from a
// GGUF file on the local disk, so that the head of a cross-host pool does not
// have to send the model weights over the network.
//
// Background (ggml-rpc.cpp): for every weight tensor larger than 10 MiB the
// client first sends only the FNV-1a hash of the tensor data
// (RPC_CMD_SET_TENSOR_HASH). An rpc-server started with `-c` answers from
// <cache>/<16 hex digits of the hash> and the data stays off the wire. That
// cache is normally filled by the first transfer; this tool writes the same
// files from the local copy of the model instead.
//
// Usage:
//   rpc-seed-cache [--cache-dir DIR] [--threads N] [--dry-run] MODEL.gguf [MODEL2.gguf ...]
//
// Default cache directory: the one `rpc-server -c` uses
// ($LLAMA_CACHE/rpc, else $XDG_CACHE_HOME/llama.cpp/rpc, else ~/.cache/llama.cpp/rpc).
//
// Built in the image against libggml-base (GGUF reader of the same llama.cpp
// revision as rpc-server).

#include "ggml.h"
#include "gguf.h"

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

// Must match HASH_THRESHOLD and fnv_hash() in ggml/src/ggml-rpc/ggml-rpc.cpp.
static const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static fs::path default_cache_dir() {
    const char * env = std::getenv("LLAMA_CACHE");
    if (env && *env) {
        return fs::path(env) / "rpc";
    }
    fs::path base;
    env = std::getenv("XDG_CACHE_HOME");
    if (env && *env) {
        base = env;
    } else {
        env = std::getenv("HOME");
        if (!env || !*env) {
            fprintf(stderr, "error: neither LLAMA_CACHE, XDG_CACHE_HOME nor HOME is set; pass --cache-dir\n");
            exit(1);
        }
        base = fs::path(env) / ".cache";
    }
    return base / "llama.cpp" / "rpc";
}

struct job {
    const uint8_t * data;
    size_t          size;
    std::string     name;
};

static bool write_all(int fd, const uint8_t * data, size_t size) {
    while (size > 0) {
        ssize_t n = write(fd, data, size < ((size_t) 1 << 30) ? size : ((size_t) 1 << 30));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        data += n;
        size -= (size_t) n;
    }
    return true;
}

static void usage(const char * argv0) {
    fprintf(stderr, "Usage: %s [--cache-dir DIR] [--threads N] [--dry-run] MODEL.gguf [MODEL2.gguf ...]\n", argv0);
}

int main(int argc, char ** argv) {
    fs::path cache_dir;
    int n_threads = 4;
    bool dry_run = false;
    std::vector<std::string> models;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--cache-dir" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        } else {
            models.push_back(arg);
        }
    }
    if (models.empty() || n_threads <= 0) {
        usage(argv[0]);
        return 1;
    }
    if (cache_dir.empty()) {
        cache_dir = default_cache_dir();
    }
    if (!dry_run) {
        std::error_code ec;
        fs::create_directories(cache_dir, ec);
        if (ec) {
            fprintf(stderr, "error: cannot create %s: %s\n", cache_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }
    printf("cache directory: %s%s\n", cache_dir.c_str(), dry_run ? " (dry run)" : "");

    int rc = 0;
    for (const auto & model : models) {
        struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        struct gguf_context * gctx = gguf_init_from_file(model.c_str(), params);
        if (!gctx) {
            fprintf(stderr, "error: not a readable GGUF file: %s\n", model.c_str());
            rc = 1;
            continue;
        }

        int fd = open(model.c_str(), O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            fprintf(stderr, "error: cannot open %s: %s\n", model.c_str(), strerror(errno));
            gguf_free(gctx);
            rc = 1;
            continue;
        }
        const size_t file_size = (size_t) st.st_size;
        void * map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) {
            fprintf(stderr, "error: mmap failed for %s: %s\n", model.c_str(), strerror(errno));
            close(fd);
            gguf_free(gctx);
            rc = 1;
            continue;
        }
        madvise(map, file_size, MADV_SEQUENTIAL);

        const size_t data_offset = gguf_get_data_offset(gctx);
        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        std::vector<job> jobs;
        size_t n_small = 0;
        bool bad = false;
        for (int64_t t = 0; t < n_tensors; t++) {
            const size_t size = gguf_get_tensor_size(gctx, t);
            const size_t offs = data_offset + gguf_get_tensor_offset(gctx, t);
            if (offs > file_size || size > file_size - offs) {
                fprintf(stderr, "error: tensor %s exceeds the file size (truncated download?)\n", gguf_get_tensor_name(gctx, t));
                bad = true;
                break;
            }
            if (size <= HASH_THRESHOLD) {
                n_small++;
                continue;
            }
            jobs.push_back({ (const uint8_t *) map + offs, size, gguf_get_tensor_name(gctx, t) });
        }

        if (!bad) {
            printf("%s: %" PRId64 " tensors, %zu above the hash threshold, %zu below (sent inline at load)\n",
                   model.c_str(), n_tensors, jobs.size(), n_small);

            std::atomic<size_t> next{0}, n_written{0}, n_present{0}, n_failed{0}, bytes{0};
            std::mutex log_mutex;
            auto worker = [&]() {
                for (;;) {
                    const size_t idx = next.fetch_add(1);
                    if (idx >= jobs.size()) {
                        return;
                    }
                    const job & j = jobs[idx];
                    char hash_str[17];
                    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, fnv_hash(j.data, j.size));
                    const fs::path target = cache_dir / hash_str;

                    std::error_code ec;
                    if (fs::exists(target, ec) && fs::file_size(target, ec) == j.size) {
                        n_present++;
                        continue;
                    }
                    if (dry_run) {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        printf("  would write %s  %10zu bytes  %s\n", hash_str, j.size, j.name.c_str());
                        n_written++;
                        bytes += j.size;
                        continue;
                    }
                    // Write under a temporary name: rpc-server must never see a partial file.
                    const fs::path tmp = cache_dir / (std::string(hash_str) + ".tmp." + std::to_string(getpid()));
                    int out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                    bool ok = out >= 0 && write_all(out, j.data, j.size);
                    if (out >= 0) {
                        ok = close(out) == 0 && ok;
                    }
                    if (ok) {
                        fs::rename(tmp, target, ec);
                        ok = !ec;
                    }
                    if (!ok) {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        fprintf(stderr, "error: writing %s for tensor %s failed: %s\n", target.c_str(), j.name.c_str(),
                                ec ? ec.message().c_str() : strerror(errno));
                        fs::remove(tmp, ec);
                        n_failed++;
                        continue;
                    }
                    n_written++;
                    bytes += j.size;
                }
            };
            std::vector<std::thread> threads;
            for (int i = 0; i < n_threads; i++) {
                threads.emplace_back(worker);
            }
            for (auto & th : threads) {
                th.join();
            }
            printf("%s: %zu cache files %s (%.1f GiB), %zu already present, %zu failed\n", model.c_str(),
                   n_written.load(), dry_run ? "to write" : "written", bytes.load() / 1073741824.0, n_present.load(),
                   n_failed.load());
            if (n_failed > 0) {
                rc = 1;
            }
        } else {
            rc = 1;
        }

        munmap(map, file_size);
        close(fd);
        gguf_free(gctx);
    }
    return rc;
}
