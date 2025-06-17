#pragma once

// Secondary functions and utilities extracted from arg.h, common.h, sampling.h, and log.h
// Required for llama_lookahead.cpp functionality

#include "llama-cpp.h" // CRITICAL: must use the llama-cpp.h variant NOT llama.h
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#ifdef _MSC_VER
#define strdup _strdup
#endif

// Build info
extern int LLAMA_BUILD_NUMBER;
extern const char * LLAMA_COMMIT;
extern const char * LLAMA_COMPILER;
extern const char * LLAMA_BUILD_TARGET;

//
// Type definitions
//

using llama_tokens = std::vector<llama_token>;

// Note: Deleter structs are already defined in llama-cpp.h - we don't redefine them here

// Smart pointer types
using llama_model_ptr = std::unique_ptr<llama_model, llama_model_deleter>;
using llama_context_ptr = std::unique_ptr<llama_context, llama_context_deleter>;
using llama_adapter_lora_ptr = std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter>;

//
// Enums
//

enum llama_example {
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_MAIN,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,

    LLAMA_EXAMPLE_COUNT,
};

enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE        = 0,
    COMMON_SAMPLER_TYPE_DRY         = 1,
    COMMON_SAMPLER_TYPE_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_MIN_P       = 4,
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC         = 8,
    COMMON_SAMPLER_TYPE_INFILL      = 9,
    COMMON_SAMPLER_TYPE_PENALTIES   = 10,
    COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

// Note: llama enums are defined in llama-cpp.h - using them directly

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    COMMON_REASONING_FORMAT_DEEPSEEK,
};

enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

//
// Structures
//

struct cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false};
    bool     mask_valid                  = false;
    ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;
    bool     strict_cpu                  = false;
    uint32_t poll                        = 50;
};

struct common_adapter_lora_info {
    std::string path;
    float scale;
    struct llama_adapter_lora * ptr;
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

struct common_control_vector_load_info {
    float strength;
    std::string fname;
};

// Ring buffer for token history
template<typename T>
struct ring_buffer {
    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;

    ring_buffer() = default;
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    void push_back(const T& item) {
        if (sz < capacity) {
            sz++;
        } else {
            first = (first + 1) % capacity;
        }
        data[pos] = item;
        pos = (pos + 1) % capacity;
    }

    size_t size() const { return sz; }
    bool empty() const { return sz == 0; }
    
    T& back() { 
        size_t idx = (pos == 0) ? capacity - 1 : pos - 1;
        return data[idx]; 
    }
      const T& back() const { 
        size_t idx = (pos == 0) ? capacity - 1 : pos - 1;
        return data[idx]; 
    }
};

struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED;

    int32_t n_prev             = 64;
    int32_t n_probs            = 0;
    int32_t min_keep           = 0;
    int32_t top_k              = 40;
    float   top_p              = 0.95f;
    float   min_p              = 0.05f;
    float   xtc_probability    = 0.00f;
    float   xtc_threshold      = 0.10f;
    float   typ_p              = 1.00f;
    float   temp               = 0.80f;
    float   dynatemp_range     = 0.00f;
    float   dynatemp_exponent  = 1.00f;
    int32_t penalty_last_n     = 64;
    float   penalty_repeat     = 1.00f;
    float   penalty_freq       = 0.00f;
    float   penalty_present    = 0.00f;
    float   dry_multiplier     = 0.0f;
    float   dry_base           = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;
    int32_t mirostat           = 0;
    float   top_n_sigma        = -1.00f;
    float   mirostat_tau       = 5.00f;
    float   mirostat_eta       = 0.10f;
    bool    ignore_eos         = false;
    bool    no_perf            = false;
    bool    timing_per_token   = false;

    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};

    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::set<llama_token>               preserved_tokens;

    std::vector<llama_logit_bias> logit_bias;

    std::string print() const;
};

// Common sampler structure
struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr = nullptr;
    struct llama_sampler * chain = nullptr;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p = {nullptr, 0, false};

    void set_logits(struct llama_context * ctx, int idx) {
        const auto * logits = llama_get_logits_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        cur.resize(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

struct common_params_model {
    std::string path    = "";
    std::string url     = "";
    std::string hf_repo = "";
    std::string hf_file = "";
};

struct common_params_speculative {
    std::vector<ggml_backend_dev_t> devices;
    int32_t n_ctx        =     0;
    int32_t n_max        =    16;
    int32_t n_min        =     0;
    int32_t n_gpu_layers =    999;
    float   p_split      =  0.1f;
    float   p_min        = 0.75f;

    cpu_params cpuparams;
    cpu_params cpuparams_batch;

    common_params_model model;
};

struct common_params_vocoder {
    common_params_model model;

    std::string speaker_file = "";
    bool use_guide_tokens = false;
};

// Main common_params structure
struct common_params {
    uint32_t seed         = LLAMA_DEFAULT_SEED;
    int32_t n_ctx         = 0;
    int32_t n_batch       = 2048;
    int32_t n_ubatch      = 512;    int32_t n_keep        = 0;
    int32_t n_predict     = -1;
    int32_t n_draft       = 5;
    int32_t n_chunks      = -1;
    int32_t n_parallel    = 1;
    int32_t n_sequences   = 1;
    float   p_split       = 0.1f;
    int32_t n_gpu_layers  = 999;
    int32_t n_gpu_layers_draft = 999;    int32_t main_gpu      = 0;
    int split_mode        = 1;      // LLAMA_SPLIT_MODE_LAYER
    float   tensor_split[128] = {0};
    int32_t grp_attn_n    = 1;
    int32_t grp_attn_w    = 512;    int32_t n_print       = -1;    float   rope_freq_base   = 0.0f;
    float   rope_freq_scale  = 0.0f;
    int rope_scaling_type    = -1;   // LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    float   yarn_ext_factor  = -1.0f;
    float   yarn_attn_factor = 1.0f;
    float   yarn_beta_fast   = 32.0f;
    float   yarn_beta_slow   = 1.0f;
    int32_t yarn_orig_ctx    = 0;    float   defrag_thold     = -1.0f;    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    int pooling_type = -1;      // LLAMA_POOLING_TYPE_UNSPECIFIED
    int attention_type = -1;    // LLAMA_ATTENTION_TYPE_UNSPECIFIED

    common_params_sampling sampling;

    common_params_model model;
    common_params_model model_draft;

    std::string model_alias           = "unknown";
    std::string model_url             = "";
    std::string hf_token              = "";
    std::string hf_repo               = "";
    std::string hf_file               = "";
    std::string prompt                = "";
    std::string prompt_file           = "";
    std::string path_prompt_cache     = "";
    std::string input_prefix          = "";
    std::string input_suffix          = "";
    std::string logdir                = "";

    std::vector<std::string> antiprompt;    std::vector<common_adapter_lora_info> lora_adapters;

    std::vector<common_control_vector_load_info> control_vectors;
    int32_t control_vector_layer_start = -1;
    int32_t control_vector_layer_end   = -1;

    std::vector<ggml_backend_dev_t> devices;
    std::vector<std::string> in_files;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    cpu_params cpuparams;
    cpu_params cpuparams_batch;
    cpu_params draft_cpuparams;
    cpu_params draft_cpuparams_batch;

    // sampling params
    std::vector<llama_model_kv_override> kv_overrides;

    bool hellaswag        = false;
    size_t hellaswag_tasks  = 400;

    bool winogrande       = false;
    size_t winogrande_tasks = 0;

    bool multiple_choice  = false;
    size_t multiple_choice_tasks = 0;

    bool kl_divergence    = false;

    bool usage             = false;
    bool completion        = false;
    bool use_color         = false;
    bool special           = false;
    bool interactive       = false;
    bool interactive_first = false;
    bool prompt_cache_all  = false;
    bool prompt_cache_ro   = false;

    bool escape            = true;
    bool multiline_input   = false;
    bool simple_io         = false;
    bool cont_batching     = true;
    bool flash_attn        = false;
    bool no_perf           = false;
    bool ctx_shift         = true;    bool swa_full          = false;

    bool input_prefix_bos  = false;
    bool use_mmap          = true;
    bool use_mlock         = false;
    bool verbose_prompt    = false;
    bool display_prompt    = true;
    bool no_kv_offload     = false;
    bool warmup            = true;    bool check_tensors     = false;
    bool no_op_offload     = false;    bool single_turn       = false;
    bool offline           = false;
    bool lora_init_without_apply = false;

    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;

    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    // multimodal models
    common_params_model mmproj;
    bool mmproj_use_gpu = true;
    bool no_mmproj = false;
    std::vector<std::string> image;

    // embedding
    bool embedding         = false;
    int32_t embd_normalize = 2;
    std::string embd_out   = "";
    std::string embd_sep   = "\n";
    bool reranking         = false;

    // server params
    int32_t port           = 8080;
    int32_t timeout_read   = 600;
    int32_t timeout_write  = 600;
    int32_t n_threads_http = -1;
    int32_t n_cache_reuse  = 0;

    std::string hostname      = "127.0.0.1";
    std::string public_path   = "";
    std::string chat_template = "";
    bool use_jinja = false;
    bool enable_chat_template = true;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int reasoning_budget = -1;
    bool prefill_assistant = true;

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";
    std::string ssl_file_cert = "";

    bool webui            = true;
    bool endpoint_slots   = false;
    bool endpoint_props   = false;
    bool endpoint_metrics = false;

    bool log_json = false;

    std::string slot_save_path;

    float slot_prompt_similarity = 0.5f;

    // batched-bench params
    bool is_pp_shared = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // Missing members from original structure
    std::vector<std::string> context_files;
    int32_t chunk_size = 64;
    std::string chunk_separator = "\n";
    int32_t n_junk = 250;
    int32_t i_pos = -1;
    int32_t n_out_freq = 10;
    int32_t n_save_freq = 0;
    int32_t i_chunk = 0;
    bool process_output = false;
    bool compute_ppl = true;
    bool parse_special = false;    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    std::string cvector_negative_file = "tools/cvector-generator/negative.txt";
    bool spm_infill = false;
    bool batched_bench_output_jsonl = false;
    std::string out_file;
    int32_t verbosity = 0;
    int32_t ppl_stride = 0;
    int32_t ppl_output_type = 0;
    std::string lookup_cache_static = "";
    std::string lookup_cache_dynamic = "";
    std::string logits_file = "";    std::string system_prompt = "";
    llama_progress_callback load_progress_callback = NULL;
    void * load_progress_callback_user_data = NULL;

    common_params_speculative speculative;
    common_params_vocoder vocoder;
};

// note: defines object's lifetime
struct common_init_result {
    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;
};

// Forward declaration for common_sampler
struct common_sampler;

//
// Function declarations
//

// Common initialization and cleanup
void common_init();
struct common_init_result common_init_from_params(common_params & params);

// Command line argument parsing
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// Model and context parameter conversion
struct llama_model_params     common_model_params_to_llama  (      common_params & params);
struct llama_context_params   common_context_params_to_llama(const common_params & params);
struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params);

// Batch operations
void common_batch_clear(struct llama_batch & batch);
void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

// Tokenization functions
std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::string common_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);

std::string common_detokenize(
            const struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
              const struct llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

// Sampling functions
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params);
void common_sampler_free(struct common_sampler * gsmpl);
void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar);
void common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl);
llama_token common_sampler_last(const struct common_sampler * gsmpl);
std::string common_sampler_print(const struct common_sampler * gsmpl);
std::string common_sampler_prev_str(struct common_sampler * gsmpl, struct llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type type);
std::string common_sampler_type_to_str(enum common_sampler_type type);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab, const char * grammar_kind, const char * grammar_data);

// Performance printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

//
// Logging macros and functions
//

// Log levels constants
#define LOG_DEFAULT_DEBUG 1
#define LOG_DEFAULT_LLAMA 0

// Color codes
#define LOG_CLR_TO_EOL  "\033[K\r"
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD    "\033[1m"
#define LOG_COL_RED     "\033[31m"
#define LOG_COL_GREEN   "\033[32m"
#define LOG_COL_YELLOW  "\033[33m"
#define LOG_COL_BLUE    "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN    "\033[36m"
#define LOG_COL_WHITE   "\033[37m"

#ifndef __GNUC__
#    define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

// Global verbosity threshold
extern int common_log_verbosity_thold;

void common_log_set_verbosity_thold(int verbosity);

// Log entry structure
struct common_log_entry {
    enum ggml_log_level level;
    bool prefix;
    int64_t timestamp;
    std::vector<char> msg;
    bool is_end;

    void print(FILE * file = nullptr) const {
        FILE * fcur = file;
        if (!fcur) {
            fcur = (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) ? stderr : stdout;
        }

        if (level != GGML_LOG_LEVEL_NONE && level != GGML_LOG_LEVEL_CONT && prefix) {
            switch (level) {
                case GGML_LOG_LEVEL_INFO:  fprintf(fcur, "I "); break;
                case GGML_LOG_LEVEL_WARN:  fprintf(fcur, "W "); break;
                case GGML_LOG_LEVEL_ERROR: fprintf(fcur, "E "); break;
                case GGML_LOG_LEVEL_DEBUG: fprintf(fcur, "D "); break;
                default: break;
            }
        }

        fprintf(fcur, "%s", msg.data());
        fflush(fcur);
    }
};

// Common log structure
struct common_log {
    std::mutex mtx;
    std::thread thrd;
    std::condition_variable cv;

    FILE * file;
    bool prefix;
    bool timestamps;
    bool running;
    int64_t t_start;

    std::vector<common_log_entry> entries;
    size_t head;
    size_t tail;
    common_log_entry cur;

    common_log() : common_log(256) {}

    common_log(size_t capacity) {
        file = nullptr;
        prefix = false;
        timestamps = false;
        running = false;
        t_start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        entries.resize(capacity);
        for (auto & entry : entries) {
            entry.msg.resize(1024);
        }

        head = 0;
        tail = 0;
        resume();
    }

    ~common_log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

    void add(enum ggml_log_level level, const char * fmt, va_list args) {
        std::lock_guard<std::mutex> lock(mtx);

        if (!running) {
            return;
        }

        auto & entry = entries[tail];

        va_list args_copy;
        va_copy(args_copy, args);

        const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
        if (n >= entry.msg.size()) {
            entry.msg.resize(n + 1);
            vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args_copy);
        }
        va_end(args_copy);

        entry.level = level;
        entry.prefix = prefix;
        entry.timestamp = 0;
        if (timestamps) {
            entry.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
        }
        entry.is_end = false;

        tail = (tail + 1) % entries.size();
        if (tail == head) {
            head = (head + 1) % entries.size();
        }

        cv.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock(mtx);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return head != tail; });

                cur = entries[head];
                head = (head + 1) % entries.size();

                lock.unlock();

                if (cur.is_end) {
                    break;
                }

                cur.print(file);
            }
        });
    }

    void pause() {
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!running) {
                return;
            }

            running = false;

            auto & entry = entries[tail];
            entry.is_end = true;
            tail = (tail + 1) % entries.size();

            cv.notify_one();
        }

        if (thrd.joinable()) {
            thrd.join();
        }
    }

    void set_file(const char * path) {
        if (file) {
            fclose(file);
        }
        file = fopen(path, "w");
    }

    void set_colors(bool colors) {
        // Implementation would set color codes
    }

    void set_prefix(bool prefix_val) {
        prefix = prefix_val;
    }

    void set_timestamps(bool timestamps_val) {
        timestamps = timestamps_val;
    }
};

struct common_log * common_log_init();
struct common_log * common_log_main();
void                common_log_pause (struct common_log * log);
void                common_log_resume(struct common_log * log);
void                common_log_free  (struct common_log * log);

LOG_ATTRIBUTE_FORMAT(3, 4)
void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...);

void common_log_set_file      (struct common_log * log, const char * file);
void common_log_set_colors    (struct common_log * log,       bool   colors);
void common_log_set_prefix    (struct common_log * log,       bool   prefix);
void common_log_set_timestamps(struct common_log * log,       bool   timestamps);

// Logging macros
#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= common_log_verbosity_thold) { \
            common_log_add(common_log_main(), (level), __VA_ARGS__); \
        } \
    } while (0)

#define LOG(...)             LOG_TMPL(GGML_LOG_LEVEL_NONE, 0,         __VA_ARGS__)
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity, __VA_ARGS__)

#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  0,                 __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  0,                 __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, 0,                 __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_DEFAULT_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  0,                 __VA_ARGS__)

#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  verbosity, __VA_ARGS__)

//
// String utilities
//

std::string string_format(const char * fmt, ...);
std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
void string_replace_all(std::string & s, const std::string & search, const std::string & replace);
bool string_ends_with(const std::string_view & str, const std::string_view & suffix);
size_t string_find_partial_stop(const std::string_view & str, const std::string_view & stop);
std::string regex_escape(const std::string & s);
std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);
std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);
std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch);
void string_process_escapes(std::string & input);
bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);

//
// CPU utilities
//

int32_t cpu_get_num_physical_cores();
int32_t cpu_get_num_math();
void postprocess_cpu_params(cpu_params& cpuparams, const cpu_params* role_model);
bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]);
bool set_process_priority(enum ggml_sched_priority prio);

//
// Filesystem utilities
//

bool fs_validate_filename(const std::string & filename);
bool fs_create_directory_with_parents(const std::string & path);
std::string fs_get_cache_directory();
std::string fs_get_cache_file(const std::string & filename);

//
// System information
//

std::string common_params_get_system_info(const common_params & params);

//
// Token utilities
//

size_t common_lcp(const llama_tokens & a, const llama_tokens & b);
size_t common_lcs(const llama_tokens & a, const llama_tokens & b);

//
// Embedding utilities
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm);
float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utilities
//

struct common_control_vector_data {
    int n_embd;
    std::vector<float> data;
};

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos);

//
// LoRA adapter utilities
//

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora);

//
// Model endpoint utilities
//

std::string get_model_endpoint();

//
// Remote utilities
//

struct common_remote_params {
    std::vector<std::string> headers;
    long timeout = 0;
    long max_size = 0;
};

std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);
bool common_has_curl();

//
// Training utilities
//

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride);

//
// IMPLEMENTATION SECTION
//

// Global variables
int common_log_verbosity_thold = LOG_DEFAULT_LLAMA;

// Common initialization
inline void common_init() {
    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        if (LOG_DEFAULT_LLAMA <= common_log_verbosity_thold) {
            common_log_add(common_log_main(), level, "%s", text);
        }
    }, NULL);
}

// Log verbosity functions
inline void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

// Log functions
inline struct common_log * common_log_init() {
    return new common_log;
}

inline struct common_log * common_log_main() {
    static common_log log;
    return &log;
}

inline void common_log_pause(struct common_log * log) {
    log->pause();
}

inline void common_log_resume(struct common_log * log) {
    log->resume();
}

inline void common_log_free(struct common_log * log) {
    delete log;
}

inline void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, fmt, args);
    va_end(args);
}

inline void common_log_set_file(struct common_log * log, const char * file) {
    log->set_file(file);
}

inline void common_log_set_colors(struct common_log * log, bool colors) {
    log->set_colors(colors);
}

inline void common_log_set_prefix(struct common_log * log, bool prefix) {
    log->set_prefix(prefix);
}

inline void common_log_set_timestamps(struct common_log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}

// Model parameter conversion functions
inline struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }

    mparams.main_gpu        = params.main_gpu;
    mparams.tensor_split    = params.tensor_split;
    mparams.use_mmap        = params.use_mmap;
    mparams.use_mlock       = params.use_mlock;
    mparams.check_tensors   = params.check_tensors;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    return mparams;
}

inline struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.rope_scaling_type = (llama_rope_scaling_type)params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.defrag_thold      = params.defrag_thold;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.flash_attn        = params.flash_attn;
    cparams.no_perf           = params.no_perf;

    return cparams;
}

inline struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params) {
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(params.n_threads);

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio     = params.priority;
    tpp.poll     = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

inline common_init_result common_init_from_params(common_params & params) {
    common_init_result iparams;
    auto mparams = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return iparams;
    }

    auto cparams = common_context_params_to_llama(params);

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        llama_model_free(model);
        return iparams;
    }

    iparams.model.reset(model);
    iparams.context.reset(lctx);

    return iparams;
}

// Argument parsing stub - simplified version
inline bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // Simple argument parsing - would need full implementation for real use
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) {
                params.model.path = argv[i];
            }
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i < argc) {
                params.prompt = argv[i];
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (++i < argc) {
                // params.n_predict = std::stoi(argv[i]);
            }
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i < argc) {
                params.n_ctx = std::stoi(argv[i]);
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) {
                params.cpuparams.n_threads = std::stoi(argv[i]);
            }
        } else if (arg == "--help" || arg == "-h") {
            if (print_usage) {
                print_usage(argc, argv);
            }
            return false;
        }
    }
    
    // Set default model path if not provided
    if (params.model.path.empty()) {
        params.model.path = "C:\\Llama-3.2-4X3B-MOE-Hell-California-10B-D_AU-Q5_k_s.gguf";
    }
    
    return true;
}

// String utility functions
inline std::string string_format(const char * fmt, ...) {
    va_list args, args_copy;
    va_start(args, fmt);
    va_copy(args_copy, args);

    int size = vsnprintf(nullptr, 0, fmt, args);
    va_end(args);

    if (size < 0) {
        va_end(args_copy);
        return "";
    }

    std::string result;
    result.resize(size + 1);
    vsnprintf(&result[0], size + 1, fmt, args_copy);
    va_end(args_copy);

    result.resize(size);
    return result;
}

inline std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    if (values.empty()) {
        return "";
    }

    std::string result = values[0];
    for (size_t i = 1; i < values.size(); i++) {
        result += separator + values[i];
    }
    return result;
}

inline bool string_ends_with(const std::string_view & str, const std::string_view & suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// CPU utility functions
inline int32_t cpu_get_num_physical_cores() {
#if defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#elif defined(__linux__)
    return sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(__APPLE__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    return num_physical_cores;
#else
    return std::thread::hardware_concurrency();
#endif
}

inline int32_t cpu_get_num_math() {
    return cpu_get_num_physical_cores();
}

// Additional required implementations
inline uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return gsmpl ? gsmpl->params.seed : LLAMA_DEFAULT_SEED;
}

inline llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl) {
    return gsmpl ? &gsmpl->cur_p : nullptr;
}

inline llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return (gsmpl && !gsmpl->prev.empty()) ? gsmpl->prev.back() : 0;
}

inline std::string common_sampler_print(const struct common_sampler * gsmpl) {
    if (!gsmpl) return "";
    return gsmpl->params.print();
}

inline std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n) {
    if (!gsmpl || !ctx || gsmpl->prev.empty()) return "";
    
    const int start = std::max(0, (int)gsmpl->prev.size() - n);
    std::vector<llama_token> tokens;
    
    for (size_t i = start; i < gsmpl->prev.size(); i++) {
        tokens.push_back(gsmpl->prev.data[(gsmpl->prev.first + i) % gsmpl->prev.capacity]);
    }
    
    return common_detokenize(ctx, tokens, true);
}

// Sampling parameter print function
inline std::string common_params_sampling::print() const {
    return string_format(
            "repeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "top_k = %d, top_p = %.3f, min_p = %.3f, temp = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            top_k, top_p, min_p, temp);
}

// Additional missing vector sampling functions stubs
inline std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler * gsmpl, 
    struct llama_context * ctx, 
    const std::vector<int> & idxs, 
    const llama_tokens & draft, 
    bool grammar_first) {
    std::vector<llama_token> result;
    for (size_t i = 0; i < idxs.size() && i < draft.size() + 1; i++) {
        llama_token token = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        if (i < draft.size() && token != draft[i]) {
            break;
        }
        result.push_back(token);
        common_sampler_accept(gsmpl, token, true);
    }
    return result;
}

inline std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler * gsmpl, 
    struct llama_context * ctx, 
    const llama_tokens & draft, 
    bool grammar_first) {
    std::vector<int> idxs;
    for (size_t i = 0; i <= draft.size(); i++) {
        idxs.push_back(i);
    }
    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

// Stub implementations for other functions that might not be used by llama_lookahead.cpp
inline std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::vector<enum common_sampler_type> result;
    // Implementation would parse names and return corresponding types
    return result;
}

inline std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::vector<enum common_sampler_type> result;
    // Implementation would parse chars and return corresponding types
    return result;
}

//
// Missing function implementations
//

// Batch utils
inline void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

inline void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

// Token utils
inline std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

inline std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

inline std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

inline std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }    return piece;
}

// Sampler functions
inline struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    struct llama_sampler * grmr;
    if (params.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<std::string> patterns_anywhere;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    patterns_anywhere.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    patterns_anywhere.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        if (!patterns_anywhere.empty()) {
            trigger_patterns.push_back("^[\\s\\S]*?(" + string_join(patterns_anywhere, "|") + ")[\\s\\S]*");
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        grmr = params.grammar_lazy
             ? llama_sampler_init_grammar_lazy_patterns(vocab, params.grammar.c_str(), "root",
                                                        trigger_patterns_c.data(), trigger_patterns_c.size(),
                                                        trigger_tokens.data(), trigger_tokens.size())
             :      llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
        if (!grmr) {
            return nullptr;
        }
    }

    auto * result = new common_sampler {
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

    llama_sampler_chain_add(result->chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                params.logit_bias.size(),
                params.logit_bias.data()));

    if (params.mirostat == 0) {
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }

                        llama_sampler_chain_add(result->chain, llama_sampler_init_dry      (vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_top_k       (params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_top_p       (params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_top_n_sigma (params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_min_p       (params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_xtc         (params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_typical     (params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_temp_ext    (params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_infill      (vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    llama_sampler_chain_add(result->chain, llama_sampler_init_penalties   (params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));
    } else if (params.mirostat == 1) {
        llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat(params.top_k, params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    return result;
}

inline void common_sampler_free(struct common_sampler * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

inline void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

inline void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        llama_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        llama_perf_context_print(ctx);
    }
}

inline llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    if (grammar_first) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    const llama_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(grmr,  &cur_p);
    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

// String utility functions that might be missing
inline std::string regex_escape(const std::string & s) {    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$0");
}
