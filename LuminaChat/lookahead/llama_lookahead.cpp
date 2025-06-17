// ============================================================================
// LLAMA LOOKAHEAD DECODING IMPLEMENTATION
// ============================================================================
// This file implements lookahead decoding for faster text generation
// using n-gram verification and Jacobi iteration patterns.
// ============================================================================

// ============================================================================
// HEADERS AND INCLUDES
// ============================================================================

#include "llama-cpp.h" // CRITICAL: must use the llama-cpp.h variant NOT llama.h

// Standard library headers
#include <cstdio>
#include <cstdarg>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <functional>
#include <memory>
#include <sstream>
#include <regex>
#include <chrono>

// Platform-specific headers
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <locale>
#include <windows.h>
#include <fcntl.h>
#include <io.h>

// ============================================================================
// PLATFORM-SPECIFIC DEFINITIONS AND MACROS
// ============================================================================

#define strdup _strdup

// Helper function for timing
inline int64_t ggml_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// ============================================================================
// CONSTANTS AND ENUMS
// ============================================================================

#define LOG_DEFAULT_LLAMA 0

// Sampler types for different sampling strategies
enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE = 0,
    COMMON_SAMPLER_TYPE_DRY = 1,
    COMMON_SAMPLER_TYPE_TOP_K = 2,
    COMMON_SAMPLER_TYPE_TOP_P = 3,
    COMMON_SAMPLER_TYPE_MIN_P = 4,
    COMMON_SAMPLER_TYPE_TYPICAL_P = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC = 8,
    COMMON_SAMPLER_TYPE_INFILL = 9,
    COMMON_SAMPLER_TYPE_PENALTIES = 10,
    COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
};

// Grammar trigger types for different trigger patterns
enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Token handling functions
std::vector<llama_token> common_tokenize(
    const struct llama_context * ctx,
    const std::string & text,
    bool add_special,
    bool parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
    const std::string & text,
    bool add_special,
    bool parse_special = false);

std::string common_token_to_piece(
    const struct llama_context * ctx,
    llama_token token,
    bool special = true);

std::string common_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    bool special = true);

// Sampler functions
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params);
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// ============================================================================
// DATA STRUCTURES AND CLASSES
// ============================================================================

// Ring buffer template for efficient token storage
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
// Simple logging structure
struct simple_log {
    void log(const char* level, const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        
        FILE* output = (strcmp(level, "ERROR") == 0) ? stderr : stdout;
        if (strlen(level) > 0) {
            fprintf(output, "[%s] ", level);
        }
        vfprintf(output, fmt, args);
        fflush(output);
        
        va_end(args);
    }
};

// CPU parameters for thread configuration
struct cpu_params {
    int      n_threads                   = -1;
};

// Model parameters structure
struct common_params_model {
    std::string path    = "";
};

// Grammar trigger configuration
struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

// Sampling parameters for text generation - streamlined for lookahead decoding
struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED;

    int32_t n_prev = 64;
    int32_t min_keep = 0;
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   min_p = 0.05f;
    float   xtc_probability = 0.00f;
    float   xtc_threshold = 0.10f;
    float   typ_p = 1.00f;
    float   temp = 0.80f;
    float   dynatemp_range = 0.00f;
    float   dynatemp_exponent = 1.00f;
    int32_t penalty_last_n = 64;
    float   penalty_repeat = 1.00f;
    float   penalty_freq = 0.00f;
    float   penalty_present = 0.00f;
    float   dry_multiplier = 0.0f;
    float   dry_base = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;
    int32_t mirostat = 0;
    float   top_n_sigma = -1.00f;
    float   mirostat_tau = 5.00f;
    float   mirostat_eta = 0.10f;
    
    std::vector<std::string> dry_sequence_breakers = { "\n", ":", "\"", "*" };

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

    std::string grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;

    std::vector<llama_logit_bias> logit_bias;
};

// Main parameters structure containing only the configuration options used in lookahead decoding
struct common_params {
    uint32_t seed         = LLAMA_DEFAULT_SEED;
    int32_t n_ctx         = 0;
    int32_t n_batch       = 2048;
    int32_t n_ubatch      = 512;
    int32_t n_predict     = -1;
    int32_t n_parallel    = 1;
    int32_t n_gpu_layers  = 999;
    int32_t main_gpu      = 0;
    float   tensor_split[128] = {0};
    float   rope_freq_base   = 0.0f;
    float   rope_freq_scale  = 0.0f;
    int rope_scaling_type    = -1;   // LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    float   yarn_ext_factor  = -1.0f;
    float   yarn_attn_factor = 1.0f;
    float   yarn_beta_fast   = 32.0f;
    float   yarn_beta_slow   = 1.0f;
    int32_t yarn_orig_ctx    = 0;
    float   defrag_thold     = -1.0f;
    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    common_params_sampling sampling_params;
    common_params_model model;

    std::string prompt = "";

    cpu_params cpuparams;
    cpu_params cpuparams_batch;

    std::vector<llama_model_kv_override> kv_overrides;

    bool flash_attn        = false;
    bool no_perf           = false;
    bool use_mmap          = true;
    bool use_mlock         = false;
    bool no_kv_offload     = false;
    bool check_tensors     = false;
};

// Initialization result structure (note: defines object's lifetime)
struct common_init_result {
    llama_model_ptr   model;
    llama_context_ptr context;
    
    common_init_result() : model(nullptr), context(nullptr) {}
    
    common_init_result(common_init_result&&) = default;
    common_init_result& operator=(common_init_result&&) = default;
    
    // Delete copy constructor and assignment to prevent accidental copying
    common_init_result(const common_init_result&) = delete;
    common_init_result& operator=(const common_init_result&) = delete;
};

// Sampler structure for token sampling
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
        }
        
        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

// N-gram data structure for lookahead verification
struct ngram_data {
    bool active = false;

    llama_seq_id seq_id = -1;

    std::vector<int> i_batch;

    std::vector<llama_token> tokens;
};

// N-gram container for efficient n-gram storage and lookup
struct ngram_container {
    ngram_container(int n_vocab, int N, int G) {
        cnt.resize(n_vocab);
        head.resize(n_vocab);
        tokens.resize(n_vocab * G * (N - 1));
        
        // Use STL algorithms for initialization
        std::fill(cnt.begin(), cnt.end(), 0);
        std::fill(head.begin(), head.end(), 0);
    }

    int n_total = 0;

    std::vector<int> cnt;
    std::vector<int> head;

    // [n_vocab][G][N - 1]
    // for each token of the vocab, keep a ring-buffer of capacity G of n-grams of size N - 1
    std::vector<llama_token> tokens;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// String utility functions
inline std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$0");
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

// ============================================================================
// SIMPLIFIED LOGGING SYSTEM
// ============================================================================

inline simple_log * get_main_log() {
    static simple_log log;
    return &log;
}

#define LOG(...)     get_main_log()->log("", __VA_ARGS__)
#define LOG_INF(...) get_main_log()->log("INFO", __VA_ARGS__)
#define LOG_ERR(...) get_main_log()->log("ERROR", __VA_ARGS__)

// ============================================================================
// TOKEN HANDLING FUNCTIONS
// ============================================================================

inline std::vector<llama_token> common_tokenize(
    const struct llama_context * ctx,
    const std::string & text,
    bool add_special,
    bool parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

inline std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
    const std::string & text,
    bool add_special,
    bool parse_special) {
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
    }
    return piece;
}

// ============================================================================
// BATCH HANDLING FUNCTIONS
// ============================================================================

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

// ============================================================================
// PARAMETER CONVERSION FUNCTIONS
// ============================================================================

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

inline common_init_result common_init_from_params(common_params & params) {
    common_init_result iparams;
    auto mparams = common_model_params_to_llama(params);

    ggml_backend_load_all();

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return iparams;
    }    auto cparams = common_context_params_to_llama(params);

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        llama_model_free(model);
        return iparams;    }

    iparams.model.reset(model);
    iparams.context.reset(lctx);    return iparams;
}

// ============================================================================
// PERFORMANCE AND UTILITY FUNCTIONS
// ============================================================================

void common_init() {
    llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
        simple_log* log = get_main_log();
        const char* level_str = "";
        if (level == GGML_LOG_LEVEL_ERROR) level_str = "ERROR";
        else if (level == GGML_LOG_LEVEL_WARN) level_str = "WARN";
        else if (level == GGML_LOG_LEVEL_INFO) level_str = "INFO";
        log->log(level_str, "%s", text);
    }, NULL);
}

// Parameter parsing function - simplified for lookahead decoding
inline bool common_params_parse(int argc, char ** argv, common_params & params, void(*print_usage)(int, char **)) {
    // Simple argument parsing for lookahead-specific parameters
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
                params.n_predict = std::stoi(argv[i]);
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

// Overload for backward compatibility
inline bool common_params_parse(int argc, char ** argv, common_params & params) {
    return common_params_parse(argc, argv, params, nullptr);
}

// ============================================================================
// MAIN LOOKAHEAD DECODING ALGORITHM
// ============================================================================

// ============================================================================
// SAMPLER IMPLEMENTATION
// ============================================================================

inline struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = false; // Use default value since removed from sampling params

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

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
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

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);    }
    
    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_batch_add(
    struct llama_batch & batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id> & seq_ids,
    bool logits) {
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

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        llama_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        llama_perf_context_print(ctx);
    }
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

int main(int argc, char** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params)) {
        return 1;
    }

    common_init();

    const int W = 15; // lookahead window
    const int N = 5;  // n-gram size
    const int G = 15; // max verification n-grams

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the target model
    common_init_result llama_init = common_init_from_params(params);    llama_model* model = llama_init.model.get();
    llama_context* ctx = llama_init.context.get();

    auto* mem = llama_get_memory(ctx);

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    std::vector<llama_token> all;

    inp = common_tokenize(ctx, params.prompt, true, true);
    all = inp;

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int)inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int)inp.size(), max_tokens_list_size);
        return 1;
    }    LOG("\n\n");

    // Use range-based for loop for token printing
    for (const auto& token_id : inp) {
        LOG("%s", common_token_to_piece(ctx, token_id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();    // eval the prompt
    llama_decode(ctx, llama_batch_get_one(inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(), 1));    // Use STL algorithm to initialize memory sequences
    std::vector<int> seq_range(W + G);
    std::iota(seq_range.begin(), seq_range.end(), 1);
    std::for_each(seq_range.begin(), seq_range.end(), 
        [&mem](int s) { 
            llama_memory_seq_cp(mem, 0, s, -1, -1);
        });

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_accept = 0;

    int n_past = inp.size();

    llama_token id = 0;

    // used to determine end of generation
    bool has_eos = false;    // for each decoded batch, we have at most W + G + 1 distinct sequences:
    // seq_id == 0           : the current input token
    // seq_id [1, W]         : tokens from the past N - 1 Jacobi iterations
    // seq_id [W + 1, W + G] : verification n-grams
    
    // Calculate maximum batch size needed:
    // 1 token for current + G*(N-1) for verification + (W-1) for first level + (N-1)*W for other levels
    const int max_batch_size = 1 + G * (N - 1) + (W - 1) + (N - 1) * W;
    
    llama_batch batch = llama_batch_init(max_batch_size, 0, W + G + 1);

    // target model sampling context
    struct common_sampler* smpl = common_sampler_init(model, params.sampling_params);    // verification n-grams
    std::vector<ngram_data> ngrams_cur;
    ngrams_cur.reserve(G); // Pre-reserve capacity    // tokens for the past N - 1 Jacobi iterations
    std::vector<llama_token> tokens_j_prev;
    tokens_j_prev.reserve(W);
    std::vector<std::vector<llama_token>> tokens_j(N - 1);
    
    // Use STL algorithms for better initialization
    std::for_each(tokens_j.begin(), tokens_j.end(), [W](std::vector<llama_token>& vec) {
        vec.reserve(W);
        vec.resize(W);
    });
      for (int j = 0; j < N - 1; j++) {
        // Use STL algorithm for initialization instead of manual loop
        if (0) {
            // initialize randomly from the prompt tokens
            if (all.size() > 1) {
                std::generate(tokens_j[j].begin(), tokens_j[j].end(), 
                    [&all]() { return all[1 + rand() % (all.size() - 1)]; });
            } else {
                // fallback if all has insufficient tokens
                std::fill(tokens_j[j].begin(), tokens_j[j].end(), all.empty() ? 0 : all[0]);
            }
        }
        else {
            // initialize with a sequence of increasing numbers
            std::iota(tokens_j[j].begin(), tokens_j[j].end(), 100);
        }
    }std::vector<llama_seq_id> seq_id_look;
    seq_id_look.reserve(W); // Reserve capacity to avoid reallocations

    // the input token belongs both to all sequences
    std::vector<llama_seq_id> seq_id_all(W + G + 1);
    std::iota(seq_id_all.begin(), seq_id_all.end(), 0); // Use iota instead of manual loop

    // here we keep adding new n-grams as we go
    ngram_container ngrams_observed(llama_vocab_n_tokens(vocab), N, G);

    const auto t_dec_start = ggml_time_us();

    // sample first token
    {
        id = common_sampler_sample(smpl, ctx, 0);

        common_sampler_accept(smpl, id, true);

        {
            const std::string token_str = common_token_to_piece(ctx, id);

            LOG("%s", token_str.c_str());
            fflush(stdout);
        }
    }

    while (true) {
        // build the mask from https://lmsys.org/blog/2023-11-21-lookahead-decoding/
        //
        // Example for W = 5, N = 4, G = 2:
        // (I = input, L = lookahead, V = verification)
        //
        // Batch:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
        // T:        -2 -2 -2 -2 -1 -1 -1 -1 -1  0  0  0  0  0  0
        // Info:   I  L  L  L  L  L  L  L  L  L  L  L  L  L  L  V  V  V  V  V  V
        // Pos:    0  1  2  3  4  1  2  3  4  5  2  3  4  5  6  1  2  3  1  2  3   (+ n_past)
        // Logits: 1  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1
        // ---------------------------------------------------------------------
        // Seq:    0
        //         1              1              1
        //         2  2              2              2
        //         3  3  3              3              3
        //         4  4  4  4              4              4
        //         5  5  5  5  5              5              5
        //         6                                            6  6  6
        //         7                                                     7  7  7
        // ---------------------------------------------------------------------
        //                                       |  |  |  |  |  |  |  |  |  |  |
        //                                       V  V  V  V  V  |  |  |  |  |  |
        //                                         j_tokens     |  |  |  |  |  |
        //                                                      V  V  V  V  V  V
        //                                                             id
        {
            common_batch_clear(batch);

            // current token - first token of the first level
            common_batch_add(batch, id, n_past, seq_id_all, true);            // verification n-grams - queue this before the lookahead tokens for less KV cache fragmentation
            {
                const int g_cur = ngrams_observed.cnt[id];

                ngrams_cur.resize(g_cur);
                
                // Use STL algorithm to initialize ngram data structures
                std::generate_n(ngrams_cur.begin(), g_cur, [&, g = 0]() mutable -> ngram_data {
                    ngram_data data;
                    data.active = true;
                    data.tokens.reserve(N);
                    data.tokens.resize(N);
                    data.i_batch.reserve(N);
                    data.i_batch.resize(N);
                    data.seq_id = W + 1 + g;
                    data.i_batch[0] = 0;
                    data.tokens[0] = id;
                    ++g;
                    return data;
                });

                for (int j = 0; j < N - 1; j++) {
                    for (int g = 0; g < g_cur; g++) {
                        const int idx = id * (N - 1) * G + g * (N - 1);

                        const llama_token t = ngrams_observed.tokens[idx + j];

                        ngrams_cur[g].tokens[j + 1] = t;
                        ngrams_cur[g].i_batch[j + 1] = batch.n_tokens;

                        common_batch_add(batch, t, n_past + j + 1, { W + 1 + g }, true);
                    }
                }
            }
            
            // fill the remaining W - 1 tokens for the first level
            for (int i = 1; i < W; i++) {
                const int seq_count = W - i;
                seq_id_look.resize(seq_count);
                std::iota(seq_id_look.begin(), seq_id_look.end(), i + 1); // Use iota for sequence

                common_batch_add(batch, tokens_j[0][i], n_past + i, seq_id_look, false);
            }

            // fill the rest of the levels
            for (int j = 1; j < N - 1; j++) {
                for (int i = 0; i < W; i++) {
                    common_batch_add(batch, tokens_j[j][i], n_past + j + i, { i + 1 }, j == N - 2);
                }
            }
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("\n\n%s: llama_decode failed - increase KV cache size\n", __func__);
            return 1;
        }

        int seq_id_best = 0;

        for (int v = 0; v < N; ++v) {
            int i_batch = 0;
            
            // if no active ngrams are left, it means the sampled token does not pass the verification
            if (v > 0) {
                // Use STL algorithm to find first active ngram
                auto it = std::find_if(ngrams_cur.begin(), ngrams_cur.end(),
                    [](const ngram_data& ngram) { return ngram.active; });
                
                if (it != ngrams_cur.end()) {
                    const int g = std::distance(ngrams_cur.begin(), it);
                    i_batch = ngrams_cur[g].i_batch[v];
                    seq_id_best = ngrams_cur[g].seq_id;
                    ++n_accept;
                }

                // no more matches -> create a new batch
                if (i_batch == 0) {
                    break;
                }
            }

            // sample the next token
            id = common_sampler_sample(smpl, ctx, i_batch);

            common_sampler_accept(smpl, id, true);

            // print
            {
                const std::string token_str = common_token_to_piece(ctx, id);

                if (v == 0) {
                    LOG("%s", token_str.c_str());
                }
                else {
                    // print light cyan
                    LOG("\033[0;96m%s\033[0m", token_str.c_str());
                }
                fflush(stdout);

                if (llama_vocab_is_eog(vocab, id)) {
                    has_eos = true;
                }

                all.push_back(id);
            }

            ++n_predict;
            ++n_past;

            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
                break;
            }
              // verify across active n-grams using STL algorithms
            std::for_each(ngrams_cur.begin(), ngrams_cur.end(), [v, N, id](ngram_data& ngram) {
                if (ngram.active) {
                    if (v == N - 1) {
                        ngram.active = false;
                    }
                    else {
                        ngram.active = (id == ngram.tokens[v + 1]);
                    }
                }
            });            // print known n-grams starting with token id (debug)
            if (0 && v == 0) {
                if (ngrams_observed.cnt[id] > 0) {
                    LOG("\n - %d n-grams starting with '%s'\n", ngrams_observed.cnt[id], common_token_to_piece(ctx, id).c_str());
                }

                // Use STL algorithm for debug printing
                std::vector<int> ngram_indices(ngrams_observed.cnt[id]);
                std::iota(ngram_indices.begin(), ngram_indices.end(), 0);
                
                std::for_each(ngram_indices.begin(), ngram_indices.end(), [&](int i) {
                    LOG("   - ngram %2d: ", i);

                    const int idx = id * (N - 1) * G + i * (N - 1);

                    for (int j = 0; j < N - 1; j++) {
                        const std::string token_str = common_token_to_piece(ctx, ngrams_observed.tokens[idx + j]);
                        LOG("%s", token_str.c_str());
                    }

                    LOG("\n");
                });
            }
              // update lookahead tokens
            {
                // Resize tokens_j_prev if needed and use STL copy
                tokens_j_prev.resize(tokens_j[0].size());
                std::copy(tokens_j[0].begin(), tokens_j[0].end(), tokens_j_prev.begin());

                // Shift tokens using move semantics for better performance
                std::rotate(tokens_j.begin(), tokens_j.begin() + 1, tokens_j.end());

                if (v == 0) {
                    // sample from the last level
                    auto& last_level = tokens_j.back();
                    last_level.resize(W);
                    
                    // Use STL generate for sampling
                    std::generate(last_level.begin(), last_level.end(), [&, i = 0]() mutable {
                        return common_sampler_sample(smpl, ctx, ngrams_cur.size() * (N - 1) + W * (N - 2) + i++);
                    });
                }
                else {
                    auto& last_level = tokens_j.back();
                    last_level.resize(W);                    // there are different ways to init these tokens
                    if (0) {
                        // random init using STL algorithm
                        if (all.size() > 1) {
                            std::generate(last_level.begin(), last_level.end(),
                                [&all]() { return all[1 + rand() % (all.size() - 1)]; });
                        } else {
                            // fallback if all has insufficient tokens
                            std::fill(last_level.begin(), last_level.end(), all.empty() ? 0 : all[0]);
                        }
                    }
                    else {
                        // init from the previous level using STL copy
                        std::copy(tokens_j[0].begin(), tokens_j[0].end(), last_level.begin());
                    }
                }
            }
              // update observed ngrams
            if (v == 0) {
                // the first token of the n-gram is determined by the index in the container so it is not stored
                std::vector<llama_token> ngram(N - 1);

                // n-gram generation using STL algorithms
                // ref: https://github.com/hao-ai-lab/LookaheadDecoding/issues/14#issuecomment-1826198518
                std::vector<int> f_range(W);
                std::iota(f_range.begin(), f_range.end(), 0);
                
                std::for_each(f_range.begin(), f_range.end(), [&](int f) {
                    const int ft = tokens_j_prev[f]; // first token of the n-gram

                    // Use STL transform to extract ngram tokens more efficiently
                    std::transform(tokens_j.begin(), tokens_j.end(), ngram.begin(),
                        [f](const std::vector<llama_token>& tokens) { return tokens[f]; });

                    // filter-out repeating n-grams using STL algorithms
                    std::vector<int> k_range(ngrams_observed.cnt[ft]);
                    std::iota(k_range.begin(), k_range.end(), 0);
                    
                    bool is_unique = std::none_of(k_range.begin(), k_range.end(), [&](int k) {
                        const int idx = ft * (N - 1) * G + k * (N - 1);
                        // Use STL equal for comparison instead of manual loop
                        return std::equal(ngram.begin(), ngram.end(), 
                                        ngrams_observed.tokens.begin() + idx);
                    });

                    if (!is_unique) {
                        return; // Continue to next iteration
                    }

                    const int head = ngrams_observed.head[ft];
                    const int idx = ft * (N - 1) * G + head * (N - 1);

                    // Use STL copy for better performance
                    std::copy(ngram.begin(), ngram.end(), ngrams_observed.tokens.begin() + idx);

                    ngrams_observed.cnt[ft] = std::min(G, ngrams_observed.cnt[ft] + 1);
                    ngrams_observed.head[ft] = (head + 1) % G;

                    ngrams_observed.n_total++;
                });
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management        // if no verification token matched, we simply remove all cells from this batch -> no fragmentation
        llama_memory_seq_rm(mem, -1, n_past, -1);
        
        if (seq_id_best != 0) {
            // if a verification token matched, we keep the best sequence and remove the rest
            // this leads to some KV cache fragmentation
            llama_memory_seq_keep(mem, seq_id_best);
            llama_memory_seq_cp(mem, seq_id_best, 0, -1, -1);
            llama_memory_seq_rm(mem, seq_id_best, -1, -1);

            // Use STL algorithm for memory sequence copying
            std::vector<int> seq_copy_range(W + G);
            std::iota(seq_copy_range.begin(), seq_copy_range.end(), 1);
            std::for_each(seq_copy_range.begin(), seq_copy_range.end(),
                [mem](int s) { 
                    llama_memory_seq_cp(mem, 0, s, -1, -1);
                    (void)s; // suppress unused variable warning
                });
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input, (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("W = %2d\n", W);
    LOG_INF("N = %2d\n", N);
    LOG_INF("G = %2d\n", G);
    LOG_INF("\n");
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_accept  = %d\n", n_accept);

    LOG_INF("\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_batch_free(batch);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}