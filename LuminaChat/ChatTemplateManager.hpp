#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <array>

enum class TemplateSection {
    OVERARCHING_ENVIRONMENT,
    IDENTITY_DIRECTIVE,
    SYSTEM_PROMPT,
    OLD_CHAT_SUMMARY,
    SUMMARY,
    INTERNAL_REFLECTION,
    EMOTIONAL_STATE
};

struct TemplateVariable {
    std::string content;
    bool active = false;
    
    TemplateVariable() = default;
    TemplateVariable(std::string content, bool active = true) 
        : content(std::move(content)), active(active) {}
    
    // Move constructor for better performance
    TemplateVariable(TemplateVariable&& other) noexcept
        : content(std::move(other.content)), active(other.active) {}
    
    // Move assignment to avoid copies
    TemplateVariable& operator=(TemplateVariable&& other) noexcept {
        if (this != &other) [[likely]] {  // Self-assignment rare
            content = std::move(other.content);
            active = other.active;
        }
        return *this;
    }
    
    // Copy assignment for completeness
    TemplateVariable& operator=(const TemplateVariable& other) {
        if (this != &other) [[likely]] {  // Self-assignment rare
            content = other.content;
            active = other.active;
        }
        return *this;
    }
};

/**
 * Chat Template Manager - Direct template building without legacy Jinja2 processing
 * 
 * This implementation uses direct string building for template rendering:
 * - Direct variable substitution for the 7 key variables only
 * - Proper conversation history formatting with exact role handling
 * - Conditional section rendering for optional content
 * - No artificial processing or message echoing
 */
class ChatTemplateManager {
private:
    // Use array for better cache locality and faster access
    std::array<TemplateVariable, 7> sections;
    std::vector<std::string> summaries;
    
    // Enhanced caching system
    mutable std::string cached_rendered_template;
    mutable std::size_t cached_template_hash = 0;
    mutable bool template_dirty = true;
    
    // Pre-computed reserved token set for fast validation
    static constexpr std::array<std::string_view, 3> RESERVED_TOKENS = {
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
    };
    
    // Template section names for replacement - ONLY these 7 variables
    static constexpr std::array<std::string_view, 7> section_names = {
        "overarching_environment", "identity_directive", "system_prompt",
        "old_chat_summary", "summary", 
        "internal_reflection", "emotional_state"
    };
    
    // Compile-time constants for memory estimation
    static constexpr size_t HEADER_OVERHEAD_PER_SECTION = 100;
    static constexpr size_t HEADER_OVERHEAD_PER_MESSAGE = 150;
    static constexpr size_t BASE_TEMPLATE_OVERHEAD = 512;
    static constexpr size_t SUMMARY_SEPARATOR_OVERHEAD = 4;
    static constexpr size_t COMBINED_SUMMARIES_OVERHEAD = 100;
    
    // Fast validation using compile-time constants with branch prediction hints
    [[nodiscard]] static constexpr bool HasReservedTokens(std::string_view text) noexcept {
        for (const auto& token : RESERVED_TOKENS) {
            if (text.find(token) != std::string_view::npos) [[unlikely]] {
                return true;
            }
        }
        return false;
    }
    
    // Compute hash for cache invalidation (template sections only)
    [[nodiscard]] std::size_t ComputeTemplateHash() const noexcept;
    
    // Compute complete hash including message history for caching
    [[nodiscard]] std::size_t ComputeCompleteHash(const std::vector<std::pair<std::string, std::string>>& messages) const noexcept;
    
    // Optimized template building with pre-calculated sizes
    [[nodiscard]] std::string BuildTemplateInternal(const std::vector<std::pair<std::string, std::string>>& messages) const;
    
public:
    ChatTemplateManager();
    
    // Section management with move semantics
    void SetSection(TemplateSection section, std::string content, bool active = true);
    void SetSection(TemplateSection section, std::string_view content, bool active = true);
    void ActivateSection(TemplateSection section) noexcept;
    void DeactivateSection(TemplateSection section) noexcept;
    void ClearSection(TemplateSection section) noexcept;
    
    // Get section content and status
    [[nodiscard]] std::string_view GetSection(TemplateSection section) const noexcept;
    [[nodiscard]] bool IsSectionActive(TemplateSection section) const noexcept;
    
    // Template rendering with message history - optimized version
    [[nodiscard]] const std::string& RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages);
    
    // Get the last rendered template for debugging/inspection
    [[nodiscard]] const std::string& GetLastRenderedTemplate() const noexcept { return cached_rendered_template; }
    
    // Convenience methods for common operations with move semantics
    void UpdateEnvironment(std::string env);
    void UpdateEnvironment(std::string_view env);
    void UpdateIdentity(std::string identity);
    void UpdateIdentity(std::string_view identity);
    void UpdateSystemPrompt(std::string system_msg);
    void UpdateSystemPrompt(std::string_view system_msg);
    void UpdateSummary(std::string summary);
    void UpdateSummary(std::string_view summary);
    void UpdateOldChatSummary(std::string old_summary);
    void UpdateOldChatSummary(std::string_view old_summary);
    void UpdateInternalReflection(std::string reflection);
    void UpdateInternalReflection(std::string_view reflection);
    void UpdateEmotionalState(std::string emotional_state);
    void UpdateEmotionalState(std::string_view emotional_state);
    
    // Multiple summary management with move semantics
    void UpdateMultipleSummaries(std::vector<std::string> summary_list);
    void AddSummaryToList(std::string summary, size_t max_summaries = 5);
    void ClearAllSummaries() noexcept;
    [[nodiscard]] const std::vector<std::string>& GetAllSummaries() const noexcept { return summaries; }
    
    // Template status
    [[nodiscard]] bool IsTemplateDirty() const noexcept { return template_dirty; }
    void MarkDirty() noexcept { template_dirty = true; }
};

// Optimized hash computation for template caching with minimal memory access
[[nodiscard]] inline std::size_t ChatTemplateManager::ComputeTemplateHash() const noexcept {
    std::size_t hash = 0;
    constexpr std::size_t hash_multiplier = 0x9e3779b9;
    
    // Hash all sections with single pass and branch prediction
    for (size_t i = 0; i < sections.size(); ++i) {
        if (sections[i].active) [[unlikely]] {  // Most sections inactive most of the time
            const auto& content = sections[i].content;
            if (!content.empty()) [[likely]] {  // Active sections usually have content
                hash ^= std::hash<std::string>{}(content) + hash_multiplier + (hash << 6) + (hash >> 2);
            }
        }
    }
    
    // Hash summaries with branch prediction
    if (!summaries.empty()) [[unlikely]] {  // Summaries less common
        for (const auto& summary : summaries) {
            hash ^= std::hash<std::string>{}(summary) + hash_multiplier + (hash << 6) + (hash >> 2);
        }
    }
    
    return hash;
}

// Compute complete hash including message history for caching
[[nodiscard]] inline std::size_t ChatTemplateManager::ComputeCompleteHash(const std::vector<std::pair<std::string, std::string>>& messages) const noexcept {
    std::size_t hash = ComputeTemplateHash(); // Start with template hash
    constexpr std::size_t hash_multiplier = 0x9e3779b9;
    
    // Hash message history - this is the critical missing piece!
    for (const auto& [role, content] : messages) {
        hash ^= std::hash<std::string>{}(role) + hash_multiplier + (hash << 6) + (hash >> 2);
        hash ^= std::hash<std::string>{}(content) + hash_multiplier + (hash << 6) + (hash >> 2);
    }
    
    return hash;
}

// Fast string building with pre-calculated memory requirements
[[nodiscard]] inline std::string ChatTemplateManager::BuildTemplateInternal(const std::vector<std::pair<std::string, std::string>>& messages) const {
    // Pre-calculate total size needed to minimize allocations
    size_t total_size = BASE_TEMPLATE_OVERHEAD; // Use compile-time constant
    
    // Calculate section sizes with cache-friendly access pattern
    for (size_t i = 0; i < sections.size(); ++i) {
        if (sections[i].active && !sections[i].content.empty()) [[unlikely]] {
            total_size += sections[i].content.size() + HEADER_OVERHEAD_PER_SECTION;
        }
    }
    
    // Calculate summaries size
    size_t summaries_size = 0;
    for (const auto& summary : summaries) {
        if (!summary.empty()) [[likely]] {
            summaries_size += summary.size() + SUMMARY_SEPARATOR_OVERHEAD;
        }
    }
    if (summaries_size > 0) [[unlikely]] {
        total_size += summaries_size + COMBINED_SUMMARIES_OVERHEAD;
    }
    
    // Calculate messages size
    for (const auto& [role, content] : messages) {
        total_size += role.size() + content.size() + HEADER_OVERHEAD_PER_MESSAGE;
    }
    
    // Build the template with pre-allocated buffer
    std::string result;
    result.reserve(total_size);
    
    // Build single system message containing all sections
    std::string system_content;
    size_t system_size = 0;
    
    // Pre-calculate system content size
    for (size_t i = 0; i < sections.size(); ++i) {
        if (sections[i].active && !sections[i].content.empty()) {
            system_size += sections[i].content.size() + 20; // padding for section labels
        }
    }
    for (const auto& summary : summaries) {
        system_size += summary.size() + 10; // padding for separators
    }
    
    system_content.reserve(system_size);
    
    // Add sections to system content in logical order
    if (sections[0].active && !sections[0].content.empty()) { // OVERARCHING_ENVIRONMENT
        if (!system_content.empty()) system_content += "\n\n";
        system_content += "###THE ENVIRONMENT###\n";
        system_content += sections[0].content;
    }
    
    if (sections[1].active && !sections[1].content.empty()) { // IDENTITY_DIRECTIVE
        if (!system_content.empty()) system_content += "\n\n";
        system_content += "###YOUR PERSONA/IDENTITY###\n";
        system_content += sections[1].content;
    }
    
    if (sections[5].active && !sections[5].content.empty()) { // INTERNAL_REFLECTION
        if (!system_content.empty()) system_content += "\n\n";
        system_content += "###YOUR INTERNAL THOUGHTS AND REASONING###\n";
        system_content += sections[5].content;
    }
    
    if (sections[2].active && !sections[2].content.empty()) { // SYSTEM_PROMPT
        if (!system_content.empty()) system_content += "\n\n";
        system_content += "###CORE DIRECTIVES###\n";
        system_content += sections[2].content;
    }
    
    // Combined summaries
    if (sections[3].active || sections[4].active || !summaries.empty()) { // OLD_CHAT_SUMMARY, SUMMARY, or summaries vector
        std::string combined_summaries;
        
        if (sections[3].active && !sections[3].content.empty()) {
            combined_summaries += sections[3].content;
        }
        
        for (const auto& summary : summaries) {
            if (!summary.empty()) {
                if (!combined_summaries.empty()) {
                    combined_summaries += "\n\n";
                }
                combined_summaries += summary;
            }
        }
        
        if (sections[4].active && !sections[4].content.empty()) {
            if (!combined_summaries.empty()) {
                combined_summaries += "\n\n";
            }
            combined_summaries += sections[4].content;
        }
        
        if (!combined_summaries.empty()) {
            if (!system_content.empty()) system_content += "\n\n";
            system_content += "###OLD CHAT HISTORY###\n";
            system_content += combined_summaries;
        }
    }
    
    if (sections[6].active && !sections[6].content.empty()) { // EMOTIONAL_STATE
        if (!system_content.empty()) system_content += "\n\n";
        system_content += "###YOUR CURRENT EMOTIONAL STATE###\n";
        system_content += sections[6].content;
    }
    
    // Add the complete system message FIRST
    if (!system_content.empty()) {
        result += "<|start_header_id|>system<|end_header_id|>\n";
        result += system_content;
        result += "\n<|eot_id|>\n\n";
    }
    
    // Add conversation history AFTER system message
    for (const auto& [role, content] : messages) {
        if (role == "assistant") [[likely]] { // Assistant messages more common in history
            result += "<|start_header_id|>assistant<|end_header_id|>\n";
            result += content;
            result += "\n<|eot_id|>\n";
        } else {
            // CRITICAL: ALL user input has [USERNAME] prefix; Prompts allow the AI to interpret properly here
            result += "<|start_header_id|>user<|end_header_id|>\n[";
            result += role;
            result += "] ";
            result += content;
            result += "\n<|eot_id|>\n";
        }
    }
    
    // Add final assistant turn marker
    result += "<|start_header_id|>assistant<|end_header_id|>\n";
    
    return result;
}

inline ChatTemplateManager::ChatTemplateManager() {
    // Initialize all sections as inactive using array indexing
    for (auto& section : sections) {
        section = TemplateVariable("", false);
    }
    
    // Reserve space for common use cases
    summaries.reserve(8);
}

// Optimized section management with move semantics and fast validation
inline void ChatTemplateManager::SetSection(TemplateSection section, std::string content, bool active) {
    const auto index = static_cast<size_t>(section);
    if (index >= sections.size()) [[unlikely]] return;
    
    // Fast validation check
    if (!content.empty() && HasReservedTokens(content)) [[unlikely]] {
        throw std::invalid_argument("SetSection: Content contains reserved template tokens");
    }
    
    sections[index] = TemplateVariable(std::move(content), active);
    template_dirty = true;
}

inline void ChatTemplateManager::SetSection(TemplateSection section, std::string_view content, bool active) {
    const auto index = static_cast<size_t>(section);
    if (index >= sections.size()) [[unlikely]] return;
    
    // Fast validation check
    if (!content.empty() && HasReservedTokens(content)) [[unlikely]] {
        throw std::invalid_argument("SetSection: Content contains reserved template tokens");
    }
    
    sections[index] = TemplateVariable(std::string(content), active);
    template_dirty = true;
}

inline void ChatTemplateManager::ActivateSection(TemplateSection section) noexcept {
    const auto index = static_cast<size_t>(section);
    if (index < sections.size()) [[likely]] {
        // Assume: index is always valid for enum values
        sections[index].active = true;
        template_dirty = true;
    }
}

inline void ChatTemplateManager::DeactivateSection(TemplateSection section) noexcept {
    const auto index = static_cast<size_t>(section);
    if (index < sections.size()) [[likely]] {
        // Assume: index is always valid for enum values
        sections[index].active = false;
        template_dirty = true;
    }
}

inline void ChatTemplateManager::ClearSection(TemplateSection section) noexcept {
    const auto index = static_cast<size_t>(section);
    if (index < sections.size()) [[likely]] {
        // Assume: index is always valid for enum values
        sections[index].content.clear();
        sections[index].active = false;
        template_dirty = true;
    }
}

inline std::string_view ChatTemplateManager::GetSection(TemplateSection section) const noexcept {
    const auto index = static_cast<size_t>(section);
    if (index < sections.size()) [[likely]] {
        return std::string_view(sections[index].content);
    }
    return std::string_view{};
}

inline bool ChatTemplateManager::IsSectionActive(TemplateSection section) const noexcept {
    const auto index = static_cast<size_t>(section);
    if (index < sections.size()) [[likely]] {
        return sections[index].active;
    }
    return false;
}

// Optimized template rendering with intelligent caching
[[nodiscard]] inline const std::string& ChatTemplateManager::RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages) {
    // Fast input validation for messages
    for (const auto& [role, content] : messages) {
        if (role.empty() || content.empty()) [[unlikely]] {
            throw std::invalid_argument("RenderTemplate: Empty role or content detected in message history");
        }
        if (HasReservedTokens(content)) [[unlikely]] {
            throw std::invalid_argument("RenderTemplate: Message content contains reserved template tokens");
        }
    }
    
    // Check if we can use cached result
    if (!template_dirty) [[likely]] {
        const auto current_hash = ComputeCompleteHash(messages);
        if (current_hash == cached_template_hash && !cached_rendered_template.empty()) [[likely]] {
            // Cache hit - return existing template
            return cached_rendered_template;
        }
        cached_template_hash = current_hash;
    }
    
    // Build new template (cache miss or template dirty)
    cached_rendered_template = BuildTemplateInternal(messages);
    cached_template_hash = ComputeCompleteHash(messages);
    template_dirty = false;
    
    return cached_rendered_template;
}

// Optimized convenience methods with move semantics
inline void ChatTemplateManager::UpdateEnvironment(std::string env) {
    SetSection(TemplateSection::OVERARCHING_ENVIRONMENT, std::move(env), !env.empty());
}

inline void ChatTemplateManager::UpdateEnvironment(std::string_view env) {
    SetSection(TemplateSection::OVERARCHING_ENVIRONMENT, env, !env.empty());
}

inline void ChatTemplateManager::UpdateIdentity(std::string identity) {
    SetSection(TemplateSection::IDENTITY_DIRECTIVE, std::move(identity), !identity.empty());
}

inline void ChatTemplateManager::UpdateIdentity(std::string_view identity) {
    SetSection(TemplateSection::IDENTITY_DIRECTIVE, identity, !identity.empty());
}

inline void ChatTemplateManager::UpdateSystemPrompt(std::string system_msg) {
    SetSection(TemplateSection::SYSTEM_PROMPT, std::move(system_msg), !system_msg.empty());
}

inline void ChatTemplateManager::UpdateSystemPrompt(std::string_view system_msg) {
    SetSection(TemplateSection::SYSTEM_PROMPT, system_msg, !system_msg.empty());
}

inline void ChatTemplateManager::UpdateSummary(std::string summary) {
    SetSection(TemplateSection::SUMMARY, std::move(summary), !summary.empty());
}

inline void ChatTemplateManager::UpdateSummary(std::string_view summary) {
    SetSection(TemplateSection::SUMMARY, summary, !summary.empty());
}

inline void ChatTemplateManager::UpdateOldChatSummary(std::string old_summary) {
    SetSection(TemplateSection::OLD_CHAT_SUMMARY, std::move(old_summary), !old_summary.empty());
}

inline void ChatTemplateManager::UpdateOldChatSummary(std::string_view old_summary) {
    SetSection(TemplateSection::OLD_CHAT_SUMMARY, old_summary, !old_summary.empty());
}

inline void ChatTemplateManager::UpdateInternalReflection(std::string reflection) {
    SetSection(TemplateSection::INTERNAL_REFLECTION, std::move(reflection), !reflection.empty());
}

inline void ChatTemplateManager::UpdateInternalReflection(std::string_view reflection) {
    SetSection(TemplateSection::INTERNAL_REFLECTION, reflection, !reflection.empty());
}

inline void ChatTemplateManager::UpdateEmotionalState(std::string emotional_state) {
    SetSection(TemplateSection::EMOTIONAL_STATE, std::move(emotional_state), !emotional_state.empty());
}

inline void ChatTemplateManager::UpdateEmotionalState(std::string_view emotional_state) {
    SetSection(TemplateSection::EMOTIONAL_STATE, emotional_state, !emotional_state.empty());
}

inline void ChatTemplateManager::UpdateMultipleSummaries(std::vector<std::string> summary_list) {
    summaries = std::move(summary_list);
    template_dirty = true;
}

inline void ChatTemplateManager::AddSummaryToList(std::string summary, size_t max_summaries) {
    // Fast validation check
    if (!summary.empty() && HasReservedTokens(summary)) [[unlikely]] {
        throw std::invalid_argument("AddSummaryToList: Summary content contains reserved template tokens");
    }
    
    summaries.emplace_back(std::move(summary));
    
    // Enforce maximum size
    if (summaries.size() > max_summaries) [[unlikely]] {
        summaries.erase(summaries.begin()); // Remove oldest summary
    }
    template_dirty = true;
}

inline void ChatTemplateManager::ClearAllSummaries() noexcept {
    summaries.clear();
    template_dirty = true;
}
