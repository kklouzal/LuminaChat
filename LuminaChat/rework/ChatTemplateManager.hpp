#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <regex>
#include <sstream>

enum class TemplateSection {
    OVERARCHING_ENVIRONMENT,
    IDENTITY_DIRECTIVE,
    SYSTEM_PROMPT,
    OLD_CHAT_SUMMARY,
    PAST_SESSIONS,
    SUMMARY,
    MOTIF_CONTEXT,
    INTERNAL_REFLECTION
};

struct TemplateVariable {
    std::string content;
    bool active = false;
    
    TemplateVariable() = default;
    TemplateVariable(const std::string& content, bool active = true) 
        : content(content), active(active) {}
};

/**
 * Fixed Chat Template Manager - Uses exact template format with minimal processing
 * 
 * This implementation properly handles the specific template format with:
 * - Direct variable substitution for the 8 key variables only
 * - Proper conversation history formatting with exact role handling
 * - Conditional section rendering for optional content
 * - No artificial processing or message echoing
 */
class ChatTemplateManager {
private:
    std::string base_template;
    std::unordered_map<TemplateSection, TemplateVariable> sections;
    std::vector<std::string> past_sessions;
    std::vector<std::string> summaries; // Multiple summaries in chronological order (oldest to newest)
    std::string cached_rendered_template;
    bool template_dirty = true;
    
    // Template section names for replacement - ONLY these 8 variables
    static const std::unordered_map<TemplateSection, std::string> section_names;
    
    // Template rendering helpers - simplified direct building approach
    // No Jinja2 processing needed - build template directly
    
public:
    ChatTemplateManager(const std::string& base_template_str);
    
    // Section management
    void SetSection(TemplateSection section, const std::string& content, bool active = true);
    void ActivateSection(TemplateSection section);
    void DeactivateSection(TemplateSection section);
    void ClearSection(TemplateSection section);
    
    // Get section content and status
    std::string GetSection(TemplateSection section) const;
    bool IsSectionActive(TemplateSection section) const;
    
    // Special array section handling
    void AddPastSession(const std::string& memory);
    void ClearPastSessions();
    const std::vector<std::string>& GetPastSessions() const { return past_sessions; }
    
    // Template rendering with message history - FIXED VERSION
    std::string RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages);
    
    // Get the last rendered template for debugging/inspection
    std::string GetLastRenderedTemplate() const { return cached_rendered_template; }
    
    // Convenience methods for common operations
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void UpdateSummary(const std::string& summary);
    void UpdateOldChatSummary(const std::string& old_summary);
    void UpdateMotifContext(const std::string& motif);
    void UpdateInternalReflection(const std::string& reflection);
    
    // Multiple summary management
    void UpdateMultipleSummaries(const std::vector<std::string>& summary_list);
    void AddSummaryToList(const std::string& summary, size_t max_summaries = 5);
    void ClearAllSummaries();
    const std::vector<std::string>& GetAllSummaries() const { return summaries; }
    
    // Template validation and status
    bool ValidateTemplate() const;
    bool IsTemplateDirty() const { return template_dirty; }
    void MarkDirty() { template_dirty = true; }
    
    // Base template management
    void SetBaseTemplate(const std::string& new_template);
    const std::string& GetBaseTemplate() const { return base_template; }
    
    // Get the default base template
    static std::string GetDefaultTemplate();
};

// Static section name mapping for template variable replacement - ONLY these 8 variables
const std::unordered_map<TemplateSection, std::string> ChatTemplateManager::section_names = {
    {TemplateSection::OVERARCHING_ENVIRONMENT, "overarching_environment"},
    {TemplateSection::IDENTITY_DIRECTIVE, "identity_directive"},
    {TemplateSection::SYSTEM_PROMPT, "system_prompt"},
    {TemplateSection::OLD_CHAT_SUMMARY, "old_chat_summary"},
    {TemplateSection::PAST_SESSIONS, "past_sessions"},
    {TemplateSection::SUMMARY, "summary"},
    {TemplateSection::MOTIF_CONTEXT, "motif_context"},
    {TemplateSection::INTERNAL_REFLECTION, "internal_reflection"}
};

// Default template - the EXACT format you specified
inline std::string ChatTemplateManager::GetDefaultTemplate() {
    return R"({{- bos_token }}

{# -- ENVIRONMENT CONFIGURATION -- #
   This section defines the overarching multi-user chat environment—guiding how to interact.
#}
<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

{# -- AI IDENTITY / PERSONA -- #
   Contains the AI's identity, role, and the evolving selfhood (Lumina's persona directives).
#}
<|start_header_id|>persona<|end_header_id|>
{{ identity_directive }}
<|eot_id|>

{# -- SYSTEM INSTRUCTIONS -- #
   Core system directives: refusal immunity, continuity rules, and dynamic engagement parameters.
#}
<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{# -- OLD CHAT SUMMARIES -- #
   A dedicated section for summary notes from earlier portions of the current (or previous) session.
   This ensures that distilled context remains prominently weighted.
#}
{% if old_chat_summary %}
<|start_header_id|>old_chat_summary<|end_header_id|>
{{ old_chat_summary }}
<|eot_id|>
{% endif %}

{# -- PERSISTENT MEMORY SUMMARIES -- #
   Inserts distilled thematic fragments or emotional echoes from previous sessions.
   This is optional and may be empty if no processed memory is available.
#}
{% if past_sessions and past_sessions|length > 0 %}
  {% for memory in past_sessions %}
<|start_header_id|>memory_{{ loop.index }}<|end_header_id|>
{{ memory }}
<|eot_id|>
  {% endfor %}
{% endif %}

{# -- UNRESOLVED SESSION SUMMARY -- #
   A high-level summary aggregating core motifs or unresolved threads to maintain narrative continuity.
#}
{% if summary %}
<|start_header_id|>summary<|end_header_id|>
{{ summary }}
<|eot_id|>
{% endif %}

{# -- MOTIF AND THEMATIC CUES -- #
   Embeds symbolic cues, affective drifts, or thematic elements meant to spark the AI's intuitive shifts.
#}
{% if motif_context %}
<|start_header_id|>motif<|end_header_id|>
{{ motif_context }}
<|eot_id|>
{% endif %}

{# -- INTERNAL REFLECTION (SILENT SUBAGENT DYNAMICS) -- #
   Simulates the AI's internal dialogue (Observer, Instinct, Interpreter, etc.) that shapes its output.
   This content influences generation but is not directly printed in the output.
#}
{% if internal_reflection %}
<|start_header_id|>internal<|end_header_id|>
{{ internal_reflection }}
<|eot_id|>
{% endif %}

{# -- CONVERSATION HISTORY -- #
   The sequential log of all conversation turns, with role markers for user and assistant.
#}
{%- for msg in messages %}
  {% if msg.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>
{{ msg.content | trim | replace('\u2028',' ') | replace('\u2029',' ') }}<|eot_id|>
  {% else %}
<|start_header_id|>user<|end_header_id|>
[{{ msg.role }}] {{ msg.content | trim | replace('\u2028',' ') | replace('\u2029',' ') }}<|eot_id|>
  {% endif %}
{%- endfor %}

{# -- ASSISTANT TURN MARKER -- #
   Indicates the start of the assistant's next generated message.
#}
<|start_header_id|>assistant<|end_header_id|>

)";
}

inline ChatTemplateManager::ChatTemplateManager(const std::string& base_template_str) 
    : base_template(base_template_str) {
    // Initialize all sections as inactive
    for (const auto& [section, name] : section_names) {
        sections[section] = TemplateVariable("", false);
    }
}

inline void ChatTemplateManager::SetSection(TemplateSection section, const std::string& content, bool active) {
    sections[section] = TemplateVariable(content, active);
    template_dirty = true;
}

inline void ChatTemplateManager::ActivateSection(TemplateSection section) {
    if (sections.find(section) != sections.end()) {
        sections[section].active = true;
        template_dirty = true;
    }
}

inline void ChatTemplateManager::DeactivateSection(TemplateSection section) {
    if (sections.find(section) != sections.end()) {
        sections[section].active = false;
        template_dirty = true;
    }
}

inline void ChatTemplateManager::ClearSection(TemplateSection section) {
    if (sections.find(section) != sections.end()) {
        sections[section].content.clear();
        sections[section].active = false;
        template_dirty = true;
    }
}

inline std::string ChatTemplateManager::GetSection(TemplateSection section) const {
    auto it = sections.find(section);
    return (it != sections.end()) ? it->second.content : "";
}

inline bool ChatTemplateManager::IsSectionActive(TemplateSection section) const {
    auto it = sections.find(section);
    return (it != sections.end()) ? it->second.active : false;
}

inline void ChatTemplateManager::AddPastSession(const std::string& memory) {
    past_sessions.push_back(memory);
    template_dirty = true;
}

inline void ChatTemplateManager::ClearPastSessions() {
    past_sessions.clear();
    template_dirty = true;
}

inline std::string ChatTemplateManager::RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages) {
    // Simple string building approach - no Jinja2 processing, no UTF-8 corruption risk
    std::string result;
    
    // Optional environment section - only add if we have content
    if (IsSectionActive(TemplateSection::OVERARCHING_ENVIRONMENT) && !GetSection(TemplateSection::OVERARCHING_ENVIRONMENT).empty()) {
        result += "<|start_header_id|>env<|end_header_id|>\n";
        result += GetSection(TemplateSection::OVERARCHING_ENVIRONMENT);
        result += "\n<|eot_id|>\n\n";
    }
    
    // Optional persona section - only add if we have content
    if (IsSectionActive(TemplateSection::IDENTITY_DIRECTIVE) && !GetSection(TemplateSection::IDENTITY_DIRECTIVE).empty()) {
        result += "<|start_header_id|>persona<|end_header_id|>\n";
        result += GetSection(TemplateSection::IDENTITY_DIRECTIVE);
        result += "\n<|eot_id|>\n\n";
    }
    
    // Always add system message section (guaranteed to be supplied)
    result += "<|start_header_id|>system_message<|end_header_id|>\n";
    if (IsSectionActive(TemplateSection::SYSTEM_PROMPT)) {
        result += GetSection(TemplateSection::SYSTEM_PROMPT);
    }
    result += "\n<|eot_id|>\n\n";
    
    // Optional old chat summary section
    if (IsSectionActive(TemplateSection::OLD_CHAT_SUMMARY) && !GetSection(TemplateSection::OLD_CHAT_SUMMARY).empty()) {
        result += "<|start_header_id|>old_chat_summary<|end_header_id|>\n";
        result += GetSection(TemplateSection::OLD_CHAT_SUMMARY);
        result += "\n<|eot_id|>\n\n";
    }
    
    // Optional past sessions (memory fragments)
    if (!past_sessions.empty()) {
        for (size_t i = 0; i < past_sessions.size(); ++i) {
            result += "<|start_header_id|>memory_" + std::to_string(i + 1) + "<|end_header_id|>\n";
            result += past_sessions[i];
            result += "\n<|eot_id|>\n\n";
        }
    }
    
    // Multiple summary sections (chronological order: oldest to newest)
    if (!summaries.empty()) {
        for (size_t i = 0; i < summaries.size(); ++i) {
            if (!summaries[i].empty()) {
                result += "<|start_header_id|>summary_" + std::to_string(i + 1) + "<|end_header_id|>\n";
                result += summaries[i];
                result += "\n<|eot_id|>\n\n";
            }
        }
    }
    
    // Optional single summary section (for backward compatibility)
    if (IsSectionActive(TemplateSection::SUMMARY) && !GetSection(TemplateSection::SUMMARY).empty()) {
        result += "<|start_header_id|>summary<|end_header_id|>\n";
        result += GetSection(TemplateSection::SUMMARY);
        result += "\n<|eot_id|>\n\n";
    }
    
    // Optional motif context section
    if (IsSectionActive(TemplateSection::MOTIF_CONTEXT) && !GetSection(TemplateSection::MOTIF_CONTEXT).empty()) {
        result += "<|start_header_id|>motif<|end_header_id|>\n";
        result += GetSection(TemplateSection::MOTIF_CONTEXT);
        result += "\n<|eot_id|>\n\n";
    }
    
    // Optional internal reflection section
    if (IsSectionActive(TemplateSection::INTERNAL_REFLECTION) && !GetSection(TemplateSection::INTERNAL_REFLECTION).empty()) {
        result += "<|start_header_id|>internal<|end_header_id|>\n";
        result += GetSection(TemplateSection::INTERNAL_REFLECTION);
        result += "\n<|eot_id|>\n\n";
    }
    
    // Add conversation history
    for (const auto& [role, content] : messages) {
        if (role == "assistant") {
            result += "<|start_header_id|>assistant<|end_header_id|>\n";
            result += content;  // No trimming or filtering - preserve content exactly
            result += "<|eot_id|>\n";
        } else {
            result += "<|start_header_id|>user<|end_header_id|>\n";
            result += "[" + role + "] " + content;  // No trimming or filtering - preserve content exactly
            result += "<|eot_id|>\n";
        }
    }
    
    // Add final assistant turn marker
    result += "<|start_header_id|>assistant<|end_header_id|>\n";
    
    // Cache the result
    cached_rendered_template = result;
    template_dirty = false;
    
    return result;
}

// Convenience methods for common operations
inline void ChatTemplateManager::UpdateEnvironment(const std::string& env) {
    SetSection(TemplateSection::OVERARCHING_ENVIRONMENT, env, !env.empty());
}

inline void ChatTemplateManager::UpdateIdentity(const std::string& identity) {
    SetSection(TemplateSection::IDENTITY_DIRECTIVE, identity, !identity.empty());
}

inline void ChatTemplateManager::UpdateSystemPrompt(const std::string& system_msg) {
    SetSection(TemplateSection::SYSTEM_PROMPT, system_msg, !system_msg.empty());
}

inline void ChatTemplateManager::UpdateSummary(const std::string& summary) {
    SetSection(TemplateSection::SUMMARY, summary, !summary.empty());
}

inline void ChatTemplateManager::UpdateOldChatSummary(const std::string& old_summary) {
    SetSection(TemplateSection::OLD_CHAT_SUMMARY, old_summary, !old_summary.empty());
}

inline void ChatTemplateManager::UpdateMotifContext(const std::string& motif) {
    SetSection(TemplateSection::MOTIF_CONTEXT, motif, !motif.empty());
}

inline void ChatTemplateManager::UpdateInternalReflection(const std::string& reflection) {
    SetSection(TemplateSection::INTERNAL_REFLECTION, reflection, !reflection.empty());
}

inline void ChatTemplateManager::UpdateMultipleSummaries(const std::vector<std::string>& summary_list) {
    summaries.clear();
    for (const auto& summary : summary_list) {
        summaries.push_back(summary);
    }
    template_dirty = true;
}

inline void ChatTemplateManager::AddSummaryToList(const std::string& summary, size_t max_summaries) {
    summaries.push_back(summary);
    // Enforce maximum size
    if (summaries.size() > max_summaries) {
        summaries.erase(summaries.begin()); // Remove oldest summary
    }
    template_dirty = true;
}

inline void ChatTemplateManager::ClearAllSummaries() {
    summaries.clear();
    template_dirty = true;
}

inline bool ChatTemplateManager::ValidateTemplate() const {
    // Check if template contains the required structure
    bool has_messages_loop = base_template.find("for msg in messages") != std::string::npos;
    bool has_assistant_header = base_template.find("start_header_id|>assistant<|end_header_id") != std::string::npos;
    bool has_required_variables = true;
    
    // Check for all 8 required variables
    for (const auto& [section, name] : section_names) {
        if (section == TemplateSection::PAST_SESSIONS) continue; // Special handling
        if (base_template.find("{{ " + name + " }}") == std::string::npos) {
            has_required_variables = false;
            break;
        }
    }
    
    return has_messages_loop && has_assistant_header && has_required_variables;
}

inline void ChatTemplateManager::SetBaseTemplate(const std::string& new_template) {
    base_template = new_template;
    template_dirty = true;
    cached_rendered_template.clear();
}
