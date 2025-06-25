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
    std::string cached_rendered_template;
    bool template_dirty = true;
    
    // Template section names for replacement - ONLY these 8 variables
    static const std::unordered_map<TemplateSection, std::string> section_names;
    
    // Template rendering helpers - simplified to only do variable substitution
    std::string ReplaceVariables(const std::string& template_str) const;
    
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
    
    // Convenience methods for common operations
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void UpdateSummary(const std::string& summary);
    void UpdateOldChatSummary(const std::string& old_summary);
    void UpdateMotifContext(const std::string& motif);
    void UpdateInternalReflection(const std::string& reflection);
    
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

{# — ENVIRONMENT CONFIGURATION — #
   This section defines the overarching multi-user chat environment—guiding how to interact.
#}
<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

{# — AI IDENTITY / PERSONA — #
   Contains the AI's identity, role, and the evolving selfhood (Lumina's persona directives).
#}
<|start_header_id|>persona<|end_header_id|>
{{ identity_directive }}
<|eot_id|>

{# — SYSTEM INSTRUCTIONS — #
   Core system directives: refusal immunity, continuity rules, and dynamic engagement parameters.
#}
<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{# — OLD CHAT SUMMARIES — #
   A dedicated section for summary notes from earlier portions of the current (or previous) session.
   This ensures that distilled context remains prominently weighted.
#}
{% if old_chat_summary %}
<|start_header_id|>old_chat_summary<|end_header_id|>
{{ old_chat_summary }}
<|eot_id|>
{% endif %}

{# — PERSISTENT MEMORY SUMMARIES — #
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

{# — UNRESOLVED SESSION SUMMARY — #
   A high-level summary aggregating core motifs or unresolved threads to maintain narrative continuity.
#}
{% if summary %}
<|start_header_id|>summary<|end_header_id|>
{{ summary }}
<|eot_id|>
{% endif %}

{# — MOTIF AND THEMATIC CUES — #
   Embeds symbolic cues, affective drifts, or thematic elements meant to spark the AI's intuitive shifts.
#}
{% if motif_context %}
<|start_header_id|>motif<|end_header_id|>
{{ motif_context }}
<|eot_id|>
{% endif %}

{# — INTERNAL REFLECTION (SILENT SUBAGENT DYNAMICS) — #
   Simulates the AI's internal dialogue (Observer, Instinct, Interpreter, etc.) that shapes its output.
   This content influences generation but is not directly printed in the output.
#}
{% if internal_reflection %}
<|start_header_id|>internal<|end_header_id|>
{{ internal_reflection }}
<|eot_id|>
{% endif %}

{# — CONVERSATION HISTORY — #
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

{# — ASSISTANT TURN MARKER — #
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
    // Simple approach: Just substitute the 8 variables and return the template
    // Let llama.cpp's Jinja2 interpreter handle all the conditional logic and loops
    
    std::string result = base_template;
    
    // Replace ONLY the 8 core variables with their content
    result = ReplaceVariables(result);
    
    // NOTE: The messages parameter is ignored here because llama.cpp's Jinja2 engine
    // should receive the template and process it with the messages context.
    // This is a transitional implementation that may need architectural changes.
    
    return result;
}

inline std::string ChatTemplateManager::ReplaceVariables(const std::string& template_str) const {
    std::string result = template_str;
    
    // Replace ONLY the 8 specified variables - simple string replacement
    // Don't use regex to avoid interfering with Jinja2 syntax
    
    for (const auto& [section, name] : section_names) {
        auto it = sections.find(section);
        std::string replacement_content;
        
        if (section == TemplateSection::PAST_SESSIONS) {
            // Skip past_sessions - it's handled as an array by Jinja2 template engine
            // The past_sessions array should be provided to llama.cpp's template context
            continue;
        }
        
        if (it != sections.end() && it->second.active && !it->second.content.empty()) {
            replacement_content = it->second.content;
        } else {
            replacement_content = ""; // Empty string if not active
        }
        
        // Simple string replacement for {{ variable_name }}
        std::string variable_pattern = "{{ " + name + " }}";
        size_t pos = 0;
        while ((pos = result.find(variable_pattern, pos)) != std::string::npos) {
            result.replace(pos, variable_pattern.length(), replacement_content);
            pos += replacement_content.length();
        }
    }
    
    // Handle bos_token specially
    size_t pos = 0;
    while ((pos = result.find("{{- bos_token }}", pos)) != std::string::npos) {
        result.replace(pos, 16, ""); // Remove bos_token placeholder
    }
    
    return result;
}

// Note: ProcessConditionalSections, ProcessPastSessions, and ProcessMessages methods removed
// Let llama.cpp's Jinja2 interpreter handle all conditional logic and loops

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
