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
    
    // Template rendering helpers - full Jinja2-style processing
    std::string ReplaceVariables(const std::string& template_str) const;
    std::string ProcessConditionals(const std::string& template_str) const;
    std::string ProcessLoops(const std::string& template_str, const std::vector<std::pair<std::string, std::string>>& messages) const;
    std::string ProcessFilters(const std::string& text) const;
    std::string EscapeContent(const std::string& content) const;
    
    // Helper methods for Jinja2 processing
    bool EvaluateCondition(const std::string& condition) const;
    std::string ApplyFilter(const std::string& content, const std::string& filter) const;
    
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
    // Full Jinja2-style template processing
    std::string result = base_template;
    
    // Step 1: Replace our 8 variables first
    result = ReplaceVariables(result);
    
    // Step 2: Process loops with messages (must be done before conditionals)
    result = ProcessLoops(result, messages);
    
    // Step 3: Process conditionals ({% if %}, {% endif %}) - do this after loops
    result = ProcessConditionals(result);
    
    // Step 4: Remove any remaining template syntax and comments
    // Remove {# comments #}
    size_t comment_start = 0;
    while ((comment_start = result.find("{#", comment_start)) != std::string::npos) {
        size_t comment_end = result.find("#}", comment_start);
        if (comment_end != std::string::npos) {
            result.erase(comment_start, comment_end - comment_start + 2);
        } else {
            break;
        }
    }
    
    // Step 5: Handle special tokens
    // Replace {{- bos_token }} with empty string (BOS is handled by tokenizer)
    size_t pos = 0;
    while ((pos = result.find("{{- bos_token }}", pos)) != std::string::npos) {
        result.replace(pos, 16, "");
    }
    while ((pos = result.find("{{ bos_token }}", pos)) != std::string::npos) {
        result.replace(pos, 15, "");
    }
    
    // Step 6: Clean up any remaining Jinja2 syntax that we might have missed
    // Remove any remaining {% ... %} blocks (safety net)
    std::regex remaining_blocks(R"(\{%[^%]*%\})");
    result = std::regex_replace(result, remaining_blocks, "");
    
    // Remove any remaining {{ ... }} variables (except those we know about)
    std::regex remaining_vars(R"(\{\{[^}]*\}\})");
    result = std::regex_replace(result, remaining_vars, "");
    
    // Step 7: Clean up whitespace from template processing
    // Remove excessive newlines but preserve intentional spacing
    std::string cleaned_result;
    bool prev_newline = false;
    int consecutive_newlines = 0;
    
    for (char c : result) {
        if (c == '\n') {
            consecutive_newlines++;
            if (consecutive_newlines <= 2) {  // Allow max 2 consecutive newlines
                cleaned_result += c;
            }
            prev_newline = true;
        } else if (c == ' ' || c == '\t') {
            if (!prev_newline || !cleaned_result.empty()) {
                cleaned_result += c;
            }
        } else {
            cleaned_result += c;
            prev_newline = false;
            consecutive_newlines = 0;
        }
    }
    
    // Cache the result
    cached_rendered_template = cleaned_result;
    template_dirty = false;
    
    return cleaned_result;
}

inline std::string ChatTemplateManager::ReplaceVariables(const std::string& template_str) const {
    std::string result = template_str;
    
    // Replace ONLY the 8 specified variables - simple string replacement
    // Don't touch ANY other Jinja2 syntax including {{- bos_token }}, whitespace controls, etc.
    
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
        
        // Simple string replacement for {{ variable_name }} ONLY
        // Be very careful to only match our exact 8 variables
        std::string variable_pattern = "{{ " + name + " }}";
        size_t pos = 0;
        while ((pos = result.find(variable_pattern, pos)) != std::string::npos) {
            result.replace(pos, variable_pattern.length(), replacement_content);
            pos += replacement_content.length();
        }
    }
    
    // DO NOT touch {{- bos_token }} or any other Jinja2 syntax!
    // Leave ALL other template syntax for llama.cpp's Jinja2 interpreter
    
    return result;
}

inline std::string ChatTemplateManager::ProcessConditionals(const std::string& template_str) const {
    std::string result = template_str;
    
    // Process {% if condition %} ... {% endif %} blocks with multiline support
    // We'll use a simpler approach by replacing newlines temporarily
    std::string working = result;
    
    // Replace newlines with a placeholder to make regex work across lines
    const std::string newline_placeholder = "___NEWLINE_PLACEHOLDER___";
    size_t pos = 0;
    while ((pos = working.find('\n', pos)) != std::string::npos) {
        working.replace(pos, 1, newline_placeholder);
        pos += newline_placeholder.length();
    }
    
    // Now process conditionals with the modified string
    std::regex if_pattern(R"(\{%\s*if\s+([^%}]+)\s*%\}(.*?)\{%\s*endif\s*%\})");
    
    // Keep processing until no more matches (handle nested or multiple conditionals)
    bool found_match = true;
    while (found_match) {
        found_match = false;
        std::sregex_iterator iter(working.begin(), working.end(), if_pattern);
        std::sregex_iterator end;
        
        std::vector<std::pair<size_t, size_t>> replacements;
        std::vector<std::string> replacement_texts;
        
        for (auto i = iter; i != end; ++i) {
            std::smatch match = *i;
            std::string condition = match[1].str();
            std::string content = match[2].str();
            
            // Evaluate the condition
            bool condition_result = EvaluateCondition(condition);
            
            if (condition_result) {
                replacement_texts.push_back(content);
            } else {
                replacement_texts.push_back("");
            }
            
            replacements.push_back({match.position(), match.length()});
            found_match = true;
        }
        
        // Apply replacements in reverse order to maintain positions
        for (int i = replacements.size() - 1; i >= 0; --i) {
            working.replace(replacements[i].first, replacements[i].second, replacement_texts[i]);
        }
    }
    
    // Restore newlines
    pos = 0;
    while ((pos = working.find(newline_placeholder, pos)) != std::string::npos) {
        working.replace(pos, newline_placeholder.length(), "\n");
        pos += 1;
    }
    
    return working;
}

inline std::string ChatTemplateManager::ProcessLoops(const std::string& template_str, const std::vector<std::pair<std::string, std::string>>& messages) const {
    std::string result = template_str;
    
    // Replace newlines with placeholder for multiline regex matching
    const std::string newline_placeholder = "___NEWLINE_PLACEHOLDER___";
    std::string working = result;
    size_t pos = 0;
    while ((pos = working.find('\n', pos)) != std::string::npos) {
        working.replace(pos, 1, newline_placeholder);
        pos += newline_placeholder.length();
    }
    
    // First handle past_sessions loop
    std::regex past_sessions_pattern(R"(\{%\s*for\s+memory\s+in\s+past_sessions\s*%\}(.*?)\{%\s*endfor\s*%\})");
    std::smatch past_match;
    
    if (std::regex_search(working, past_match, past_sessions_pattern)) {
        std::string loop_content = past_match[1].str();
        std::string expanded_content;
        
        for (size_t i = 0; i < past_sessions.size(); ++i) {
            std::string iteration = loop_content;
            
            // Replace {{ memory }} with actual content
            size_t replace_pos = 0;
            while ((replace_pos = iteration.find("{{ memory }}", replace_pos)) != std::string::npos) {
                iteration.replace(replace_pos, 12, past_sessions[i]);
                replace_pos += past_sessions[i].length();
            }
            
            // Replace {{ loop.index }} with 1-based index
            replace_pos = 0;
            while ((replace_pos = iteration.find("{{ loop.index }}", replace_pos)) != std::string::npos) {
                iteration.replace(replace_pos, 16, std::to_string(i + 1));
                replace_pos += std::to_string(i + 1).length();
            }
            
            expanded_content += iteration;
        }
        
        working.replace(past_match.position(), past_match.length(), expanded_content);
    }
    
    // Handle messages loop with whitespace control
    std::regex messages_pattern(R"(\{%-?\s*for\s+msg\s+in\s+messages\s*-?%\}(.*?)\{%-?\s*endfor\s*-?%\})");
    std::smatch msg_match;
    
    if (std::regex_search(working, msg_match, messages_pattern)) {
        std::string loop_content = msg_match[1].str();
        std::string expanded_content;
        
        for (const auto& [role, content] : messages) {
            std::string iteration = loop_content;
            
            // Process conditional blocks within the loop ({% if msg.role == "assistant" %})
            std::regex role_condition(R"(\{%\s*if\s+msg\.role\s*==\s*["\']([^"\']+)["\']\s*%\}(.*?)(?:\{%\s*else\s*%\}(.*?))?\{%\s*endif\s*%\})");
            
            // Keep processing until no more role conditionals
            bool found_role_match = true;
            while (found_role_match) {
                found_role_match = false;
                std::sregex_iterator role_iter(iteration.begin(), iteration.end(), role_condition);
                std::sregex_iterator role_end;
                
                std::vector<std::pair<size_t, size_t>> role_replacements;
                std::vector<std::string> role_replacement_texts;
                
                for (auto i = role_iter; i != role_end; ++i) {
                    std::smatch role_match = *i;
                    std::string target_role = role_match[1].str();
                    std::string if_content = role_match[2].str();
                    std::string else_content = role_match[3].str();
                    
                    if (role == target_role) {
                        role_replacement_texts.push_back(if_content);
                    } else {
                        role_replacement_texts.push_back(else_content);
                    }
                    
                    role_replacements.push_back({role_match.position(), role_match.length()});
                    found_role_match = true;
                }
                
                // Apply role conditional replacements in reverse order
                for (int i = role_replacements.size() - 1; i >= 0; --i) {
                    iteration.replace(role_replacements[i].first, role_replacements[i].second, role_replacement_texts[i]);
                }
            }
            
            // Replace {{ msg.role }} and {{ msg.content }}
            size_t replace_pos = 0;
            while ((replace_pos = iteration.find("{{ msg.role }}", replace_pos)) != std::string::npos) {
                iteration.replace(replace_pos, 14, role);
                replace_pos += role.length();
            }
            
            // Handle {{ msg.content | trim | replace('\u2028',' ') | replace('\u2029',' ') }}
            replace_pos = 0;
            while ((replace_pos = iteration.find("{{ msg.content", replace_pos)) != std::string::npos) {
                size_t end_pos = iteration.find("}}", replace_pos);
                if (end_pos != std::string::npos) {
                    std::string filter_expr = iteration.substr(replace_pos, end_pos - replace_pos + 2);
                    std::string processed_content = ApplyFilter(content, filter_expr);
                    iteration.replace(replace_pos, end_pos - replace_pos + 2, processed_content);
                    replace_pos += processed_content.length();
                } else {
                    break;
                }
            }
            
            expanded_content += iteration;
        }
        
        working.replace(msg_match.position(), msg_match.length(), expanded_content);
    }
    
    // Restore newlines
    pos = 0;
    while ((pos = working.find(newline_placeholder, pos)) != std::string::npos) {
        working.replace(pos, newline_placeholder.length(), "\n");
        pos += 1;
    }
    
    return working;
}

inline bool ChatTemplateManager::EvaluateCondition(const std::string& condition) const {
    std::string trimmed = condition;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    trimmed.erase(trimmed.find_last_not_of(" \t") + 1);
    
    // Handle different condition types
    if (trimmed == "old_chat_summary") {
        return IsSectionActive(TemplateSection::OLD_CHAT_SUMMARY) && 
               !GetSection(TemplateSection::OLD_CHAT_SUMMARY).empty();
    }
    else if (trimmed == "summary") {
        return IsSectionActive(TemplateSection::SUMMARY) && 
               !GetSection(TemplateSection::SUMMARY).empty();
    }
    else if (trimmed == "motif_context") {
        return IsSectionActive(TemplateSection::MOTIF_CONTEXT) && 
               !GetSection(TemplateSection::MOTIF_CONTEXT).empty();
    }
    else if (trimmed == "internal_reflection") {
        return IsSectionActive(TemplateSection::INTERNAL_REFLECTION) && 
               !GetSection(TemplateSection::INTERNAL_REFLECTION).empty();
    }
    else if (trimmed == "past_sessions and past_sessions|length > 0") {
        return !past_sessions.empty();
    }
    
    return false;
}

inline std::string ChatTemplateManager::ApplyFilter(const std::string& content, const std::string& filter_expr) const {
    std::string result = content;
    
    // Apply trim filter
    if (filter_expr.find("trim") != std::string::npos) {
        result.erase(0, result.find_first_not_of(" \t\n\r"));
        result.erase(result.find_last_not_of(" \t\n\r") + 1);
    }
    
    // Apply replace filters
    if (filter_expr.find("replace('\\u2028',' ')") != std::string::npos) {
        // Replace Unicode Line Separator (U+2028)
        std::string line_sep;
        line_sep += static_cast<char>(0xe2);
        line_sep += static_cast<char>(0x80);
        line_sep += static_cast<char>(0xa8);
        size_t pos = 0;
        while ((pos = result.find(line_sep, pos)) != std::string::npos) {
            result.replace(pos, line_sep.length(), " ");
            pos += 1;
        }
    }
    
    if (filter_expr.find("replace('\\u2029',' ')") != std::string::npos) {
        // Replace Unicode Paragraph Separator (U+2029)
        std::string para_sep;
        para_sep += static_cast<char>(0xe2);
        para_sep += static_cast<char>(0x80);
        para_sep += static_cast<char>(0xa9);
        size_t pos = 0;
        while ((pos = result.find(para_sep, pos)) != std::string::npos) {
            result.replace(pos, para_sep.length(), " ");
            pos += 1;
        }
    }
    
    return result;
}

inline std::string ChatTemplateManager::ProcessFilters(const std::string& text) const {
    // This method can be used for additional filter processing if needed
    // Currently, filters are handled inline in ProcessLoops for efficiency
    return text;
}

inline std::string ChatTemplateManager::EscapeContent(const std::string& content) const {
    // Escape special characters for safe template processing
    std::string result = content;
    
    // Escape characters that might interfere with template syntax
    size_t pos = 0;
    while ((pos = result.find("{{", pos)) != std::string::npos) {
        result.replace(pos, 2, "&#123;&#123;");
        pos += 14;
    }
    
    pos = 0;
    while ((pos = result.find("}}", pos)) != std::string::npos) {
        result.replace(pos, 2, "&#125;&#125;");
        pos += 14;
    }
    
    pos = 0;
    while ((pos = result.find("{%", pos)) != std::string::npos) {
        result.replace(pos, 2, "&#123;&#37;");
        pos += 14;
    }
    
    pos = 0;
    while ((pos = result.find("%}", pos)) != std::string::npos) {
        result.replace(pos, 2, "&#37;&#125;");
        pos += 14;
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
