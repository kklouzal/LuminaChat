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
 * Dynamic, sophisticated chat template management with Jinja2-style rendering
 * 
 * Revolutionary Change: Templates are now per-context and dynamically modified during runtime.
 * System messages, summaries, and contextual information are embedded directly into template 
 * sections rather than polluting message history.
 * 
 * Key Features:
 * - Per-Context Templates: Each ContextInfo has its own template manager
 * - Section-Based Management: Individual control over template components
 * - Memory Integration: Past session memories as array-based template section
 * - Summary Integration: Summarization plugin updates template sections directly
 * - Performance Caching: Template rendering cached until sections change
 */
class ChatTemplateManager {
private:
    std::string base_template;
    std::unordered_map<TemplateSection, TemplateVariable> sections;
    std::vector<std::string> past_sessions;
    std::string cached_rendered_template;
    bool template_dirty = true;
    
    // Template section names for replacement
    static const std::unordered_map<TemplateSection, std::string> section_names;
    
    // Template validation and rendering helpers
    std::string ProcessConditionalSections(const std::string& template_str) const;
    std::string ProcessArraySections(const std::string& template_str) const;
    std::string ReplaceVariables(const std::string& template_str) const;
    std::string ProcessMessages(const std::string& template_str, 
                              const std::vector<std::pair<std::string, std::string>>& messages) const;
    
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
    
    // Template rendering with message history
    std::string RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages);
    
    // Convenience methods for common operations
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void UpdateSummary(const std::string& summary);  // Plugin integration point
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
};

// Static section name mapping for template variable replacement
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

ChatTemplateManager::ChatTemplateManager(const std::string& base_template_str) 
    : base_template(base_template_str) {
    // Initialize all sections as inactive
    for (const auto& [section, name] : section_names) {
        sections[section] = TemplateVariable("", false);
    }
}

void ChatTemplateManager::SetSection(TemplateSection section, const std::string& content, bool active) {
    sections[section] = TemplateVariable(content, active);
    template_dirty = true;
}

void ChatTemplateManager::ActivateSection(TemplateSection section) {
    if (sections.find(section) != sections.end()) {
        sections[section].active = true;
        template_dirty = true;
    }
}

void ChatTemplateManager::DeactivateSection(TemplateSection section) {
    if (sections.find(section) != sections.end()) {
        sections[section].active = false;
        template_dirty = true;
    }
}

void ChatTemplateManager::ClearSection(TemplateSection section) {
    if (sections.find(section) != sections.end()) {
        sections[section].content.clear();
        sections[section].active = false;
        template_dirty = true;
    }
}

std::string ChatTemplateManager::GetSection(TemplateSection section) const {
    auto it = sections.find(section);
    return (it != sections.end()) ? it->second.content : "";
}

bool ChatTemplateManager::IsSectionActive(TemplateSection section) const {
    auto it = sections.find(section);
    return (it != sections.end()) ? it->second.active : false;
}

void ChatTemplateManager::AddPastSession(const std::string& memory) {
    past_sessions.push_back(memory);
    template_dirty = true;
}

void ChatTemplateManager::ClearPastSessions() {
    past_sessions.clear();
    template_dirty = true;
}

std::string ChatTemplateManager::RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages) {
    if (!template_dirty && !cached_rendered_template.empty()) {
        // Template hasn't changed, return cached version with updated messages
        return ProcessMessages(cached_rendered_template, messages);
    }
    
    // Process template sections in order
    std::string processed_template = base_template;
    
    // Replace variable sections
    processed_template = ReplaceVariables(processed_template);
    
    // Process conditional sections ({% if section %})
    processed_template = ProcessConditionalSections(processed_template);
    
    // Process array sections (past_sessions)
    processed_template = ProcessArraySections(processed_template);
    
    // Cache the processed template (without messages)
    cached_rendered_template = processed_template;
    template_dirty = false;
    
    // Process messages at the end
    return ProcessMessages(processed_template, messages);
}

std::string ChatTemplateManager::ProcessConditionalSections(const std::string& template_str) const {
    std::string result = template_str;
    
    // Process each conditional section
    for (const auto& [section, name] : section_names) {
        if (section == TemplateSection::PAST_SESSIONS) continue; // Handle separately
        
        // Create regex pattern for {% if section_name %}...{% endif %} - handle multiline and whitespace
        std::string pattern = R"(\{\%\s*if\s+)" + name + R"(\s*\%\}([\s\S]*?)\{\%\s*endif\s*\%\})";
        std::regex conditional_regex(pattern, std::regex_constants::ECMAScript);
        
        auto it = sections.find(section);
        bool section_active = (it != sections.end()) && it->second.active && !it->second.content.empty();
        
        if (section_active) {
            // Replace conditional block with its content
            result = std::regex_replace(result, conditional_regex, "$1");
        } else {
            // Remove entire conditional block
            result = std::regex_replace(result, conditional_regex, "");
        }
    }
    
    return result;
}

std::string ChatTemplateManager::ProcessArraySections(const std::string& template_str) const {
    std::string result = template_str;
    
    // First handle simple {% for memory in past_sessions %} loops
    std::string simple_pattern = R"(\{\%\s*for\s+memory\s+in\s+past_sessions\s*\%\}([\s\S]*?)\{\%\s*endfor\s*\%\})";
    std::regex simple_regex(simple_pattern, std::regex_constants::ECMAScript);
    
    std::smatch simple_match;
    if (std::regex_search(result, simple_match, simple_regex)) {
        if (!past_sessions.empty()) {
            std::string loop_content;
            std::string loop_template = simple_match[1].str();
            
            for (const auto& memory : past_sessions) {
                std::string memory_instance = loop_template;
                // Replace {{ memory }} with actual memory content
                memory_instance = std::regex_replace(memory_instance, std::regex(R"(\{\{\s*memory\s*\}\})"), memory);
                loop_content += memory_instance;
            }
            
            result = std::regex_replace(result, simple_regex, loop_content);
        } else {
            // Remove the entire loop if no past sessions
            result = std::regex_replace(result, simple_regex, "");
        }
    }
    
    // Then handle complex conditional patterns with if past_sessions 
    std::string pattern = R"(\{\%\s*if\s+past_sessions\s+and\s+past_sessions\|length\s*>\s*0\s*\%\}([\s\S]*?)\{\%\s*endif\s*\%\})";
    std::regex array_regex(pattern, std::regex_constants::ECMAScript);
    
    std::smatch match;
    if (std::regex_search(result, match, array_regex)) {
        if (!past_sessions.empty()) {
            std::string inner_content = match[1].str();
            
            // Process the for loop within the conditional
            std::string for_pattern = R"(\{\%\s*for\s+memory\s+in\s+past_sessions\s*\%\}([\s\S]*?)\{\%\s*endfor\s*\%\})";
            std::regex for_regex(for_pattern, std::regex_constants::ECMAScript);
            
            std::string loop_content;
            std::smatch for_match;
            if (std::regex_search(inner_content, for_match, for_regex)) {
                std::string loop_template = for_match[1].str();
                
                for (size_t i = 0; i < past_sessions.size(); ++i) {
                    std::string memory_instance = loop_template;
                    // Replace {{ memory }} with actual memory content
                    memory_instance = std::regex_replace(memory_instance, std::regex(R"(\{\{\s*memory\s*\}\})"), past_sessions[i]);
                    // Replace {{ loop.index }} with actual index
                    memory_instance = std::regex_replace(memory_instance, std::regex(R"(\{\{\s*loop\.index\s*\}\})"), std::to_string(i + 1));
                    loop_content += memory_instance;
                }
            }
            
            // Replace the entire conditional array block with the generated content
            result = std::regex_replace(result, array_regex, loop_content);
        } else {
            // Remove the entire conditional array block if no past sessions
            result = std::regex_replace(result, array_regex, "");
        }
    }
    
    return result;
}

std::string ChatTemplateManager::ReplaceVariables(const std::string& template_str) const {
    std::string result = template_str;
    
    // Replace {{ variable_name }} with actual content
    for (const auto& [section, name] : section_names) {
        auto it = sections.find(section);
        if (it != sections.end() && it->second.active) {
            std::string pattern = R"(\{\{\s*)" + name + R"(\s*\}\})";
            std::regex var_regex(pattern);
            result = std::regex_replace(result, var_regex, it->second.content);
        } else {
            // Replace with empty string if section is not active
            std::string pattern = R"(\{\{\s*)" + name + R"(\s*\}\})";
            std::regex var_regex(pattern);
            result = std::regex_replace(result, var_regex, "");
        }
    }
    
    return result;
}

std::string ChatTemplateManager::ProcessMessages(const std::string& template_str, 
                                                const std::vector<std::pair<std::string, std::string>>& messages) const {
    std::string result = template_str;
    
    // Find the messages loop: {%- for msg in messages %}...{%- endfor %} - handle multiline
    std::string pattern = R"(\{\%-?\s*for\s+msg\s+in\s+messages\s*-?\%\}([\s\S]*?)\{\%-?\s*endfor\s*-?\%\})";
    std::regex messages_regex(pattern, std::regex_constants::ECMAScript);
    
    std::smatch match;
    if (std::regex_search(result, match, messages_regex)) {
        std::string loop_template = match[1].str();
        std::string messages_content;
        
        for (const auto& [role, content] : messages) {
            std::string message_block = loop_template;
            
            // Replace {{ msg.role }} and {{ msg.content }}
            std::regex role_regex(R"(\{\{\s*msg\.role\s*\}\})");
            std::regex content_regex(R"(\{\{\s*msg\.content\s*\|\s*trim\s*\}\})");
            
            message_block = std::regex_replace(message_block, role_regex, role);
            message_block = std::regex_replace(message_block, content_regex, content);
            
            // Process conditional role blocks - handle multiline
            std::regex if_assistant_regex(R"(\{\%\s*if\s+msg\.role\s*==\s*["\']assistant["\']\s*\%\}([\s\S]*?)\{\%\s*else\s*\%\}([\s\S]*?)\{\%\s*endif\s*\%\})");
            
            if (role == "assistant") {
                message_block = std::regex_replace(message_block, if_assistant_regex, "$1");
            } else {
                message_block = std::regex_replace(message_block, if_assistant_regex, "$2");
            }
            
            messages_content += message_block;
        }
        
        // Replace the entire messages loop with the generated content
        result = std::regex_replace(result, messages_regex, messages_content);
    }
    
    return result;
}

// Convenience methods for common operations
void ChatTemplateManager::UpdateEnvironment(const std::string& env) {
    SetSection(TemplateSection::OVERARCHING_ENVIRONMENT, env, !env.empty());
}

void ChatTemplateManager::UpdateIdentity(const std::string& identity) {
    SetSection(TemplateSection::IDENTITY_DIRECTIVE, identity, !identity.empty());
}

void ChatTemplateManager::UpdateSystemPrompt(const std::string& system_msg) {
    SetSection(TemplateSection::SYSTEM_PROMPT, system_msg, !system_msg.empty());
}

void ChatTemplateManager::UpdateSummary(const std::string& summary) {
    SetSection(TemplateSection::SUMMARY, summary, !summary.empty());
}

void ChatTemplateManager::UpdateOldChatSummary(const std::string& old_summary) {
    SetSection(TemplateSection::OLD_CHAT_SUMMARY, old_summary, !old_summary.empty());
}

void ChatTemplateManager::UpdateMotifContext(const std::string& motif) {
    SetSection(TemplateSection::MOTIF_CONTEXT, motif, !motif.empty());
}

void ChatTemplateManager::UpdateInternalReflection(const std::string& reflection) {
    SetSection(TemplateSection::INTERNAL_REFLECTION, reflection, !reflection.empty());
}

bool ChatTemplateManager::ValidateTemplate() const {
    // Basic validation - check if base template contains required elements
    bool has_messages_loop = base_template.find("for msg in messages") != std::string::npos;
    bool has_assistant_header = base_template.find("assistant<|end_header_id") != std::string::npos;
    
    return has_messages_loop && has_assistant_header;
}

void ChatTemplateManager::SetBaseTemplate(const std::string& new_template) {
    base_template = new_template;
    template_dirty = true;
    cached_rendered_template.clear();
}
