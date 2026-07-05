use crate::config::{Action, RolesConfig, When};
use crate::error::AppError;
use crate::error::AppResult;
use crate::models::chat::ChatFunctionCall;
use crate::models::chat::ChatMessage;
use crate::models::chat::ChatThinking;
use crate::models::chat::ChatTool;
use crate::models::chat::ChatToolCall;
use crate::models::chat::ChatToolDefinition;
use crate::models::responses::ContentItem;
use crate::models::responses::LocalShellAction;
use crate::models::responses::NamespaceToolSpec;
use crate::models::responses::ResponseItem;
use crate::models::responses::ResponsesRequest;
use crate::models::responses::ToolSpec;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum ToolKind {
    Function {
        public_name: String,
        namespace: Option<String>,
    },
    Custom {
        public_name: String,
    },
    LocalShell,
    ToolSearch,
    WebSearch,
    /// G4 server-side image analysis (`analyzeImage`). Registered ONLY when the
    /// image agent is active for the turn (see `lower_request`'s
    /// `image_agent_active`), so a client-supplied `analyzeImage` tool on a
    /// non-image turn still classifies as a normal client `Function`.
    ImageAnalysis,
}

#[derive(Debug, Clone)]
pub struct ToolRegistry {
    by_name: HashMap<String, ToolKind>,
}

impl ToolRegistry {
    pub fn get(&self, name: &str) -> Option<&ToolKind> {
        self.by_name.get(name)
    }

    pub(crate) fn has_active_server_tool(
        &self,
        web_search_active: bool,
        image_agent_active: bool,
    ) -> bool {
        self.by_name.values().any(|kind| match kind {
            ToolKind::WebSearch => web_search_active,
            ToolKind::ImageAnalysis => image_agent_active,
            _ => false,
        })
    }
}

#[cfg(test)]
impl ToolRegistry {
    pub fn from_map(by_name: HashMap<String, ToolKind>) -> Self {
        Self { by_name }
    }
}

#[derive(Debug, Clone)]
pub struct LoweredTurn {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ChatTool>,
    pub tool_registry: ToolRegistry,
    pub response_format: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
}

#[derive(Debug, Clone)]
struct PendingReasoning {
    text: String,
    signature: Option<String>,
}

impl PendingReasoning {
    fn from_parts(text: String, signature: Option<String>) -> Self {
        Self { text, signature }
    }

    fn append(&mut self, text: String, signature: Option<String>) {
        if !self.text.is_empty() && !text.is_empty() {
            self.text.push_str("\n\n");
            self.text.push_str(&text);
        } else if self.text.is_empty() {
            self.text = text;
        }
        if self.signature.is_none() {
            self.signature = signature;
        }
    }

    fn into_chat_parts(self) -> (Option<String>, Option<ChatThinking>) {
        let thinking = self.signature.clone().map(|signature| ChatThinking {
            content: self.text.clone(),
            signature: Some(signature),
        });
        (Some(self.text), thinking)
    }
}

pub fn lower_request(
    request: &ResponsesRequest,
    baseline_messages: Vec<ChatMessage>,
) -> AppResult<LoweredTurn> {
    lower_request_with_image_agent_and_roles(request, baseline_messages, false, None)
}

/// Like [`lower_request`], but `image_agent_active` decides whether an
/// `analyzeImage` tool classifies as the server-side [`ToolKind::ImageAnalysis`]
/// (true, the gateway runs it) or a plain client `Function` (false). The engine
/// passes `true` only on turns where G4 gating activated the image agent, so a
/// caller that happens to define its own `analyzeImage` tool on a text turn is
/// unaffected.
pub fn lower_request_with_image_agent(
    request: &ResponsesRequest,
    baseline_messages: Vec<ChatMessage>,
    image_agent_active: bool,
) -> AppResult<LoweredTurn> {
    lower_request_with_image_agent_and_roles(request, baseline_messages, image_agent_active, None)
}

/// Lower a turn while applying an optional model-profile role policy. Role
/// rules apply only to the new tail, never replayed baseline messages, so tags
/// and rewrites are not applied twice on follow-up turns.
pub fn lower_request_with_image_agent_and_roles(
    request: &ResponsesRequest,
    baseline_messages: Vec<ChatMessage>,
    image_agent_active: bool,
    roles: Option<&RolesConfig>,
) -> AppResult<LoweredTurn> {
    validate_request(request)?;
    let mut messages = baseline_messages;
    let baseline_len = messages.len();
    if messages.is_empty() && !request.instructions.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(Value::String(request.instructions.clone())),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking: None,
            tool_calls: None,
        });
    }
    let tools = lower_tools(&request.tools)?;
    let registry = build_tool_registry(&request.tools, image_agent_active)?;
    // Role rules apply only to the tail past the baseline (`split_off` below),
    // so with an active roles policy a tool call must not merge into the
    // replayed baseline's trailing assistant message — it would escape rule
    // processing. Without roles the merge may cross the boundary freely.
    let tool_call_merge_floor = if roles.is_some() { baseline_len } else { 0 };
    let mut pending_reasoning: Option<PendingReasoning> = None;
    for item in &request.input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let text = message_content_to_chat_value(content)?;
                let normalized_role = if roles.is_some() {
                    role.to_string()
                } else {
                    normalize_chat_role(role)
                };
                let (reasoning_content, thinking) = if normalized_role == "assistant" {
                    pending_reasoning
                        .take()
                        .map(PendingReasoning::into_chat_parts)
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                messages.push(ChatMessage {
                    role: normalized_role,
                    content: Some(text),
                    tool_call_id: None,
                    name: None,
                    reasoning_content,
                    thinking,
                    tool_calls: None,
                });
            }
            ResponseItem::Reasoning {
                summary,
                content,
                encrypted_content,
                ..
            } => {
                let text = reasoning_item_text(summary, content);
                let signature = encrypted_content
                    .as_ref()
                    .filter(|signature| !signature.is_empty())
                    .cloned();
                if let Some(existing) = pending_reasoning.as_mut() {
                    existing.append(text, signature);
                } else {
                    pending_reasoning = Some(PendingReasoning::from_parts(text, signature));
                }
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => append_tool_call(
                &mut messages,
                call_id.clone(),
                name.clone(),
                parse_json_string(arguments)?,
                pending_reasoning.take(),
                tool_call_merge_floor,
            ),
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => append_tool_call(
                &mut messages,
                call_id.clone(),
                name.clone(),
                json!({ "input": input }),
                pending_reasoning.take(),
                tool_call_merge_floor,
            ),
            ResponseItem::ToolSearchCall {
                call_id,
                arguments,
                execution,
                ..
            } => {
                if execution != "client" {
                    return Err(AppError::bad_request(
                        "only tool_search calls with execution=client are supported",
                    ));
                }
                append_tool_call(
                    &mut messages,
                    call_id
                        .clone()
                        .unwrap_or_else(|| "tool_search_missing_call_id".to_string()),
                    "tool_search".to_string(),
                    arguments.clone(),
                    pending_reasoning.take(),
                    tool_call_merge_floor,
                );
            }
            ResponseItem::LocalShellCall {
                call_id,
                id,
                action,
                ..
            } => {
                let call_id = call_id
                    .clone()
                    .or_else(|| id.clone())
                    .ok_or_else(|| AppError::bad_request("local_shell_call missing call_id"))?;
                let arguments = match action {
                    LocalShellAction::Exec(exec) => serde_json::to_value(exec).map_err(|err| {
                        AppError::bad_request(format!(
                            "failed to serialize local_shell_call action: {err}"
                        ))
                    })?,
                };
                append_tool_call(
                    &mut messages,
                    call_id,
                    "local_shell".to_string(),
                    arguments,
                    pending_reasoning.take(),
                    tool_call_merge_floor,
                );
            }
            ResponseItem::FunctionCallOutput { call_id, output }
            | ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => messages.push(ChatMessage {
                role: "tool".to_string(),
                content: Some(Value::String(stringify_tool_output(output))),
                tool_call_id: Some(call_id.clone()),
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            }),
            ResponseItem::ToolSearchOutput {
                call_id,
                status,
                execution,
                tools,
            } => messages.push(ChatMessage {
                role: "tool".to_string(),
                content: Some(Value::String(
                    serde_json::to_string(&json!({
                        "status": status,
                        "execution": execution,
                        "tools": tools,
                    }))
                    .map_err(|err| {
                        AppError::bad_request(format!(
                            "failed to serialize tool_search_output: {err}"
                        ))
                    })?,
                )),
                tool_call_id: call_id.clone(),
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            }),
            ResponseItem::WebSearchCall { id, action, .. } => {
                let call_id = id
                    .clone()
                    .unwrap_or_else(|| format!("web_search_missing_replay_{}", messages.len()));
                append_tool_call(
                    &mut messages,
                    call_id.clone(),
                    "web_search".to_string(),
                    web_search_arguments(action),
                    pending_reasoning.take(),
                    tool_call_merge_floor,
                );
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(Value::String(web_search_placeholder_result(action))),
                    tool_call_id: Some(call_id),
                    name: None,
                    reasoning_content: None,
                    thinking: None,
                    tool_calls: None,
                });
            }
            ResponseItem::ImageGenerationCall { .. } => {}
        }
    }
    if let Some(reasoning) = pending_reasoning.take() {
        let (reasoning_content, thinking) = reasoning.into_chat_parts();
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_call_id: None,
            name: None,
            reasoning_content,
            thinking,
            tool_calls: None,
        });
    }
    if roles.is_some() {
        let mut new_messages = messages.split_off(baseline_len);
        apply_role_rules(&mut new_messages, roles)?;
        messages.append(&mut new_messages);
    } else {
        // Preserve the fork's historical compatibility behavior unless a
        // profile explicitly opts into the more general role-policy system.
        hoist_system_messages(&mut messages);
    }
    let response_format = request
        .text
        .as_ref()
        .and_then(|text| text.format.as_ref())
        .map(|format| {
            json!({
                "type": format.kind,
                "json_schema": {
                    "name": format.name,
                    "schema": format.schema,
                    "strict": format.strict,
                }
            })
        });
    let reasoning_effort = normalize_reasoning_effort(
        request
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
    )?;
    Ok(LoweredTurn {
        messages,
        tools,
        tool_registry: registry,
        response_format,
        reasoning_effort,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
    })
}

pub(crate) fn apply_role_rules(
    messages: &mut Vec<ChatMessage>,
    roles: Option<&RolesConfig>,
) -> AppResult<()> {
    let Some(roles) = roles else {
        return Ok(());
    };
    let mut out = Vec::with_capacity(messages.len());
    for (index, mut message) in messages.drain(..).enumerate() {
        let Some(rules) = roles.rules_for(&message.role) else {
            return Err(AppError::bad_request(format!(
                "role \"{}\" is not permitted by the model profile roles config",
                message.role
            )));
        };
        let Some(rule) = rules.iter().find(|rule| match rule.when {
            None | Some(When::Always) => true,
            Some(When::Leading) => index == 0,
            Some(When::Inline) => index > 0,
        }) else {
            return Err(AppError::bad_request(format!(
                "role \"{}\" at index {} matches no rule in the model profile roles config",
                message.role, index
            )));
        };
        match rule.action {
            Action::Reject => {
                return Err(AppError::bad_request(format!(
                    "role \"{}\" is rejected by the model profile roles config",
                    message.role
                )));
            }
            Action::Drop => continue,
            Action::Accept => {
                wrap_role_tag(&mut message, &rule.tag, &rule.tag_attributes);
                out.push(message);
            }
            Action::Rewrite => {
                if let Some(target) = &rule.target_role {
                    message.role.clone_from(target);
                }
                wrap_role_tag(&mut message, &rule.tag, &rule.tag_attributes);
                out.push(message);
            }
        }
    }
    *messages = out;
    if !roles.merge_adjacent.is_empty() {
        merge_adjacent_role_runs(messages, &roles.merge_adjacent);
    }
    Ok(())
}

fn wrap_role_tag(
    message: &mut ChatMessage,
    tag: &Option<String>,
    attributes: &BTreeMap<String, String>,
) {
    let Some(tag) = tag else {
        return;
    };
    let Some(content) = message.content.take() else {
        return;
    };
    let text = match content {
        Value::String(text) => text,
        other => other.to_string(),
    };
    let mut open = format!("<{tag}");
    for (key, value) in attributes {
        open.push(' ');
        open.push_str(key);
        open.push_str("=\"");
        for ch in value.chars() {
            match ch {
                '&' => open.push_str("&amp;"),
                '"' => open.push_str("&quot;"),
                '<' => open.push_str("&lt;"),
                _ => open.push(ch),
            }
        }
        open.push('"');
    }
    open.push('>');
    message.content = Some(Value::String(format!("{open}{text}</{tag}>")));
}

fn merge_adjacent_role_runs(messages: &mut Vec<ChatMessage>, roles: &[String]) {
    let mut out = Vec::with_capacity(messages.len());
    let mut index = 0;
    while index < messages.len() {
        let role = messages[index].role.clone();
        if !roles.contains(&role) {
            out.push(messages[index].clone());
            index += 1;
            continue;
        }
        let mut parts = Vec::new();
        let mut tool_calls: Vec<ChatToolCall> = Vec::new();
        let mut reasoning: Option<PendingReasoning> = None;
        while index < messages.len() && messages[index].role == role {
            let message = &messages[index];
            if let Some(content) = &message.content {
                let text = match content {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                if !text.is_empty() {
                    parts.push(text);
                }
            }
            // Collapsing a run must not lose the run's tool calls or
            // reasoning: fold them onto the merged message instead of
            // rebuilding it content-only.
            if let Some(calls) = &message.tool_calls {
                tool_calls.extend(calls.iter().cloned());
            }
            if message.reasoning_content.is_some() || message.thinking.is_some() {
                let text = message.reasoning_content.clone().unwrap_or_default();
                let signature = message
                    .thinking
                    .as_ref()
                    .and_then(|thinking| thinking.signature.clone());
                match reasoning.as_mut() {
                    Some(existing) => existing.append(text, signature),
                    None => reasoning = Some(PendingReasoning::from_parts(text, signature)),
                }
            }
            index += 1;
        }
        if !parts.is_empty() || !tool_calls.is_empty() || reasoning.is_some() {
            let (reasoning_content, thinking) = reasoning
                .map(PendingReasoning::into_chat_parts)
                .unwrap_or((None, None));
            for (position, call) in tool_calls.iter_mut().enumerate() {
                call.index = Some(position);
            }
            out.push(ChatMessage {
                role,
                content: if parts.is_empty() {
                    None
                } else {
                    Some(Value::String(parts.join("\n\n")))
                },
                tool_call_id: None,
                name: None,
                reasoning_content,
                thinking,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            });
        }
    }
    *messages = out;
}

fn normalize_chat_role(role: &str) -> String {
    match role {
        "developer" => "system".to_string(),
        _ => role.to_string(),
    }
}

fn normalize_reasoning_effort(effort: Option<&str>) -> AppResult<Option<String>> {
    // Carry the RAW canonical level through (trimmed + lowercased) so the upstream
    // leaf — the single point that knows the FINAL provider model — can apply a
    // per-model `reasoning_effort_map` (which needs `xhigh`/`max` kept distinct)
    // or clamp it to a backend's vocabulary. The clamp lives at the leaf
    // (`upstream::clamp_reasoning_effort`), not here.
    Ok(effort
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase()))
}

fn hoist_system_messages(messages: &mut Vec<ChatMessage>) {
    // Find the end of the initial contiguous block of system messages.
    let prefix_end = messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(messages.len());
    if prefix_end == 0 {
        return;
    }
    let mut system_texts: Vec<String> = Vec::new();
    for msg in &messages[..prefix_end] {
        if let Some(content) = &msg.content {
            let text = match content {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if !text.is_empty() {
                system_texts.push(text);
            }
        }
    }
    let rest: Vec<ChatMessage> = messages.drain(prefix_end..).collect();
    messages.clear();
    if !system_texts.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(Value::String(system_texts.join("\n\n"))),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking: None,
            tool_calls: None,
        });
    }
    messages.extend(rest);
}

fn validate_request(request: &ResponsesRequest) -> AppResult<()> {
    if request.previous_response_id.is_some() {
        return Err(AppError::bad_request(
            "previous_response_id is not supported in v1",
        ));
    }
    // ImageGeneration tools are silently stripped (not sent to upstream).
    // Client-side MCP servers handle image generation via function tools.
    // Validate tool_choice
    match &request.tool_choice {
        Value::String(s) => match s.as_str() {
            "auto" | "none" => {}
            "required" => {
                if request.tools.is_empty() {
                    return Err(AppError::bad_request(
                        "tool_choice is \"required\" but no tools are provided",
                    ));
                }
            }
            other => {
                return Err(AppError::bad_request(format!(
                    "invalid tool_choice string: \"{other}\"; expected \"auto\", \"none\", or \"required\""
                )));
            }
        },
        Value::Object(map) => {
            let valid = map.get("type").and_then(|v| v.as_str()) == Some("function")
                && map
                    .get("function")
                    .and_then(|f| f.as_object())
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|n| !n.is_empty())
                    == Some(true);
            if !valid {
                return Err(AppError::bad_request(
                    "invalid tool_choice object: expected {\"type\":\"function\",\"function\":{\"name\":\"<non-empty>\"}}",
                ));
            }
            if request.tools.is_empty() {
                return Err(AppError::bad_request(
                    "tool_choice specifies a function but no tools are provided",
                ));
            }
        }
        _ => {
            return Err(AppError::bad_request(
                "invalid tool_choice: expected a string (\"auto\", \"none\", \"required\") or a function object",
            ));
        }
    }
    Ok(())
}

fn lower_tools(specs: &[ToolSpec]) -> AppResult<Vec<ChatTool>> {
    let mut tools = Vec::new();
    for spec in specs {
        let lowered_tools = match spec {
            ToolSpec::Function {
                name,
                description,
                strict,
                parameters,
            } => vec![ChatTool {
                kind: "function".to_string(),
                function: ChatToolDefinition {
                    name: name.clone(),
                    description: description.clone(),
                    parameters: Some(parameters.clone()),
                    strict: *strict,
                },
            }],
            ToolSpec::Namespace {
                tools: namespace_tools,
                ..
            } => namespace_tools
                .iter()
                .map(|tool| match tool {
                    NamespaceToolSpec::Function {
                        name,
                        description,
                        strict,
                        parameters,
                    } => ChatTool {
                        kind: "function".to_string(),
                        function: ChatToolDefinition {
                            name: name.clone(),
                            description: description.clone(),
                            parameters: Some(parameters.clone()),
                            strict: *strict,
                        },
                    },
                })
                .collect(),
            ToolSpec::ToolSearch {
                description,
                parameters,
                ..
            } => vec![ChatTool {
                kind: "function".to_string(),
                function: ChatToolDefinition {
                    name: "tool_search".to_string(),
                    description: description.clone(),
                    parameters: Some(parameters.clone()),
                    strict: false,
                },
            }],
            ToolSpec::LocalShell {} => vec![ChatTool {
                kind: "function".to_string(),
                function: ChatToolDefinition {
                    name: "local_shell".to_string(),
                    description: "Execute a shell command locally.".to_string(),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "array",
                                "items": { "type": "string" }
                            },
                            "timeout_ms": { "type": "integer" },
                            "working_directory": { "type": "string" },
                            "env": {
                                "type": "object",
                                "additionalProperties": { "type": "string" }
                            },
                            "user": { "type": "string" }
                        },
                        "required": ["command"]
                    })),
                    strict: false,
                },
            }],
            ToolSpec::WebSearch { .. } => vec![ChatTool {
                kind: "function".to_string(),
                function: ChatToolDefinition {
                    name: "web_search".to_string(),
                    description: "Search the web and return relevant result snippets.".to_string(),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    })),
                    strict: false,
                },
            }],
            ToolSpec::Custom {
                name,
                description,
                format,
            } => vec![ChatTool {
                kind: "function".to_string(),
                function: ChatToolDefinition {
                    name: name.clone(),
                    description: format!(
                        "{description}\n\nReturn the raw tool input as a string matching the {} {} format.",
                        format.kind, format.syntax
                    ),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "input": { "type": "string" }
                        },
                        "required": ["input"]
                    })),
                    strict: false,
                },
            }],
            ToolSpec::ImageGeneration { .. } => Vec::new(),
        };
        // Duplicate-tool-name rejection lives solely in `build_tool_registry`
        // (case-insensitive), which `lower_request` always calls on the same
        // tool slice. `lower_tools` only builds and sorts the chat tools.
        tools.extend(lowered_tools);
    }
    tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));
    Ok(tools)
}

fn build_tool_registry(specs: &[ToolSpec], image_agent_active: bool) -> AppResult<ToolRegistry> {
    let mut by_name = HashMap::new();
    for spec in specs {
        let lowered_kinds: Vec<(String, ToolKind)> = match spec {
            // G4: on an active image-agent turn, classify the injected (or
            // caller-supplied) `analyzeImage` function as the server-side
            // ImageAnalysis tool so the engine runs it instead of handing it to
            // the client. On a non-image turn it stays a normal client Function.
            ToolSpec::Function { name, .. }
                if image_agent_active
                    && name.eq_ignore_ascii_case(crate::vision::ANALYZE_IMAGE_TOOL_NAME) =>
            {
                vec![(name.clone(), ToolKind::ImageAnalysis)]
            }
            ToolSpec::Function { name, .. } => vec![(
                name.clone(),
                ToolKind::Function {
                    public_name: name.clone(),
                    namespace: None,
                },
            )],
            ToolSpec::Namespace {
                name: namespace,
                tools,
                ..
            } => tools
                .iter()
                .map(|tool| match tool {
                    NamespaceToolSpec::Function { name, .. } => (
                        name.clone(),
                        ToolKind::Function {
                            public_name: name.clone(),
                            namespace: Some(namespace.clone()),
                        },
                    ),
                })
                .collect(),
            ToolSpec::ToolSearch { .. } => {
                vec![("tool_search".to_string(), ToolKind::ToolSearch)]
            }
            ToolSpec::LocalShell {} => vec![("local_shell".to_string(), ToolKind::LocalShell)],
            ToolSpec::WebSearch { .. } => vec![("web_search".to_string(), ToolKind::WebSearch)],
            ToolSpec::Custom { name, .. } => vec![(
                name.clone(),
                ToolKind::Custom {
                    public_name: name.clone(),
                },
            )],
            ToolSpec::ImageGeneration { .. } => Vec::new(),
        };
        for (name, kind) in lowered_kinds {
            let name_lc = name.to_ascii_lowercase();
            if by_name.insert(name_lc.clone(), kind).is_some() {
                return Err(AppError::bad_request(format!(
                    "duplicate tool name is not supported: {name_lc}"
                )));
            }
        }
    }
    Ok(ToolRegistry { by_name })
}

fn append_tool_call(
    messages: &mut Vec<ChatMessage>,
    call_id: String,
    name: String,
    arguments: Value,
    pending_reasoning: Option<PendingReasoning>,
    merge_floor: usize,
) {
    // A tool call belongs to the assistant turn that produced it. In the
    // OpenAI Chat shape one assistant message carries BOTH `content` and
    // `tool_calls`, so absorb the call into the preceding assistant message
    // regardless of whether it already has text content. Splitting a
    // content-bearing assistant turn into a separate text message + a
    // tool-call message rewrites the history into a shape no OpenAI client
    // emits, and some backends (e.g. reasoning models that pattern-match their
    // own transcript) then learn to end a turn with a text preamble instead of
    // calling the tool. Only start a new assistant message when the previous
    // message is not an assistant (i.e. a tool result or user turn closed it),
    // or when merging would reach below `merge_floor`: role rules run only on
    // the tail past the replayed baseline, so a caller with an active roles
    // policy sets the floor to the baseline length to keep the tool call
    // inside the rule-processed tail.
    let mergeable = messages.len() > merge_floor
        && messages.last().is_some_and(|last| last.role == "assistant");
    let index = if mergeable {
        messages
            .last()
            .and_then(|last| last.tool_calls.as_ref())
            .map(|calls| calls.len())
            .unwrap_or(0)
    } else {
        0
    };
    let tool_call = ChatToolCall {
        id: Some(call_id),
        index: Some(index),
        kind: "function".to_string(),
        function: ChatFunctionCall {
            name: Some(name),
            arguments: Some(arguments),
        },
    };
    if mergeable && let Some(last) = messages.last_mut() {
        if let Some(existing) = &mut last.tool_calls {
            existing.push(tool_call);
        } else {
            last.tool_calls = Some(vec![tool_call]);
        }
        if let Some(reasoning) = pending_reasoning {
            // Interleaved thinking (Reasoning -> Message -> Reasoning ->
            // FunctionCall) leaves reasoning already on the merge target; fold
            // the new block in with the same join semantics as consecutive
            // Reasoning items instead of dropping it.
            let mut merged = PendingReasoning::from_parts(
                last.reasoning_content.take().unwrap_or_default(),
                last.thinking.take().and_then(|thinking| thinking.signature),
            );
            merged.append(reasoning.text, reasoning.signature);
            let (reasoning_content, thinking) = merged.into_chat_parts();
            last.reasoning_content = reasoning_content;
            last.thinking = thinking;
        }
        return;
    }
    let (reasoning_content, thinking) = pending_reasoning
        .map(PendingReasoning::into_chat_parts)
        .unwrap_or((None, None));
    messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: None,
        tool_call_id: None,
        name: None,
        reasoning_content,
        thinking,
        tool_calls: Some(vec![tool_call]),
    });
}

fn parse_json_string(raw: &str) -> AppResult<Value> {
    serde_json::from_str(raw)
        .map_err(|err| AppError::bad_request(format!("invalid JSON tool arguments: {err}")))
}

fn web_search_arguments(action: &Option<crate::models::responses::WebSearchAction>) -> Value {
    match action {
        Some(crate::models::responses::WebSearchAction::Search { query, queries }) => {
            if let Some(query) = query {
                json!({ "query": query })
            } else if let Some(query) = queries.as_ref().and_then(|queries| queries.first()) {
                json!({ "query": query })
            } else {
                json!({})
            }
        }
        Some(crate::models::responses::WebSearchAction::OpenPage { url }) => {
            json!({ "url": url })
        }
        Some(crate::models::responses::WebSearchAction::FindInPage { url, pattern }) => {
            json!({ "url": url, "pattern": pattern })
        }
        Some(crate::models::responses::WebSearchAction::Other) | None => json!({}),
    }
}

fn web_search_placeholder_result(
    action: &Option<crate::models::responses::WebSearchAction>,
) -> String {
    use crate::models::responses::WebSearchAction;

    // One base sentence, differing only by an optional action label, plus the
    // optional fields present for this action, in field order.
    let (action_label, fragments): (&str, Vec<String>) = match action {
        Some(WebSearchAction::Search { query, queries }) => {
            let query = query.clone().or_else(|| {
                queries
                    .as_ref()
                    .and_then(|queries| queries.first().cloned())
            });
            (
                "",
                query.into_iter().map(|q| format!("Query: {q}")).collect(),
            )
        }
        Some(WebSearchAction::OpenPage { url }) => (
            " open_page",
            url.iter().map(|u| format!("URL: {u}")).collect(),
        ),
        Some(WebSearchAction::FindInPage { url, pattern }) => {
            let mut fragments = Vec::new();
            if let Some(url) = url {
                fragments.push(format!("URL: {url}"));
            }
            if let Some(pattern) = pattern {
                fragments.push(format!("Pattern: {pattern}"));
            }
            (" find_in_page", fragments)
        }
        Some(WebSearchAction::Other) | None => ("", Vec::new()),
    };

    let mut result = format!(
        "Previous web_search{action_label} completed in an earlier turn, but the original tool result is unavailable because replay state was missing."
    );
    if !fragments.is_empty() {
        result.push_str(&format!(" {}", fragments.join(". ")));
    }
    result
}

fn reasoning_item_text(
    summary: &[crate::models::responses::ReasoningSummaryItem],
    content: &Option<Vec<crate::models::responses::ReasoningContentItem>>,
) -> String {
    let mut pieces = Vec::new();
    for entry in summary {
        let crate::models::responses::ReasoningSummaryItem::SummaryText { text } = entry;
        if !text.is_empty() {
            pieces.push(text.clone());
        }
    }
    if let Some(content) = content {
        for entry in content {
            match entry {
                crate::models::responses::ReasoningContentItem::ReasoningText { text }
                | crate::models::responses::ReasoningContentItem::Text { text }
                    if !text.is_empty() =>
                {
                    pieces.push(text.clone());
                }
                crate::models::responses::ReasoningContentItem::ReasoningText { .. }
                | crate::models::responses::ReasoningContentItem::Text { .. } => {}
            }
        }
    }
    pieces.join("\n")
}

fn message_content_to_chat_value(content: &[ContentItem]) -> AppResult<Value> {
    if content.is_empty() {
        return Ok(Value::String(String::new()));
    }
    if content.len() == 1 {
        return content_item_to_chat_value(&content[0]);
    }
    let mut parts = Vec::with_capacity(content.len());
    for item in content {
        parts.push(content_item_to_chat_part(item));
    }
    Ok(Value::Array(parts))
}

fn content_item_to_chat_value(item: &ContentItem) -> AppResult<Value> {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            Ok(Value::String(text.clone()))
        }
        ContentItem::InputImage { .. } | ContentItem::InputFile { .. } | ContentItem::Other(_) => {
            Ok(Value::Array(vec![content_item_to_chat_part(item)]))
        }
    }
}

fn content_item_to_chat_part(item: &ContentItem) -> Value {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => json!({
            "type": "text",
            "text": text,
        }),
        ContentItem::InputImage {
            image_url: Some(image_url),
            detail,
            ..
        } => {
            let mut image_url_value = Map::new();
            image_url_value.insert("url".to_string(), Value::String(image_url.clone()));
            if let Some(detail) = detail {
                image_url_value.insert("detail".to_string(), Value::String(detail.clone()));
            }
            json!({
                "type": "image_url",
                "image_url": Value::Object(image_url_value)
            })
        }
        ContentItem::InputImage {
            image_url: None,
            file_id,
            detail,
        } => {
            let mut part = Map::new();
            part.insert("type".to_string(), Value::String("input_image".to_string()));
            if let Some(file_id) = file_id {
                part.insert("file_id".to_string(), Value::String(file_id.clone()));
            }
            if let Some(detail) = detail {
                part.insert("detail".to_string(), Value::String(detail.clone()));
            }
            Value::Object(part)
        }
        ContentItem::InputFile {
            file_id,
            file_url,
            filename,
            file_data,
        } => {
            let mut part = Map::new();
            part.insert("type".to_string(), Value::String("input_file".to_string()));
            insert_optional_string(&mut part, "file_id", file_id);
            insert_optional_string(&mut part, "file_url", file_url);
            insert_optional_string(&mut part, "filename", filename);
            insert_optional_string(&mut part, "file_data", file_data);
            Value::Object(part)
        }
        ContentItem::Other(value) => value.clone(),
    }
}

fn insert_optional_string(map: &mut Map<String, Value>, key: &str, value: &Option<String>) {
    if let Some(value) = value {
        map.insert(key.to_string(), Value::String(value.clone()));
    }
}

pub fn stringify_tool_output(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}

pub fn tool_call_arguments_object(arguments: &Option<Value>) -> Value {
    match arguments {
        Some(Value::Object(map)) => Value::Object(map.clone()),
        Some(other) => other.clone(),
        None => Value::Object(Map::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::responses::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn base_test_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "test".to_string(),
            instructions: String::new(),
            input: vec![],
            tools: vec![],
            tool_choice: serde_json::Value::String("auto".to_string()),
            parallel_tool_calls: Some(false),
            reasoning: None,
            thinking: None,
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
            previous_response_id: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            frequency_penalty: None,
            presence_penalty: None,
            truncation: None,
            metadata: None,
            stop: None,
            extra_body: Default::default(),
        }
    }

    fn user_msg(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase: None,
        }
    }

    #[test]
    fn validate_accepts_stream_false() {
        let mut req = base_test_request();
        req.stream = false;
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn validate_rejects_previous_response_id() {
        let mut req = base_test_request();
        req.previous_response_id = Some("resp_123".to_string());
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn validate_accepts_all_tool_choice_values() {
        let req = base_test_request();
        assert!(validate_request(&req).is_ok());

        let mut req2 = base_test_request();
        req2.tool_choice = serde_json::Value::String("required".to_string());
        req2.tools = vec![ToolSpec::Function {
            name: "f".to_string(),
            description: "d".to_string(),
            strict: false,
            parameters: json!({}),
        }];
        assert!(validate_request(&req2).is_ok());

        let mut req3 = base_test_request();
        req3.tool_choice = serde_json::Value::String("none".to_string());
        assert!(validate_request(&req3).is_ok());

        let mut req4 = base_test_request();
        req4.tool_choice = json!({"type": "function", "function": {"name": "echo"}});
        req4.tools = vec![ToolSpec::Function {
            name: "echo".to_string(),
            description: "d".to_string(),
            strict: false,
            parameters: json!({}),
        }];
        assert!(validate_request(&req4).is_ok());
    }

    #[test]
    fn validate_accepts_image_generation_tool() {
        let mut req = base_test_request();
        req.tools = vec![ToolSpec::ImageGeneration {
            output_format: None,
        }];
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn trailing_reasoning_flushed_as_message() {
        let mut req = base_test_request();
        req.input = vec![
            user_msg("hello"),
            ResponseItem::Reasoning {
                id: "rsn_1".to_string(),
                summary: vec![ReasoningSummaryItem::SummaryText {
                    text: "thinking".to_string(),
                }],
                content: None,
                encrypted_content: None,
            },
        ];
        let result = lower_request(&req, vec![]).unwrap();
        let last = result.messages.last().unwrap();
        assert_eq!(last.role, "assistant");
        assert!(last.reasoning_content.is_some());
        assert!(last.content.is_none());
    }

    #[test]
    fn signed_reasoning_history_preserves_chat_thinking_signature() {
        let mut req = base_test_request();
        req.input = vec![
            user_msg("hello"),
            ResponseItem::Reasoning {
                id: "rsn_1".to_string(),
                summary: Vec::new(),
                content: Some(vec![ReasoningContentItem::ReasoningText {
                    text: "private chain".to_string(),
                }]),
                encrypted_content: Some("sig_history".to_string()),
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "answer".to_string(),
                }],
                phase: None,
            },
        ];

        let result = lower_request(&req, vec![]).unwrap();
        let assistant = result
            .messages
            .iter()
            .find(|message| message.role == "assistant")
            .expect("assistant message");
        assert_eq!(
            assistant.reasoning_content.as_deref(),
            Some("private chain")
        );
        let thinking = assistant.thinking.as_ref().expect("signed thinking");
        assert_eq!(thinking.content, "private chain");
        assert_eq!(thinking.signature.as_deref(), Some("sig_history"));
    }

    #[test]
    fn lowers_reasoning_effort_raw_for_the_leaf() {
        // Lowering passes the canonical level through RAW (no clamp); the leaf
        // clamps or maps it. xhigh/max stay distinct here.
        for raw in ["none", "low", "medium", "high", "xhigh", "max"] {
            let mut req = base_test_request();
            req.reasoning = Some(ReasoningRequest {
                effort: Some(raw.to_string()),
                summary: None,
            });
            let result = lower_request(&req, vec![]).expect("lower_request");
            assert_eq!(
                result.reasoning_effort.as_deref(),
                Some(raw),
                "{raw} must pass through raw"
            );
        }
    }

    #[test]
    fn lowers_reasoning_effort_trimmed_and_lowercased() {
        let mut req = base_test_request();
        req.reasoning = Some(ReasoningRequest {
            effort: Some("  XHigh  ".to_string()),
            summary: None,
        });
        let result = lower_request(&req, vec![]).expect("lower_request");
        assert_eq!(result.reasoning_effort.as_deref(), Some("xhigh"));
    }

    #[test]
    fn role_rules_shape_only_new_tail_then_merge_adjacent() {
        let roles: RolesConfig = serde_json::from_value(json!({
            "merge_adjacent": ["user"],
            "developer": {
                "action": "rewrite",
                "target_role": "user",
                "tag": "instruction",
                "tag_attributes": {"priority": "high & urgent"}
            },
            "user": {},
            "*": {"action": "reject"}
        }))
        .expect("roles config");
        let baseline = vec![ChatMessage {
            role: "developer".to_string(),
            content: Some(json!("already shaped history")),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking: None,
            tool_calls: None,
        }];
        let mut request = base_test_request();
        request.input = vec![
            ResponseItem::Message {
                id: None,
                role: "developer".to_string(),
                content: vec![ContentItem::InputText {
                    text: "new policy".to_string(),
                }],
                phase: None,
            },
            user_msg("question"),
        ];

        let lowered =
            lower_request_with_image_agent_and_roles(&request, baseline, false, Some(&roles))
                .expect("lower request");

        assert_eq!(lowered.messages.len(), 2);
        assert_eq!(lowered.messages[0].role, "developer");
        assert_eq!(
            lowered.messages[0].content,
            Some(json!("already shaped history")),
            "replay baseline must not be rewritten or tagged again"
        );
        assert_eq!(lowered.messages[1].role, "user");
        assert_eq!(
            lowered.messages[1].content.as_ref().and_then(Value::as_str),
            Some(
                "<instruction priority=\"high &amp; urgent\">new policy</instruction>\n\nquestion"
            )
        );
    }

    #[test]
    fn duplicate_tool_name_rejected() {
        // Rejection is the registry's sole responsibility, reached end-to-end
        // via `lower_request`. Exact-case duplicate.
        let mut req = base_test_request();
        req.tools = vec![
            ToolSpec::Function {
                name: "echo".to_string(),
                description: "a".to_string(),
                strict: false,
                parameters: json!({}),
            },
            ToolSpec::Function {
                name: "echo".to_string(),
                description: "b".to_string(),
                strict: false,
                parameters: json!({}),
            },
        ];
        let err = lower_request(&req, vec![]).expect_err("duplicate name must be rejected");
        assert!(
            err.to_string()
                .contains("duplicate tool name is not supported: "),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_tool_name_rejected_case_insensitive() {
        // The surviving (registry) check folds case: `echo` vs `ECHO` collide,
        // which the removed case-sensitive `lower_tools` map never caught.
        let mut req = base_test_request();
        req.tools = vec![
            ToolSpec::Function {
                name: "echo".to_string(),
                description: "a".to_string(),
                strict: false,
                parameters: json!({}),
            },
            ToolSpec::Function {
                name: "ECHO".to_string(),
                description: "b".to_string(),
                strict: false,
                parameters: json!({}),
            },
        ];
        let err = lower_request(&req, vec![]).expect_err("case-insensitive duplicate must reject");
        assert!(
            err.to_string()
                .contains("duplicate tool name is not supported: "),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn mixed_text_and_image_content() {
        let content = vec![
            ContentItem::InputText {
                text: "hi".to_string(),
            },
            ContentItem::InputImage {
                image_url: Some("http://img.png".to_string()),
                file_id: None,
                detail: None,
            },
        ];
        let value = message_content_to_chat_value(&content).unwrap();
        assert!(value.is_array());
        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "image_url");
        assert_eq!(arr[1]["image_url"]["url"], "http://img.png");
    }

    #[test]
    fn single_input_image_wraps_as_array() {
        let item = ContentItem::InputImage {
            image_url: Some("http://img.png".to_string()),
            file_id: None,
            detail: Some("high".to_string()),
        };
        let value = content_item_to_chat_value(&item).unwrap();
        assert!(value.is_array());
        assert_eq!(value[0]["type"], "image_url");
        assert_eq!(value[0]["image_url"]["url"], "http://img.png");
        assert_eq!(value[0]["image_url"]["detail"], "high");
    }

    #[test]
    fn input_file_passes_through_as_content_part() {
        let item = ContentItem::InputFile {
            file_id: Some("file_123".to_string()),
            file_url: None,
            filename: Some("brief.pdf".to_string()),
            file_data: None,
        };
        let value = content_item_to_chat_value(&item).unwrap();
        assert!(value.is_array());
        assert_eq!(value[0]["type"], "input_file");
        assert_eq!(value[0]["file_id"], "file_123");
        assert_eq!(value[0]["filename"], "brief.pdf");
    }

    #[test]
    fn web_search_arguments_all_actions() {
        let search = Some(WebSearchAction::Search {
            query: Some("test".to_string()),
            queries: None,
        });
        assert_eq!(web_search_arguments(&search), json!({"query": "test"}));

        let open = Some(WebSearchAction::OpenPage {
            url: Some("http://x.com".to_string()),
        });
        assert_eq!(web_search_arguments(&open), json!({"url": "http://x.com"}));

        let find = Some(WebSearchAction::FindInPage {
            url: Some("http://x.com".to_string()),
            pattern: Some("foo".to_string()),
        });
        assert_eq!(
            web_search_arguments(&find),
            json!({"url": "http://x.com", "pattern": "foo"})
        );

        assert_eq!(
            web_search_arguments(&Some(WebSearchAction::Other)),
            json!({})
        );
        assert_eq!(web_search_arguments(&None), json!({}));
    }

    #[test]
    fn web_search_placeholder_result_all_actions() {
        let search = Some(WebSearchAction::Search {
            query: Some("test".to_string()),
            queries: None,
        });
        assert!(web_search_placeholder_result(&search).contains("test"));

        let open = Some(WebSearchAction::OpenPage {
            url: Some("http://x.com".to_string()),
        });
        assert!(web_search_placeholder_result(&open).contains("http://x.com"));

        let find = Some(WebSearchAction::FindInPage {
            url: Some("http://x.com".to_string()),
            pattern: Some("foo".to_string()),
        });
        let result = web_search_placeholder_result(&find);
        assert!(result.contains("http://x.com"));
        assert!(result.contains("foo"));

        assert!(
            web_search_placeholder_result(&Some(WebSearchAction::Other))
                .contains("replay state was missing")
        );
        assert!(web_search_placeholder_result(&None).contains("replay state was missing"));
    }

    #[test]
    fn web_search_placeholder_result_byte_exact() {
        // Full-string equality guards against wording/spacing/punctuation drift.
        const BASE: &str = "Previous web_search completed in an earlier turn, but the original tool result is unavailable because replay state was missing.";

        // Search: with `query`, with only `queries`, with neither.
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::Search {
                query: Some("cats".to_string()),
                queries: Some(vec!["ignored".to_string()]),
            })),
            format!("{BASE} Query: cats")
        );
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::Search {
                query: None,
                queries: Some(vec!["dogs".to_string(), "second".to_string()]),
            })),
            format!("{BASE} Query: dogs")
        );
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::Search {
                query: None,
                queries: None,
            })),
            BASE
        );

        // OpenPage: with/without url.
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::OpenPage {
                url: Some("http://x.com".to_string()),
            })),
            "Previous web_search open_page completed in an earlier turn, but the original tool result is unavailable because replay state was missing. URL: http://x.com"
        );
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::OpenPage { url: None })),
            "Previous web_search open_page completed in an earlier turn, but the original tool result is unavailable because replay state was missing."
        );

        // FindInPage: url+pattern, url-only, pattern-only, neither.
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::FindInPage {
                url: Some("http://x.com".to_string()),
                pattern: Some("foo".to_string()),
            })),
            "Previous web_search find_in_page completed in an earlier turn, but the original tool result is unavailable because replay state was missing. URL: http://x.com. Pattern: foo"
        );
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::FindInPage {
                url: Some("http://x.com".to_string()),
                pattern: None,
            })),
            "Previous web_search find_in_page completed in an earlier turn, but the original tool result is unavailable because replay state was missing. URL: http://x.com"
        );
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::FindInPage {
                url: None,
                pattern: Some("foo".to_string()),
            })),
            "Previous web_search find_in_page completed in an earlier turn, but the original tool result is unavailable because replay state was missing. Pattern: foo"
        );
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::FindInPage {
                url: None,
                pattern: None,
            })),
            "Previous web_search find_in_page completed in an earlier turn, but the original tool result is unavailable because replay state was missing."
        );

        // Other and None: bare base.
        assert_eq!(
            web_search_placeholder_result(&Some(WebSearchAction::Other)),
            BASE
        );
        assert_eq!(web_search_placeholder_result(&None), BASE);
    }

    #[test]
    fn tool_search_call_non_client_error() {
        let mut req = base_test_request();
        req.input = vec![
            user_msg("hello"),
            ResponseItem::ToolSearchCall {
                call_id: Some("ts_1".to_string()),
                status: None,
                execution: "server".to_string(),
                arguments: json!({}),
            },
        ];
        let result = lower_request(&req, vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("execution=client"));
    }

    #[test]
    fn stringify_tool_output_non_string() {
        assert_eq!(stringify_tool_output(&json!(42)), "42");
        assert_eq!(stringify_tool_output(&json!(true)), "true");
        assert_eq!(stringify_tool_output(&json!({"a": 1})), r#"{"a":1}"#);
    }

    #[test]
    fn tool_call_arguments_object_edge_cases() {
        assert_eq!(tool_call_arguments_object(&None), json!({}));
        assert_eq!(tool_call_arguments_object(&Some(json!("x"))), json!("x"));
        assert_eq!(
            tool_call_arguments_object(&Some(json!({"a": 1}))),
            json!({"a": 1})
        );
    }

    #[test]
    fn hoist_system_messages_non_string_content() {
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(json!({"key": "value"})),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!("hello")),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
        ];
        hoist_system_messages(&mut messages);
        assert_eq!(messages[0].role, "system");
        let content = messages[0].content.as_ref().unwrap().as_str().unwrap();
        assert!(content.contains("key"));
        assert!(content.contains("value"));
    }

    #[test]
    fn reasoning_item_text_variants() {
        let summary = vec![ReasoningSummaryItem::SummaryText {
            text: "summary".to_string(),
        }];
        let content = Some(vec![
            ReasoningContentItem::ReasoningText {
                text: "reasoning".to_string(),
            },
            ReasoningContentItem::Text {
                text: "text".to_string(),
            },
        ]);
        let result = reasoning_item_text(&summary, &content);
        assert!(result.contains("summary"));
        assert!(result.contains("reasoning"));
        assert!(result.contains("text"));
    }

    // --- C1 tests ---

    #[test]
    fn test_validate_tool_choice_valid_strings() {
        for val in &["auto", "none"] {
            let mut req = base_test_request();
            req.tool_choice = Value::String(val.to_string());
            assert!(validate_request(&req).is_ok(), "expected {val} to pass");
        }
        let mut req = base_test_request();
        req.tool_choice = Value::String("required".to_string());
        req.tools = vec![ToolSpec::Function {
            name: "f".to_string(),
            description: "d".to_string(),
            strict: false,
            parameters: json!({}),
        }];
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn test_validate_tool_choice_valid_object() {
        let mut req = base_test_request();
        req.tool_choice = json!({"type": "function", "function": {"name": "foo"}});
        req.tools = vec![ToolSpec::Function {
            name: "foo".to_string(),
            description: "d".to_string(),
            strict: false,
            parameters: json!({}),
        }];
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn test_validate_tool_choice_rejects_arbitrary_json() {
        let mut req = base_test_request();
        req.tool_choice = json!(42);
        assert!(validate_request(&req).is_err());

        let mut req2 = base_test_request();
        req2.tool_choice = json!([1, 2, 3]);
        assert!(validate_request(&req2).is_err());

        let mut req3 = base_test_request();
        req3.tool_choice = json!({"type": "unknown"});
        assert!(validate_request(&req3).is_err());

        let mut req4 = base_test_request();
        req4.tool_choice = Value::String("bogus".to_string());
        assert!(validate_request(&req4).is_err());
    }

    #[test]
    fn test_validate_tool_choice_required_without_tools_rejected() {
        let mut req = base_test_request();
        req.tool_choice = Value::String("required".to_string());
        req.tools = vec![];
        assert!(validate_request(&req).is_err());
    }

    // --- append_tool_call merge tests ---

    #[test]
    fn test_append_tool_call_sequential_indices() {
        let mut messages: Vec<ChatMessage> = vec![];
        for i in 0..3 {
            append_tool_call(
                &mut messages,
                format!("call_{i}"),
                format!("fn_{i}"),
                json!({}),
                None,
                0,
            );
        }
        assert_eq!(messages.len(), 1);
        let calls = messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[1].index, Some(1));
        assert_eq!(calls[2].index, Some(2));
    }

    #[test]
    fn test_append_tool_call_merges_into_content_message() {
        // A content-bearing assistant message must absorb the following tool
        // call into the SAME message (`content` + `tool_calls`), the canonical
        // OpenAI Chat shape — not split into two adjacent assistant messages.
        let mut messages = vec![ChatMessage {
            role: "assistant".to_string(),
            content: Some(Value::String("some text".to_string())),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking: None,
            tool_calls: None,
        }];
        append_tool_call(
            &mut messages,
            "call_1".to_string(),
            "fn_1".to_string(),
            json!({}),
            None,
            0,
        );
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].content,
            Some(Value::String("some text".to_string())),
            "text content must be preserved on the merged message"
        );
        let calls = messages[0].tool_calls.as_ref().expect("tool call merged");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[0].function.name.as_deref(), Some("fn_1"));
    }

    #[test]
    fn lowering_keeps_assistant_text_and_tool_call_in_one_message() {
        // End-to-end: an assistant turn that emitted BOTH text and a tool call
        // (as separate canonical items) must lower to a SINGLE upstream chat
        // assistant message carrying `content` + `tool_calls` — the OpenAI Chat
        // shape a real client emits — not two adjacent assistant messages.
        let mut req = base_test_request();
        req.input = vec![
            user_msg("look at the repo"),
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "Let me explore the key directories.".to_string(),
                }],
                phase: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "read".to_string(),
                namespace: None,
                arguments: "{\"path\":\"src\"}".to_string(),
                call_id: "call_1".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: Value::String("file list".to_string()),
            },
            user_msg("continue"),
        ];

        let lowered = lower_request(&req, vec![]).expect("lower_request");
        let roles: Vec<&str> = lowered.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "user"]);

        let assistant = &lowered.messages[1];
        assert_eq!(
            assistant.content,
            Some(Value::String(
                "Let me explore the key directories.".to_string()
            )),
            "assistant text must be preserved on the same message as the tool call"
        );
        let calls = assistant
            .tool_calls
            .as_ref()
            .expect("assistant message carries the tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name.as_deref(), Some("read"));
        assert_eq!(lowered.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn interleaved_reasoning_survives_tool_call_merge() {
        // Interleaved reasoning: Reasoning -> Message -> Reasoning ->
        // FunctionCall. Any ingress can produce this ordering (raw Responses
        // reasoning items, chat `reasoning_content`, Anthropic `thinking`
        // blocks) — a model that thinks, emits text, thinks again, then calls
        // a tool. The text Message consumes the first reasoning block, so the
        // merge target already carries `reasoning_content` when the second
        // block arrives with the tool call. The second block (and its
        // signature) must be folded in with the same `\n\n` join as
        // consecutive Reasoning items — not dropped.
        let mut req = base_test_request();
        req.input = vec![
            user_msg("look at the repo"),
            ResponseItem::Reasoning {
                id: "rs_1".to_string(),
                summary: vec![ReasoningSummaryItem::SummaryText {
                    text: "first block".to_string(),
                }],
                content: None,
                encrypted_content: None,
            },
            ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "Let me explore.".to_string(),
                }],
                phase: None,
            },
            ResponseItem::Reasoning {
                id: "rs_2".to_string(),
                summary: vec![ReasoningSummaryItem::SummaryText {
                    text: "second block".to_string(),
                }],
                content: None,
                encrypted_content: Some("sig_2".to_string()),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "read".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "call_1".to_string(),
            },
        ];

        let lowered = lower_request(&req, vec![]).expect("lower_request");
        let roles: Vec<&str> = lowered.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant"]);

        let assistant = &lowered.messages[1];
        assert_eq!(
            assistant.reasoning_content.as_deref(),
            Some("first block\n\nsecond block"),
            "second reasoning block must be appended, not dropped"
        );
        let thinking = assistant
            .thinking
            .as_ref()
            .expect("signature from the second block must survive the merge");
        assert_eq!(thinking.signature.as_deref(), Some("sig_2"));
        assert!(assistant.tool_calls.is_some());
    }

    #[test]
    fn roles_policy_keeps_tool_call_out_of_replay_baseline() {
        // Role rules run only on the tail past the replayed baseline. With an
        // active roles policy a tool call that follows a baseline-terminal
        // assistant message must start a NEW tail message (rule-processed),
        // not merge below the baseline boundary where rules never look.
        let roles: RolesConfig = serde_json::from_value(json!({
            "user": {},
            "assistant": {},
            "tool": {},
            "*": {"action": "reject"}
        }))
        .expect("roles config");
        let baseline = vec![ChatMessage {
            role: "assistant".to_string(),
            content: Some(Value::String("already served text".to_string())),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking: None,
            tool_calls: None,
        }];
        let mut request = base_test_request();
        request.input = vec![ResponseItem::FunctionCall {
            id: None,
            name: "read".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "call_1".to_string(),
        }];

        let lowered = lower_request_with_image_agent_and_roles(
            &request,
            baseline.clone(),
            false,
            Some(&roles),
        )
        .expect("lower request");
        assert_eq!(lowered.messages.len(), 2);
        assert!(
            lowered.messages[0].tool_calls.is_none(),
            "replay baseline message must stay untouched"
        );
        assert_eq!(lowered.messages[1].role, "assistant");
        assert!(
            lowered.messages[1].tool_calls.is_some(),
            "tool call must land in the rule-processed tail"
        );

        // Without a roles policy there is no tail scoping: the canonical
        // merged shape crosses the baseline boundary freely.
        let lowered = lower_request_with_image_agent_and_roles(&request, baseline, false, None)
            .expect("lower request");
        assert_eq!(lowered.messages.len(), 1);
        assert!(lowered.messages[0].tool_calls.is_some());
    }

    #[test]
    fn merge_adjacent_role_runs_carries_tool_calls_and_reasoning() {
        // Collapsing an adjacent same-role run must not lose the run's tool
        // calls, reasoning, or thinking signature — the merged message carries
        // them, re-indexed, instead of being rebuilt content-only.
        let mut messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("part one".to_string())),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_call_id: None,
                name: None,
                reasoning_content: Some("thinking".to_string()),
                thinking: Some(ChatThinking {
                    content: "thinking".to_string(),
                    signature: Some("sig_1".to_string()),
                }),
                tool_calls: Some(vec![ChatToolCall {
                    id: Some("call_1".to_string()),
                    index: Some(0),
                    kind: "function".to_string(),
                    function: ChatFunctionCall {
                        name: Some("read".to_string()),
                        arguments: Some(json!({})),
                    },
                }]),
            },
        ];
        merge_adjacent_role_runs(&mut messages, &["assistant".to_string()]);
        assert_eq!(messages.len(), 1);
        let merged = &messages[0];
        assert_eq!(merged.content, Some(Value::String("part one".to_string())));
        assert_eq!(merged.reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(
            merged
                .thinking
                .as_ref()
                .and_then(|thinking| thinking.signature.as_deref()),
            Some("sig_1")
        );
        let calls = merged.tool_calls.as_ref().expect("tool calls carried");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name.as_deref(), Some("read"));
        assert_eq!(calls[0].index, Some(0));
    }

    // --- M2 test ---

    #[test]
    fn test_hoist_preserves_mid_conversation_system_messages() {
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("top".to_string())),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("hello".to_string())),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "system".to_string(),
                content: Some(Value::String("mid".to_string())),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::String("hi".to_string())),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            },
        ];
        hoist_system_messages(&mut messages);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content.as_ref().unwrap().as_str().unwrap(),
            "top"
        );
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "system");
        assert_eq!(
            messages[2].content.as_ref().unwrap().as_str().unwrap(),
            "mid"
        );
        assert_eq!(messages[3].role, "assistant");
    }
}
