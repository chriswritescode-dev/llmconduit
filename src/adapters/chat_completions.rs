use crate::engine::SseEvent;
use crate::error::AppError;
use crate::error::AppResult;
use crate::models::chat::ChatCompletionRequest;
use crate::models::chat::ChatFunctionCall;
use crate::models::chat::ChatMessage;
use crate::models::chat::ChatTool;
use crate::models::chat::ChatToolCall;
use crate::models::responses::ContentItem;
use crate::models::responses::ReasoningContentItem;
use crate::models::responses::ReasoningRequest;
use crate::models::responses::ResponseItem;
use crate::models::responses::ResponsesRequest;
use crate::models::responses::TerminalReason;
use crate::models::responses::TextControls;
use crate::models::responses::TextFormat;
use crate::models::responses::ToolSpec;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

pub fn convert_request(request: ChatCompletionRequest) -> AppResult<ResponsesRequest> {
    let input = convert_messages(&request.messages)?;
    let tools = convert_tools(&request.tools);
    let extra_body = request.extra_body.clone();
    Ok(ResponsesRequest {
        model: request.model,
        instructions: String::new(),
        input,
        tools,
        tool_choice: request
            .tool_choice
            .unwrap_or_else(|| Value::String("auto".to_string())),
        parallel_tool_calls: request.parallel_tool_calls,
        reasoning: request.reasoning_effort.map(|effort| ReasoningRequest {
            effort: Some(effort),
            summary: None,
        }),
        // Chat Completions clients control the upstream thinking kwarg directly via extra_body.
        thinking: None,
        store: false,
        stream: true,
        include: Vec::new(),
        service_tier: None,
        prompt_cache_key: None,
        text: convert_response_format(request.response_format.as_ref()),
        client_metadata: None,
        previous_response_id: None,
        temperature: request.temperature,
        top_p: request.top_p,
        max_output_tokens: request.max_output_tokens,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        truncation: None,
        metadata: None,
        stop: request.stop,
        extra_body,
    })
}

fn convert_messages(messages: &[ChatMessage]) -> AppResult<Vec<ResponseItem>> {
    let mut items = Vec::new();
    for message in messages {
        convert_message(message, &mut items)?;
    }
    Ok(items)
}

fn convert_message(message: &ChatMessage, items: &mut Vec<ResponseItem>) -> AppResult<()> {
    match message.role.as_str() {
        "tool" | "function" => {
            let call_id = message
                .tool_call_id
                .clone()
                .or_else(|| message.name.clone())
                .ok_or_else(|| {
                    AppError::bad_request("Chat tool message is missing tool_call_id")
                })?;
            items.push(ResponseItem::FunctionCallOutput {
                call_id,
                output: tool_output_value(message.content.as_ref()),
            });
        }
        "assistant" => {
            if let Some(reasoning) = message
                .reasoning_content
                .as_deref()
                .filter(|text| !text.is_empty())
            {
                items.push(ResponseItem::Reasoning {
                    id: format!("rsn_{}", Uuid::new_v4().simple()),
                    summary: Vec::new(),
                    content: Some(vec![ReasoningContentItem::ReasoningText {
                        text: reasoning.to_string(),
                    }]),
                    encrypted_content: None,
                });
            }
            let content = chat_content_to_response_content(&message.role, message.content.as_ref());
            if !content.is_empty() {
                items.push(ResponseItem::Message {
                    id: None,
                    role: message.role.clone(),
                    content,
                    phase: None,
                });
            }
            if let Some(tool_calls) = &message.tool_calls {
                for tool_call in tool_calls {
                    items.push(chat_tool_call_to_response_item(tool_call)?);
                }
            }
        }
        _ => {
            items.push(ResponseItem::Message {
                id: None,
                role: message.role.clone(),
                content: chat_content_to_response_content(&message.role, message.content.as_ref()),
                phase: None,
            });
        }
    }
    Ok(())
}

fn chat_tool_call_to_response_item(tool_call: &ChatToolCall) -> AppResult<ResponseItem> {
    let name = tool_call.function.name.clone().ok_or_else(|| {
        AppError::bad_request("Chat assistant tool_call is missing function name")
    })?;
    Ok(ResponseItem::FunctionCall {
        id: None,
        name,
        namespace: None,
        arguments: arguments_to_string(tool_call.function.arguments.as_ref())?,
        call_id: tool_call
            .id
            .clone()
            .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple())),
    })
}

fn arguments_to_string(arguments: Option<&Value>) -> AppResult<String> {
    match arguments {
        Some(Value::String(text)) if text.trim().is_empty() => Ok("{}".to_string()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(value) => serde_json::to_string(value)
            .map_err(|err| AppError::bad_request(format!("invalid Chat tool arguments: {err}"))),
        None => Ok("{}".to_string()),
    }
}

fn tool_output_value(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => Value::String(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(chat_content_part_text)
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                Value::Array(parts.clone())
            } else {
                Value::String(text)
            }
        }
        Some(value) => value.clone(),
        None => Value::Null,
    }
}

fn chat_content_to_response_content(role: &str, content: Option<&Value>) -> Vec<ContentItem> {
    let Some(content) = content else {
        return Vec::new();
    };
    match content {
        Value::Null => Vec::new(),
        Value::String(text) => vec![text_content_item(role, text.clone())],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| chat_content_part_to_response_content(role, part))
            .collect(),
        other => vec![text_content_item(role, other.to_string())],
    }
}

fn chat_content_part_to_response_content(role: &str, part: &Value) -> Option<ContentItem> {
    let object = part.as_object()?;
    let kind = object.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "image_url" => extract_image_url(object).map(|image_url| {
            if role == "assistant" {
                text_content_item(role, part.to_string())
            } else {
                ContentItem::InputImage {
                    image_url: Some(image_url),
                    file_id: None,
                    detail: extract_image_detail(object),
                }
            }
        }),
        "input_image" => {
            if role == "assistant" {
                return Some(text_content_item(role, part.to_string()));
            }
            Some(ContentItem::InputImage {
                image_url: extract_image_url(object),
                file_id: optional_string(object, "file_id"),
                detail: extract_image_detail(object),
            })
        }
        "input_file" => {
            if role == "assistant" {
                return Some(text_content_item(role, part.to_string()));
            }
            Some(ContentItem::InputFile {
                file_id: optional_string(object, "file_id"),
                file_url: optional_string(object, "file_url"),
                filename: optional_string(object, "filename"),
                file_data: optional_string(object, "file_data"),
            })
        }
        "text" | "input_text" | "output_text" => object
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text_content_item(role, text.to_string())),
        _ if role == "assistant" => Some(text_content_item(role, part.to_string())),
        _ => Some(ContentItem::Other(part.clone())),
    }
}

fn chat_content_part_text(part: &Value) -> Option<String> {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    if part.is_null() {
        None
    } else {
        Some(part.to_string())
    }
}

fn extract_image_url(object: &serde_json::Map<String, Value>) -> Option<String> {
    object
        .get("image_url")
        .and_then(|value| match value {
            Value::String(url) => Some(url.clone()),
            Value::Object(map) => map
                .get("url")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            _ => None,
        })
        .or_else(|| {
            object
                .get("url")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

fn extract_image_detail(object: &serde_json::Map<String, Value>) -> Option<String> {
    optional_string(object, "detail").or_else(|| {
        object
            .get("image_url")
            .and_then(Value::as_object)
            .and_then(|map| optional_string(map, "detail"))
    })
}

fn optional_string(object: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn text_content_item(role: &str, text: String) -> ContentItem {
    if role == "assistant" {
        ContentItem::OutputText { text }
    } else {
        ContentItem::InputText { text }
    }
}

fn convert_tools(tools: &Option<Vec<ChatTool>>) -> Vec<ToolSpec> {
    let mut converted: Vec<ToolSpec> = tools
        .as_ref()
        .map(|tools| tools.iter().map(convert_tool).collect())
        .unwrap_or_default();
    converted.sort_by(|a, b| tool_sort_key(a).cmp(tool_sort_key(b)));
    converted
}

fn convert_tool(tool: &ChatTool) -> ToolSpec {
    if tool.kind == "function" && tool.function.name == "web_search" {
        return ToolSpec::WebSearch {
            external_web_access: Some(true),
            filters: None,
            user_location: None,
            search_context_size: None,
            search_content_types: None,
        };
    }
    ToolSpec::Function {
        name: tool.function.name.clone(),
        description: tool.function.description.clone(),
        strict: tool.function.strict,
        parameters: tool
            .function
            .parameters
            .clone()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
    }
}

fn tool_sort_key(tool: &ToolSpec) -> &str {
    match tool {
        ToolSpec::Function { name, .. } => name,
        ToolSpec::WebSearch { .. } => "web_search",
        ToolSpec::Namespace { name, .. } => name,
        ToolSpec::ToolSearch { .. } => "tool_search",
        ToolSpec::LocalShell {} => "local_shell",
        ToolSpec::Custom { name, .. } => name,
        ToolSpec::ImageGeneration { .. } => "image_generation",
    }
}

fn convert_response_format(format: Option<&Value>) -> Option<TextControls> {
    let map = format?.as_object()?;
    if map.get("type").and_then(Value::as_str) != Some("json_schema") {
        return None;
    }
    let schema = map.get("json_schema")?.as_object()?;
    Some(TextControls {
        verbosity: None,
        format: Some(TextFormat {
            kind: "json_schema".to_string(),
            strict: schema
                .get("strict")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            schema: schema.get("schema").cloned().unwrap_or_else(|| json!({})),
            name: schema
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("response")
                .to_string(),
        }),
    })
}

pub enum ChatSseEvent {
    Data(Value),
    Done,
}

impl ChatSseEvent {
    pub fn to_sse_data(&self) -> String {
        match self {
            Self::Data(value) => value.to_string(),
            Self::Done => "[DONE]".to_string(),
        }
    }
}

pub struct ChatCompletionStreamConverter {
    id: Option<String>,
    model: String,
    created: i64,
    include_usage: bool,
    role_sent: bool,
    emitted_tool_calls: HashMap<String, usize>,
    pending_tool_arguments: HashMap<String, String>,
    /// When true, drop `reasoning_content` deltas: reasoning was FORCED by
    /// family injection (e.g. Kimi `thinking=true`) for a Chat client that did
    /// NOT request it, and must not leak server-side internals (G2, Finding 2).
    /// Normal Chat reasoning (client asked, or non-forced) is unaffected.
    suppress_reasoning: bool,
    /// Suppressed reasoning buffered for terminal promotion (G8 for Chat): a
    /// backend that puts the ANSWER in the reasoning channel would otherwise
    /// yield a visibly empty turn for a suppressed client. Buffered only while
    /// no real content has started; cleared at a web-search boundary
    /// (pre-search reasoning is a genuine CoT preface, never the answer).
    suppressed_reasoning: String,
    /// Whether any visible `content` delta was emitted. Gates promotion (a
    /// turn with real text keeps its reasoning suppressed as usual).
    content_started: bool,
    /// A reasoning signature marks genuine chain-of-thought — never promoted.
    reasoning_signed: bool,
}

impl ChatCompletionStreamConverter {
    pub fn new(model: String, include_usage: bool) -> Self {
        Self::with_reasoning_suppression(model, include_usage, false)
    }

    pub fn with_reasoning_suppression(
        model: String,
        include_usage: bool,
        suppress_reasoning: bool,
    ) -> Self {
        Self {
            id: None,
            model,
            created: current_unix_timestamp(),
            include_usage,
            role_sent: false,
            emitted_tool_calls: HashMap::new(),
            pending_tool_arguments: HashMap::new(),
            suppress_reasoning,
            suppressed_reasoning: String::new(),
            content_started: false,
            reasoning_signed: false,
        }
    }

    pub fn convert(&mut self, event: &SseEvent) -> Vec<ChatSseEvent> {
        let mut output = Vec::new();
        match event.event.as_str() {
            "response.created" => {
                self.id = event
                    .data
                    .get("response")
                    .and_then(|response| response.get("id"))
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                self.ensure_role_chunk(&mut output);
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.data.get("delta").and_then(Value::as_str) {
                    self.content_started = true;
                    // Real text arrived: earlier reasoning was a genuine CoT
                    // preface, so it stays suppressed (never promoted).
                    self.suppressed_reasoning.clear();
                    self.ensure_role_chunk(&mut output);
                    output.push(ChatSseEvent::Data(self.chunk(vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            content: Some(delta.to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                    }])));
                }
            }
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                // Suppress forced-but-unrequested reasoning so it does not leak
                // to a Chat client (G2, Finding 2). Normal reasoning still flows.
                if self.suppress_reasoning {
                    // Buffer instead of discarding (G8 for Chat): if the turn
                    // ends with NO visible output on a clean stop, the backend
                    // put the answer here and the terminal handler promotes it
                    // to `content`. Late reasoning (after text began) stays
                    // dropped, mirroring the Anthropic converter.
                    if !self.content_started
                        && let Some(delta) = event.data.get("delta").and_then(Value::as_str)
                    {
                        self.suppressed_reasoning.push_str(delta);
                    }
                    return output;
                }
                if let Some(delta) = event.data.get("delta").and_then(Value::as_str) {
                    self.ensure_role_chunk(&mut output);
                    output.push(ChatSseEvent::Data(self.chunk(vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            reasoning_content: Some(delta.to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                    }])));
                }
            }
            "response.function_call_arguments.delta" => {
                if let (Some(call_id), Some(delta)) = (
                    event.data.get("call_id").and_then(Value::as_str),
                    event.data.get("delta").and_then(Value::as_str),
                ) {
                    self.pending_tool_arguments
                        .entry(call_id.to_string())
                        .or_default()
                        .push_str(delta);
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(tool_call) = tool_call_from_function_args_done(&event.data) {
                    self.emit_tool_call(tool_call, &mut output);
                }
            }
            "response.output_item.done" => {
                if let Some(tool_call) = tool_call_from_output_item_done(&event.data) {
                    self.emit_tool_call(tool_call, &mut output);
                }
            }
            "response.reasoning_summary_text.signature_delta" => {
                // A signature marks genuine chain-of-thought: pin the buffer to
                // suppression (never promoted to visible content).
                self.reasoning_signed = true;
            }
            "response.web_search_results" => {
                // A server-side web-search boundary: reasoning buffered so far
                // was the pre-search CoT preface, not the answer. Drop it so
                // only a post-search reasoning-only continuation can promote.
                self.suppressed_reasoning.clear();
            }
            "response.completed" | "response.incomplete" => {
                self.ensure_role_chunk(&mut output);
                self.promote_suppressed_reasoning(event, &mut output);
                let finish_reason = finish_reason_from_response(&event.data, self.has_tool_calls());
                output.push(ChatSseEvent::Data(self.chunk(vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta::default(),
                    finish_reason: Some(finish_reason),
                }])));
                if self.include_usage
                    && let Some(usage) = usage_from_response(&event.data)
                {
                    output.push(ChatSseEvent::Data(self.usage_chunk(usage)));
                }
                output.push(ChatSseEvent::Done);
            }
            "response.failed" => {
                output.push(ChatSseEvent::Data(json!({
                    "error": {
                        "message": response_error_message(&event.data),
                        // The OpenAI Chat error object carries a structured `code`
                        // — surface the canonical terminal's code (e.g. E1's
                        // `invalid_tool_call`) rather than dropping it.
                        "code": response_error_code(&event.data),
                    }
                })));
                output.push(ChatSseEvent::Done);
            }
            _ => {}
        }
        output
    }

    /// G8 promotion for Chat (mirrors the Anthropic converter's terminal
    /// matrix): when suppressed reasoning was the ONLY output this turn, the
    /// backend put the answer in the reasoning channel — promote it to a
    /// visible `content` delta, but ONLY on a CLEAN STOP (typed
    /// `terminal_reason: stop`, falling back to a literal `response.completed`
    /// when untagged), with no content started, no tool calls, and no
    /// signature. Every other shape keeps the buffer suppressed.
    fn promote_suppressed_reasoning(&mut self, event: &SseEvent, output: &mut Vec<ChatSseEvent>) {
        if self.suppressed_reasoning.is_empty() {
            return;
        }
        let promoted = std::mem::take(&mut self.suppressed_reasoning);
        let clean_stop = event
            .data
            .get("response")
            .and_then(TerminalReason::from_resource_value)
            .map(TerminalReason::is_clean_stop)
            .unwrap_or(event.event == "response.completed");
        if !clean_stop || self.content_started || self.has_tool_calls() || self.reasoning_signed {
            return;
        }
        self.content_started = true;
        output.push(ChatSseEvent::Data(self.chunk(vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                content: Some(promoted),
                ..Default::default()
            },
            finish_reason: None,
        }])));
    }

    fn ensure_role_chunk(&mut self, output: &mut Vec<ChatSseEvent>) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push(ChatSseEvent::Data(self.chunk(vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: Some("assistant"),
                ..Default::default()
            },
            finish_reason: None,
        }])));
    }

    fn emit_tool_call(&mut self, mut tool_call: ChatToolCall, output: &mut Vec<ChatSseEvent>) {
        let Some(call_id) = tool_call.id.clone() else {
            return;
        };
        if self.emitted_tool_calls.contains_key(&call_id) {
            return;
        }
        if tool_call.function.arguments.is_none()
            && let Some(arguments) = self.pending_tool_arguments.remove(&call_id)
        {
            tool_call.function.arguments = Some(Value::String(arguments));
        }
        let next_index = self.emitted_tool_calls.len();
        let index = *self.emitted_tool_calls.entry(call_id).or_insert(next_index);
        tool_call.index = Some(index);
        self.ensure_role_chunk(output);
        output.push(ChatSseEvent::Data(self.chunk(vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                tool_calls: Some(vec![tool_call]),
                ..Default::default()
            },
            finish_reason: None,
        }])));
    }

    fn has_tool_calls(&self) -> bool {
        !self.emitted_tool_calls.is_empty()
    }

    fn chunk(&self, choices: Vec<ChatStreamChoice>) -> Value {
        serde_json::to_value(ChatStreamChunk {
            id: self.chat_id(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices,
            usage: None,
        })
        .unwrap_or(Value::Null)
    }

    fn usage_chunk(&self, usage: ChatUsage) -> Value {
        serde_json::to_value(ChatStreamChunk {
            id: self.chat_id(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: Vec::new(),
            usage: Some(usage),
        })
        .unwrap_or(Value::Null)
    }

    fn chat_id(&self) -> String {
        self.id
            .clone()
            .unwrap_or_else(|| format!("chatcmpl_{}", Uuid::new_v4().simple()))
    }
}

pub struct ChatCompletionCollector {
    model: String,
    final_response: Option<Value>,
    error: Option<String>,
    /// See `ChatCompletionStreamConverter::suppress_reasoning` (G2, Finding 2):
    /// drop forced-but-unrequested `reasoning_content` from the assembled
    /// non-streaming Chat response.
    suppress_reasoning: bool,
}

impl ChatCompletionCollector {
    pub fn new(model: String) -> Self {
        Self::with_reasoning_suppression(model, false)
    }

    pub fn with_reasoning_suppression(model: String, suppress_reasoning: bool) -> Self {
        Self {
            model,
            final_response: None,
            error: None,
            suppress_reasoning,
        }
    }

    pub fn process(&mut self, event: &SseEvent) {
        match event.event.as_str() {
            "response.completed" | "response.incomplete" => {
                self.final_response = event.data.get("response").cloned();
            }
            "response.failed" => {
                self.error = Some(response_error_message(&event.data));
            }
            _ => {}
        }
    }

    pub fn into_response(self) -> AppResult<Value> {
        if let Some(error) = self.error {
            return Err(AppError::upstream(error));
        }
        let response = self.final_response.ok_or_else(|| {
            AppError::upstream("stream ended before a final response resource was emitted")
        })?;
        Ok(chat_completion_response_from_response(
            &self.model,
            &response,
            self.suppress_reasoning,
        ))
    }
}

fn chat_completion_response_from_response(
    model_hint: &str,
    response: &Value,
    suppress_reasoning: bool,
) -> Value {
    let (mut content, mut reasoning_content, tool_calls) = message_from_response_output(response);
    if suppress_reasoning {
        // G8 promotion for Chat (see `promote_suppressed_reasoning`): a
        // reasoning-only clean stop means the backend put the answer in the
        // reasoning channel — surface it as `content` instead of returning a
        // visibly empty message. Everything else stays suppressed.
        if content.is_empty()
            && tool_calls.is_empty()
            && response_is_clean_stop(response)
            && let Some(promoted) = promotable_reasoning_text(response)
        {
            content = promoted;
        }
        reasoning_content.clear();
    }
    let has_tool_calls = !tool_calls.is_empty();
    let message = ChatCompletionMessage {
        role: "assistant",
        content: if content.is_empty() && has_tool_calls {
            None
        } else {
            Some(content)
        },
        reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
        tool_calls: has_tool_calls.then_some(tool_calls),
    };
    let finish_reason = finish_reason_from_response_value(response, has_tool_calls);
    let created = response
        .get("created_at")
        .and_then(Value::as_i64)
        .unwrap_or_else(current_unix_timestamp);
    let model = response
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(model_hint)
        .to_string();
    let id = response
        .get("id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("chatcmpl_{}", Uuid::new_v4().simple()));
    serde_json::to_value(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: response_value_usage(response),
    })
    .unwrap_or(Value::Null)
}

fn message_from_response_output(response: &Value) -> (String, String, Vec<ChatToolCall>) {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return (content, reasoning, tool_calls);
    };
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => content.push_str(&message_item_text(item)),
            Some("reasoning") => reasoning.push_str(&reasoning_item_text(item)),
            Some("function_call")
            | Some("custom_tool_call")
            | Some("tool_search_call")
            | Some("local_shell_call") => {
                if let Some(tool_call) = tool_call_from_item(item) {
                    tool_calls.push(tool_call);
                }
            }
            _ => {}
        }
    }
    for (index, tool_call) in tool_calls.iter_mut().enumerate() {
        tool_call.index = Some(index);
    }
    (content, reasoning, tool_calls)
}

/// Clean-stop gate for the non-streaming promotion: the typed
/// `terminal_reason` when the engine tagged the resource, else fall back to a
/// `completed` status (mirrors the streaming converters' T7 fallback).
fn response_is_clean_stop(response: &Value) -> bool {
    TerminalReason::from_resource_value(response)
        .map(TerminalReason::is_clean_stop)
        .unwrap_or(response.get("status").and_then(Value::as_str) == Some("completed"))
}

/// The trailing run of unsigned reasoning items — the only promotable shape.
/// Any `message` item means real content started (never promote); a tool or
/// web-search item resets the run (pre-boundary reasoning is a CoT preface,
/// not the answer); a signature (`encrypted_content`) pins genuine
/// chain-of-thought to suppression.
fn promotable_reasoning_text(response: &Value) -> Option<String> {
    let output = response.get("output").and_then(Value::as_array)?;
    let mut trailing: Vec<&Value> = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => trailing.push(item),
            Some("message") => return None,
            _ => trailing.clear(),
        }
    }
    if trailing.iter().any(|item| {
        item.get("encrypted_content")
            .and_then(Value::as_str)
            .is_some()
    }) {
        return None;
    }
    let text = trailing
        .iter()
        .map(|item| reasoning_item_text(item))
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn message_item_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .map(|content| {
            content
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn reasoning_item_text(item: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        parts.extend(
            summary
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .map(ToString::to_string),
        );
    }
    if let Some(content) = item.get("content").and_then(Value::as_array) {
        parts.extend(
            content
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .map(ToString::to_string),
        );
    }
    parts.join("\n")
}

fn tool_call_from_function_args_done(data: &Value) -> Option<ChatToolCall> {
    let call_id = data.get("call_id")?.as_str()?.to_string();
    let name = data.get("name")?.as_str()?.to_string();
    // NOTE (round-7 #2): we do NOT hide `analyzeImage` by name here. On an active
    // image-agent turn the engine classifies it as the server-side ImageAnalysis
    // tool and never emits its delta/done/item events to the stream at all, so
    // this converter never sees them. On an INACTIVE turn `analyzeImage` is a
    // legitimate CLIENT tool and must flow through normally — a name-only hide
    // would wrongly swallow it.
    let arguments = data
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}")
        .to_string();
    Some(chat_tool_call(call_id, name, arguments))
}

fn tool_call_from_output_item_done(data: &Value) -> Option<ChatToolCall> {
    let item = data.get("item")?;
    match item.get("type").and_then(Value::as_str) {
        Some("web_search_call") => None,
        Some("function_call")
        | Some("custom_tool_call")
        | Some("tool_search_call")
        | Some("local_shell_call") => tool_call_from_item(item),
        _ => None,
    }
}

fn tool_call_from_item(item: &Value) -> Option<ChatToolCall> {
    match item.get("type").and_then(Value::as_str)? {
        "function_call" => {
            let call_id = item.get("call_id")?.as_str()?.to_string();
            let name = item.get("name")?.as_str()?.to_string();
            let arguments = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_string();
            Some(chat_tool_call(call_id, name, arguments))
        }
        "custom_tool_call" => {
            let call_id = item.get("call_id")?.as_str()?.to_string();
            let name = item.get("name")?.as_str()?.to_string();
            let input = item.get("input").and_then(Value::as_str).unwrap_or("");
            Some(chat_tool_call(
                call_id,
                name,
                json!({ "input": input }).to_string(),
            ))
        }
        "tool_search_call" => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("tool_search")
                .to_string();
            let arguments = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
            Some(chat_tool_call(
                call_id,
                "tool_search".to_string(),
                arguments.to_string(),
            ))
        }
        "local_shell_call" => {
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)?
                .to_string();
            let arguments = item.get("action").cloned().unwrap_or_else(|| json!({}));
            Some(chat_tool_call(
                call_id,
                "local_shell".to_string(),
                arguments.to_string(),
            ))
        }
        _ => None,
    }
}

fn chat_tool_call(call_id: String, name: String, arguments: String) -> ChatToolCall {
    ChatToolCall {
        id: Some(call_id),
        index: None,
        kind: "function".to_string(),
        function: ChatFunctionCall {
            name: Some(name),
            arguments: Some(Value::String(arguments)),
        },
    }
}

fn finish_reason_from_response(data: &Value, has_tool_calls: bool) -> String {
    data.get("response")
        .map(|response| finish_reason_from_response_value(response, has_tool_calls))
        .unwrap_or_else(|| {
            if has_tool_calls {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            }
        })
}

fn finish_reason_from_response_value(response: &Value, has_tool_calls: bool) -> String {
    if response.get("status").and_then(Value::as_str) == Some("incomplete") {
        return "length".to_string();
    }
    if response
        .get("incomplete_details")
        .and_then(|details| details.get("reason"))
        .is_some()
    {
        return "length".to_string();
    }
    if has_tool_calls {
        "tool_calls".to_string()
    } else {
        "stop".to_string()
    }
}

fn usage_from_response(data: &Value) -> Option<ChatUsage> {
    data.get("response").and_then(response_value_usage)
}

fn response_value_usage(response: &Value) -> Option<ChatUsage> {
    let usage = response.get("usage")?;
    let prompt_tokens = usage.get("input_tokens")?.as_i64()?;
    let completion_tokens = usage.get("output_tokens")?.as_i64()?;
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(prompt_tokens + completion_tokens);
    Some(ChatUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_details: usage
            .get("input_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_i64)
            .map(|cached_tokens| ChatPromptTokensDetails { cached_tokens }),
        completion_tokens_details: usage
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_i64)
            .map(|reasoning_tokens| ChatCompletionTokensDetails { reasoning_tokens }),
    })
}

fn response_error_message(data: &Value) -> String {
    data.get("response")
        .and_then(|response| response.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("upstream request failed")
        .to_string()
}

/// Structured machine `code` from a canonical `response.failed` event, mirroring
/// [`response_error_message`]. The OpenAI Chat error object HAS a `code` field, so
/// a structured terminal (e.g. E1's `invalid_tool_call` on repair exhaustion) is
/// surfaced to Chat clients, not just the human message. Defaults to
/// `gateway_error` (matching the engine's `failure_event` default) when absent.
fn response_error_code(data: &Value) -> String {
    data.get("response")
        .and_then(|response| response.get("error"))
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("gateway_error")
        .to_string()
}

fn current_unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ChatUsage>,
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatCompletionMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatCompletionMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Serialize)]
struct ChatStreamChunk {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ChatUsage>,
}

#[derive(Serialize)]
struct ChatStreamChoice {
    index: usize,
    delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Default, Serialize)]
struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Clone, Serialize)]
struct ChatUsage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<ChatPromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<ChatCompletionTokensDetails>,
}

#[derive(Clone, Serialize)]
struct ChatPromptTokensDetails {
    cached_tokens: i64,
}

#[derive(Clone, Serialize)]
struct ChatCompletionTokensDetails {
    reasoning_tokens: i64,
}

#[cfg(test)]
mod tests {
    use super::convert_request;
    use crate::models::chat::ChatCompletionRequest;
    use crate::models::chat::ChatMessage;
    use serde_json::Value;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn convert_request_preserves_extra_body() {
        let extra_body = BTreeMap::from([(
            "chat_template_kwargs".to_string(),
            json!({
                "thinking": true,
                "preserve_thinking": false
            }),
        )]);
        let request = ChatCompletionRequest {
            model: "glm-5.1".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String("hello".to_string())),
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                thinking: None,
                tool_calls: None,
            }],
            stream: true,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning_effort: None,
            response_format: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            extra_body: extra_body.clone(),
        };

        let converted = convert_request(request).expect("convert request");

        assert_eq!(converted.extra_body, extra_body);
    }

    // --- G2 Finding 2: forced-but-unrequested reasoning suppression (Chat) ---

    use super::ChatCompletionCollector;
    use super::ChatCompletionStreamConverter;
    use super::ChatSseEvent;
    use crate::engine::SseEvent;

    fn reasoning_delta_event() -> SseEvent {
        SseEvent {
            event: "response.reasoning_text.delta".to_string(),
            data: json!({ "delta": "secret chain of thought" }),
        }
    }

    fn text_delta_event() -> SseEvent {
        SseEvent {
            event: "response.output_text.delta".to_string(),
            data: json!({ "delta": "visible answer" }),
        }
    }

    /// Collect the streamed Chat deltas' `reasoning_content` strings.
    fn streamed_reasoning(suppress: bool) -> Vec<String> {
        let mut converter = ChatCompletionStreamConverter::with_reasoning_suppression(
            "kimi-k2".to_string(),
            false,
            suppress,
        );
        let mut out = Vec::new();
        for event in [reasoning_delta_event(), text_delta_event()] {
            for chat_event in converter.convert(&event) {
                if let ChatSseEvent::Data(value) = chat_event
                    && let Some(reasoning) =
                        value["choices"][0]["delta"]["reasoning_content"].as_str()
                {
                    out.push(reasoning.to_string());
                }
            }
        }
        out
    }

    /// Forced-but-unrequested reasoning is dropped from the streamed Chat output.
    #[test]
    fn stream_suppresses_forced_unrequested_reasoning() {
        assert!(
            streamed_reasoning(true).is_empty(),
            "forced reasoning must not surface to a Chat client that did not ask"
        );
    }

    /// When NOT suppressed (client requested, or non-forced), reasoning still
    /// surfaces exactly as today — normal Chat behavior is unchanged.
    #[test]
    fn stream_keeps_reasoning_when_not_suppressed() {
        assert_eq!(streamed_reasoning(false), vec!["secret chain of thought"]);
    }

    fn response_with_reasoning() -> SseEvent {
        SseEvent {
            event: "response.completed".to_string(),
            data: json!({
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "output": [
                        { "type": "reasoning", "content": [{ "type": "reasoning_text", "text": "hidden" }] },
                        { "type": "message", "content": [{ "type": "output_text", "text": "answer" }] }
                    ]
                }
            }),
        }
    }

    fn collected_reasoning(suppress: bool) -> Option<String> {
        let mut collector =
            ChatCompletionCollector::with_reasoning_suppression("kimi-k2".to_string(), suppress);
        collector.process(&response_with_reasoning());
        let value = collector.into_response().expect("response");
        value["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .map(ToString::to_string)
    }

    /// Non-streaming Chat: suppression drops `reasoning_content`; content stays.
    #[test]
    fn collected_suppresses_forced_unrequested_reasoning() {
        assert_eq!(collected_reasoning(true), None);
    }

    #[test]
    fn collected_keeps_reasoning_when_not_suppressed() {
        assert_eq!(collected_reasoning(false), Some("hidden".to_string()));
    }

    // --- G8 for Chat: reasoning-only clean stop promotes to visible content ---

    fn completed_event(terminal_reason: &str) -> SseEvent {
        SseEvent {
            event: if terminal_reason == "length" {
                "response.incomplete".to_string()
            } else {
                "response.completed".to_string()
            },
            data: json!({
                "response": {
                    "id": "resp_1",
                    "status": if terminal_reason == "length" { "incomplete" } else { "completed" },
                    "terminal_reason": terminal_reason,
                    "output": []
                }
            }),
        }
    }

    /// Run a SUPPRESSED converter over `events` + the terminal event; collect
    /// the streamed visible `content` deltas.
    fn streamed_content_suppressed(events: Vec<SseEvent>, terminal_reason: &str) -> Vec<String> {
        let mut converter = ChatCompletionStreamConverter::with_reasoning_suppression(
            "kimi-k2".to_string(),
            false,
            true,
        );
        let mut out = Vec::new();
        for event in events.into_iter().chain([completed_event(terminal_reason)]) {
            for chat_event in converter.convert(&event) {
                if let ChatSseEvent::Data(value) = chat_event
                    && let Some(content) = value["choices"][0]["delta"]["content"].as_str()
                {
                    out.push(content.to_string());
                }
            }
        }
        out
    }

    /// A suppressed reasoning-only turn on a clean stop is the backend putting
    /// the ANSWER in the reasoning channel: promote it to visible content
    /// instead of serving a visibly empty message.
    #[test]
    fn stream_promotes_suppressed_reasoning_only_on_clean_stop() {
        assert_eq!(
            streamed_content_suppressed(vec![reasoning_delta_event()], "stop"),
            vec!["secret chain of thought"]
        );
    }

    /// A non-clean terminal (`length`) never promotes.
    #[test]
    fn stream_does_not_promote_on_incomplete() {
        assert!(streamed_content_suppressed(vec![reasoning_delta_event()], "length").is_empty());
    }

    /// Once real text streamed, the reasoning was a CoT preface: stays
    /// suppressed, and the visible content is only the real answer.
    #[test]
    fn stream_does_not_promote_after_content_started() {
        assert_eq!(
            streamed_content_suppressed(vec![reasoning_delta_event(), text_delta_event()], "stop"),
            vec!["visible answer"]
        );
    }

    /// Signed reasoning is genuine chain-of-thought — never promoted.
    #[test]
    fn stream_does_not_promote_signed_reasoning() {
        let signature = SseEvent {
            event: "response.reasoning_summary_text.signature_delta".to_string(),
            data: json!({ "signature": "sig_123" }),
        };
        assert!(
            streamed_content_suppressed(vec![reasoning_delta_event(), signature], "stop")
                .is_empty()
        );
    }

    /// A server-side web-search boundary drops the pre-search CoT preface;
    /// only the post-search reasoning-only continuation promotes.
    #[test]
    fn stream_web_search_boundary_keeps_only_post_search_reasoning() {
        let post_search = SseEvent {
            event: "response.reasoning_text.delta".to_string(),
            data: json!({ "delta": "the searched answer" }),
        };
        let boundary = SseEvent {
            event: "response.web_search_results".to_string(),
            data: json!({ "tool_use_id": "ws_1", "query": "q", "results": [] }),
        };
        assert_eq!(
            streamed_content_suppressed(
                vec![reasoning_delta_event(), boundary, post_search],
                "stop"
            ),
            vec!["the searched answer"]
        );
    }

    fn collected_content_suppressed(output: Value, terminal_reason: &str) -> Option<String> {
        let mut collector =
            ChatCompletionCollector::with_reasoning_suppression("kimi-k2".to_string(), true);
        collector.process(&SseEvent {
            event: "response.completed".to_string(),
            data: json!({
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "terminal_reason": terminal_reason,
                    "output": output
                }
            }),
        });
        let value = collector.into_response().expect("response");
        value["choices"][0]["message"]["content"]
            .as_str()
            .map(ToString::to_string)
    }

    /// Non-streaming mirror of the streaming promotion.
    #[test]
    fn collected_promotes_reasoning_only_on_clean_stop() {
        let output = json!([
            { "type": "reasoning", "content": [{ "type": "reasoning_text", "text": "the answer" }] }
        ]);
        assert_eq!(
            collected_content_suppressed(output, "stop"),
            Some("the answer".to_string())
        );
    }

    /// Signed (`encrypted_content`) reasoning never promotes non-streaming.
    #[test]
    fn collected_does_not_promote_signed_reasoning() {
        let output = json!([
            {
                "type": "reasoning",
                "encrypted_content": "sig_123",
                "content": [{ "type": "reasoning_text", "text": "genuine cot" }]
            }
        ]);
        assert_eq!(
            collected_content_suppressed(output, "stop"),
            Some(String::new())
        );
    }
}
