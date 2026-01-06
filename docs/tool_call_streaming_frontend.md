# Tool Call Response Streaming - Frontend Guide

## Overview

The `PydanticMultiAgent` streams tool call responses in real-time, allowing the frontend to display tool execution progress as it happens. This document describes the data structure and how to consume and display these streaming updates.

## Data Structure

### ChatAgentResponse

Each streaming chunk follows this structure:

```typescript
interface ChatAgentResponse {
  response: string; // Text content from agent (may be empty for tool-only updates)
  tool_calls: ToolCallResponse[]; // Array of tool call objects
  citations: string[]; // File citations
}
```

### ToolCallResponse

```typescript
interface ToolCallResponse {
  call_id: string; // Unique identifier for this tool call (used to track streaming updates)
  event_type: "call" | "result" | "delegation_call" | "delegation_result";
  tool_name: string; // Name of the tool (e.g., "fetch_file", "delegate_to_think_execute")
  tool_response: string; // User-friendly status message (e.g., "Fetching file: path/to/file")
  tool_call_details: {
    summary: string; // Detailed information about the tool call/result (markdown-formatted)
    [key: string]: any; // Additional metadata
  };
  stream_part?: string; // Partial content for streaming updates (only present when streaming)
  is_complete: boolean; // Whether this is the final part (default: true, false for streaming parts)
}
```

## Event Types

### `call` - Tool Call Initiated

- Emitted when a tool starts executing
- `tool_response`: User-friendly message from `get_tool_run_message()` (e.g., "Fetching file: path/to/file")
- `tool_call_details.summary`: Detailed info from `get_tool_call_info_content()` (e.g., "-> fetching contents for file path/to/file")

### `result` - Tool Call Completed

- Emitted when a regular tool finishes
- `tool_response`: Completion message from `get_tool_response_message()` (e.g., "File content fetched successfully")
- `tool_call_details.summary`: Result details from `get_tool_result_info_content()` (may include code snippets, formatted output, etc.)

### `delegation_call` - Subagent Delegation Started

- Emitted when supervisor delegates to a subagent
- `tool_response`: Delegation message from `get_delegation_call_message()` (e.g., "🚀 Starting subagent with full tool access...")
- `tool_call_details.summary`: Detailed delegation info from `get_delegation_info_content()` (includes task description, context, agent type)

### `delegation_result` - Subagent Completed

- Emitted when subagent finishes and returns result
- `tool_response`: Completion message from `get_delegation_response_message()` (e.g., "✅ Subagent completed - returning task result to supervisor")
- `tool_call_details.summary`: Full result from `get_delegation_result_content()` (includes complete task result, may contain code snippets)

## Streaming Behavior

### Non-Streaming Tool Calls (Standard)

For regular tool calls:

- Single chunk with `is_complete: true`
- `stream_part` is `undefined` or `null`
- `tool_response` contains the complete status message
- `tool_call_details.summary` contains complete result information

### Streaming Tool Calls (Delegation Results)

For streaming tool calls (typically delegation results):

- **Multiple chunks** arrive with the **same `call_id`**
- Each chunk has `stream_part` with a portion of the content
- `is_complete: false` for partial updates
- `is_complete: true` for the final chunk
- `tool_response` may be empty or partial until the final chunk
- `tool_call_details.summary` accumulates across chunks (final chunk has complete summary)

## Frontend Implementation Pattern

### 1. State Management

Track tool calls by `call_id` and accumulate stream parts:

```typescript
interface ToolCallState {
  call_id: string;
  event_type: string;
  tool_name: string;
  tool_response: string; // Status message
  summary: string; // Accumulated from tool_call_details.summary
  accumulated_stream: string; // Accumulated from stream_part chunks
  is_complete: boolean;
  is_streaming: boolean;
}

const toolCallsMap = new Map<string, ToolCallState>();
```

### 2. Processing Stream Chunks

```typescript
function processStreamChunk(response: ChatAgentResponse) {
  // Process text message content
  if (response.message) {
    appendMessageContent(response.message);
  }

  // Process tool calls
  if (response.tool_calls && response.tool_calls.length > 0) {
    response.tool_calls.forEach((toolCall: ToolCallResponse) => {
      const {
        call_id,
        stream_part,
        is_complete,
        tool_response,
        tool_call_details,
      } = toolCall;

      if (stream_part !== undefined && stream_part !== null) {
        // This is a streaming update
        updateStreamingToolCall(call_id, toolCall, stream_part, is_complete);
      } else {
        // Regular tool call (complete)
        addCompleteToolCall(call_id, toolCall);
      }
    });
  }

  // Process citations
  if (response.citations && response.citations.length > 0) {
    addCitations(response.citations);
  }
}

function updateStreamingToolCall(
  callId: string,
  toolCall: ToolCallResponse,
  streamPart: string,
  isComplete: boolean
) {
  const existing = toolCallsMap.get(callId);

  if (existing) {
    // Update existing tool call
    existing.accumulated_stream += streamPart;
    existing.summary = toolCall.tool_call_details.summary || existing.summary;
    existing.tool_response = toolCall.tool_response || existing.tool_response;
    existing.is_complete = isComplete;
    existing.is_streaming = !isComplete;
  } else {
    // Create new tool call entry
    toolCallsMap.set(callId, {
      call_id: callId,
      event_type: toolCall.event_type,
      tool_name: toolCall.tool_name,
      tool_response: toolCall.tool_response,
      summary: toolCall.tool_call_details.summary || "",
      accumulated_stream: streamPart,
      is_complete: isComplete,
      is_streaming: !isComplete,
    });
  }

  // Trigger UI update
  updateToolCallDisplay(callId);
}

function addCompleteToolCall(callId: string, toolCall: ToolCallResponse) {
  toolCallsMap.set(callId, {
    call_id: callId,
    event_type: toolCall.event_type,
    tool_name: toolCall.tool_name,
    tool_response: toolCall.tool_response,
    summary: toolCall.tool_call_details.summary || "",
    accumulated_stream: "",
    is_complete: true,
    is_streaming: false,
  });

  updateToolCallDisplay(callId);
}
```

### 3. Displaying Tool Calls

```typescript
function renderToolCall(callId: string) {
  const toolCall = toolCallsMap.get(callId);
  if (!toolCall) return null;

  const {
    event_type,
    tool_name,
    tool_response,
    summary,
    accumulated_stream,
    is_complete,
    is_streaming,
  } = toolCall;

  return (
    <ToolCallCard>
      {/* Status indicator */}
      <StatusBadge eventType={event_type} isStreaming={is_streaming} />

      {/* Tool name and status message */}
      <ToolHeader>
        <ToolName>{tool_name}</ToolName>
        <ToolStatus>{tool_response}</ToolStatus>
      </ToolHeader>

      {/* Detailed summary (markdown-formatted) */}
      {summary && (
        <ToolSummary>
          <MarkdownRenderer content={summary} />
        </ToolSummary>
      )}

      {/* Streaming content (for delegation results) */}
      {is_streaming && accumulated_stream && (
        <StreamingContent>
          <StreamingIndicator />
          <MarkdownRenderer content={accumulated_stream} />
        </StreamingContent>
      )}

      {/* Complete indicator */}
      {is_complete && !is_streaming && (
        <CompleteIndicator>✓ Complete</CompleteIndicator>
      )}
    </ToolCallCard>
  );
}
```

## Tool Helper Functions Reference

The backend uses helper functions from `tool_helpers.py` to generate user-friendly messages:

### For Regular Tools

- **`get_tool_run_message(tool_name, args)`**: Returns status message when tool starts

  - Example: `"Fetching file: path/to/file"` for `fetch_file`
  - Example: `"Running: git status"` for `bash_command`

- **`get_tool_response_message(tool_name)`**: Returns completion message

  - Example: `"File content fetched successfully"` for `fetch_file`
  - Example: `"Bash command executed successfully"` for `bash_command`

- **`get_tool_call_info_content(tool_name, args)`**: Returns detailed info about what tool will do

  - Example: `"-> fetching contents for file path/to/file"` for `fetch_file`
  - Example: `"-> executing command: git status in directory '/path'"` for `bash_command`

- **`get_tool_result_info_content(tool_name, content)`**: Returns formatted result details
  - May include code snippets, formatted output, error messages
  - Example: For `bash_command`, includes exit code, output, and error messages

### For Delegation Tools

- **`get_delegation_call_message(agent_type)`**: Returns delegation start message

  - Example: `"🚀 Starting subagent with full tool access - streaming work in real-time..."`

- **`get_delegation_response_message(agent_type)`**: Returns delegation completion message

  - Example: `"✅ Subagent completed - returning task result to supervisor"`

- **`get_delegation_info_content(agent_type, task_description, context)`**: Returns detailed delegation info

  - Includes agent type, task description, and context preview
  - Example: `"🤖 **General Subagent**\n\n**Task:**\nFix the bug\n\n**Context Provided:**\n..."`

- **`get_delegation_result_content(agent_type, result)`**: Returns formatted subagent result
  - Includes complete task result with label
  - Example: `"**General Subagent Result:**\n\n[complete result content]"`

## Special Tool Names

### Delegation Tools

Tool names starting with `delegate_to_` indicate delegation:

- `delegate_to_think_execute`: General subagent with full tool access
- `delegate_to_jira`: Jira integration agent
- `delegate_to_github`: GitHub integration agent
- `delegate_to_confluence`: Confluence integration agent
- `delegate_to_linear`: Linear integration agent

The agent type is extracted by removing the `delegate_to_` prefix.

### Common Tool Names

See `tool_helpers.py` for the complete list. Common examples:

- `fetch_file`: Fetch file content
- `bash_command`: Execute bash command
- `GetCodeanddocstringFromProbableNodeName`: Retrieve code from knowledge graph
- `create_todo`, `update_todo_status`, etc.: Todo management
- `add_file_to_changes`, `update_file_in_changes`, etc.: Code changes management

## Best Practices

1. **Always track by `call_id`**: Use `call_id` as the key to track tool calls across multiple chunks
2. **Accumulate `stream_part`**: For streaming tool calls, append `stream_part` to previous content
3. **Update `summary` from `tool_call_details.summary`**: The summary may be updated in later chunks
4. **Handle markdown in `summary`**: The summary field contains markdown-formatted content
5. **Show loading indicators**: Display a loading/spinner indicator when `is_streaming: true`
6. **Display status messages**: Show `tool_response` as a user-friendly status indicator
7. **Handle errors gracefully**: Tool calls may fail; check for error indicators in the summary

## Example Flow

````
1. Tool Call Initiated:
   {
     call_id: "call_123",
     event_type: "call",
     tool_name: "fetch_file",
     tool_response: "Fetching file: app/main.py",
     tool_call_details: { summary: "-> fetching contents for file app/main.py" },
     is_complete: true
   }

2. Tool Call Result (if streaming):
   {
     call_id: "call_123",
     event_type: "result",
     tool_name: "fetch_file",
     tool_response: "File content fetched successfully",
     tool_call_details: { summary: "```python\n...code...\n```" },
     stream_part: "```python\n...",  // Partial content
     is_complete: false
   }

3. Tool Call Result (final chunk):
   {
     call_id: "call_123",
     event_type: "result",
     tool_name: "fetch_file",
     tool_response: "File content fetched successfully",
     tool_call_details: { summary: "```python\n...complete code...\n```" },
     stream_part: "...complete code...\n```",  // Final chunk
     is_complete: true
   }
````

## Related Documentation

- [Subagent Streaming API Guide](./subagent_streaming_api.md) - Complete guide for subagent text streaming and delegation
