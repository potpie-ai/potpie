# Frontend Guide: Consuming Streaming Tool Call Responses

> **Note**: For complete subagent streaming documentation including tool calls from subagents, see [Subagent Streaming API Guide](./subagent_streaming_api.md)

## Overview

Tool call responses can now be streamed in parts when using agent delegation (sub-agents). The backend sends partial updates for tool calls with the same `call_id`, allowing the frontend to display real-time progress.

**Important**: When a subagent makes tool calls (e.g., Confluence API calls), those tool calls and their results are also streamed to the frontend in real-time. See the [Subagent Streaming API Guide](./subagent_streaming_api.md) for complete details.

## Response Format

### ChatMessageResponse Structure

Each chunk in the SSE stream follows this format:

```typescript
interface ChatMessageResponse {
  message: string; // Text content (may be empty for tool-only updates)
  citations: string[]; // File citations
  tool_calls: ToolCallResponse[]; // Array of tool call objects
}
```

### ToolCallResponse Structure

```typescript
interface ToolCallResponse {
  call_id: string; // Unique identifier for this tool call
  event_type: "call" | "result" | "delegation_call" | "delegation_result";
  tool_name: string; // Name of the tool
  tool_response: string; // Full/complete response text
  tool_call_details: {
    // Additional metadata
    summary?: string;
    [key: string]: any;
  };
  stream_part?: string; // NEW: Partial content for this update (only present when streaming)
  is_complete: boolean; // NEW: Whether this is the final part (default: true)
}
```

## Streaming Behavior

### Normal Tool Calls (Non-Streaming)

For regular tool calls, you'll receive:

- `stream_part` is `undefined` or `null`
- `is_complete` is `true` (default)
- `tool_response` contains the complete response

### Streaming Tool Calls (Delegation/Sub-Agent)

For streaming tool calls (typically delegation results):

- Multiple chunks arrive with the **same `call_id`**
- Each chunk has `stream_part` with a portion of the content
- `is_complete` is `false` for partial updates
- `is_complete` is `true` for the final chunk
- `tool_response` may be empty or partial until the final chunk

## Frontend Implementation Pattern

### 1. State Management

Track tool calls by `call_id` and accumulate stream parts:

```typescript
interface ToolCallState {
  call_id: string;
  event_type: string;
  tool_name: string;
  accumulated_response: string; // Accumulated from stream_part chunks
  tool_response: string; // Final complete response
  tool_call_details: Record<string, any>;
  is_complete: boolean;
  is_streaming: boolean;
}

// In your component/store
const toolCallsMap = new Map<string, ToolCallState>();
```

### 2. Processing Stream Chunks

```typescript
function processStreamChunk(response: ChatMessageResponse) {
  // Process text message content
  if (response.message) {
    // Append to message buffer
    appendMessageContent(response.message);
  }

  // Process tool calls
  if (response.tool_calls && response.tool_calls.length > 0) {
    response.tool_calls.forEach((toolCall: ToolCallResponse) => {
      const { call_id, stream_part, is_complete, tool_response } = toolCall;

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
    existing.accumulated_response += streamPart;
    existing.is_complete = isComplete;
    existing.is_streaming = !isComplete;

    // Update final response if provided in this chunk
    if (toolCall.tool_response) {
      existing.tool_response = toolCall.tool_response;
    }
  } else {
    // Create new streaming tool call
    toolCallsMap.set(callId, {
      call_id: callId,
      event_type: toolCall.event_type,
      tool_name: toolCall.tool_name,
      accumulated_response: streamPart,
      tool_response: toolCall.tool_response || streamPart,
      tool_call_details: toolCall.tool_call_details || {},
      is_complete: isComplete,
      is_streaming: !isComplete,
    });
  }

  // Trigger UI update
  notifyToolCallUpdate(callId);
}

function addCompleteToolCall(callId: string, toolCall: ToolCallResponse) {
  toolCallsMap.set(callId, {
    call_id: callId,
    event_type: toolCall.event_type,
    tool_name: toolCall.tool_name,
    accumulated_response: toolCall.tool_response,
    tool_response: toolCall.tool_response,
    tool_call_details: toolCall.tool_call_details || {},
    is_complete: true,
    is_streaming: false,
  });

  notifyToolCallUpdate(callId);
}
```

### 3. Display Logic

```typescript
function renderToolCall(toolCall: ToolCallState) {
  const displayText = toolCall.is_complete
    ? toolCall.tool_response
    : toolCall.accumulated_response;

  return (
    <ToolCallComponent
      callId={toolCall.call_id}
      toolName={toolCall.tool_name}
      content={displayText}
      isStreaming={toolCall.is_streaming}
      eventType={toolCall.event_type}
      details={toolCall.tool_call_details}
    />
  );
}
```

### 4. Complete Example (React Hook)

```typescript
import { useState, useCallback, useRef } from "react";

interface ToolCallState {
  call_id: string;
  event_type: string;
  tool_name: string;
  content: string;
  tool_call_details: Record<string, any>;
  is_complete: boolean;
}

function useToolCallStream() {
  const [toolCalls, setToolCalls] = useState<Map<string, ToolCallState>>(
    new Map()
  );
  const toolCallsRef = useRef<Map<string, ToolCallState>>(new Map());

  const processChunk = useCallback((response: ChatMessageResponse) => {
    if (!response.tool_calls || response.tool_calls.length === 0) {
      return;
    }

    const updated = new Map(toolCallsRef.current);

    response.tool_calls.forEach((toolCall: ToolCallResponse) => {
      const {
        call_id,
        stream_part,
        is_complete,
        tool_response,
        tool_call_details,
      } = toolCall;

      if (stream_part !== undefined && stream_part !== null) {
        // Streaming update
        const existing = updated.get(call_id);
        if (existing) {
          updated.set(call_id, {
            ...existing,
            content: existing.content + stream_part,
            tool_response: tool_response || existing.content + stream_part,
            is_complete: is_complete || false,
          });
        } else {
          updated.set(call_id, {
            call_id,
            event_type: toolCall.event_type,
            tool_name: toolCall.tool_name,
            content: stream_part,
            tool_call_details: tool_call_details || {},
            is_complete: is_complete || false,
          });
        }
      } else {
        // Complete tool call
        updated.set(call_id, {
          call_id,
          event_type: toolCall.event_type,
          tool_name: toolCall.tool_name,
          content: tool_response,
          tool_call_details: tool_call_details || {},
          is_complete: true,
        });
      }
    });

    toolCallsRef.current = updated;
    setToolCalls(new Map(updated));
  }, []);

  return { toolCalls, processChunk };
}
```

## Event Types

- `"call"`: Initial tool call event (before execution)
- `"result"`: Tool execution result (non-delegation)
- `"delegation_call"`: Supervisor delegating to a sub-agent
- `"delegation_result"`: Sub-agent completion result (this is what streams)

## UI Considerations

1. **Visual Indicators**: Show a loading/spinner indicator when `is_streaming: true`
2. **Incremental Display**: Append `stream_part` content as it arrives
3. **Completion State**: When `is_complete: true`, finalize the display and remove loading indicators
4. **Error Handling**: Handle cases where streaming might be interrupted

## Example SSE Consumption

```typescript
async function consumeStream(conversationId: string, runId: string) {
  const eventSource = new EventSource(
    `/api/conversations/${conversationId}/stream?run_id=${runId}`
  );

  eventSource.onmessage = (event) => {
    const response: ChatMessageResponse = JSON.parse(event.data);
    processStreamChunk(response);
  };

  eventSource.onerror = (error) => {
    console.error("Stream error:", error);
    eventSource.close();
  };
}
```

## Notes

- Multiple tool calls can stream simultaneously (different `call_id`s)
- The same `call_id` will appear in multiple chunks during streaming
- Always check `stream_part` to determine if it's a streaming update
- Accumulate `stream_part` values until `is_complete: true`
- The final chunk may have both `stream_part` and complete `tool_response`
