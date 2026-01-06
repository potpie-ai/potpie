# Subagent Streaming API Guide

## Overview

When the supervisor agent delegates tasks to subagents (specialized agents like Confluence, GitHub, Jira, etc.), the subagent's **text responses** stream to the frontend in real-time.

**Important**: Subagent tool calls (e.g., Confluence API calls) execute silently in the background and are **not** streamed to the frontend. Only the subagent's text responses are streamed.

## Architecture

```
1. Supervisor delegates → Frontend receives delegation_call event
2. Subagent starts → Frontend receives "🤖 Subagent Starting..."
3. Subagent executes tools silently → (No events sent to frontend)
4. Subagent streams text → Frontend receives text chunks in real-time
5. Subagent completes → Frontend receives "✅ Subagent Complete"
6. Final result → Frontend receives delegation_result event
```

## API Endpoint

```
POST /api/conversations/{conversation_id}/message/
GET /api/conversations/{conversation_id}/stream?run_id={run_id}
```

## Response Format

### ChatMessageResponse Structure

Each chunk in the SSE (Server-Sent Events) stream follows this format:

```typescript
interface ChatMessageResponse {
  message: string; // Text content from agent/subagent
  citations: string[]; // File citations
  tool_calls: ToolCallResponse[]; // Array of tool call objects
}
```

**Note**: For subagent responses, `tool_calls` will be empty. Only supervisor tool calls (like delegation) appear in `tool_calls`.

### ToolCallResponse Structure

```typescript
interface ToolCallResponse {
  call_id: string; // Unique identifier for this tool call
  event_type: "call" | "result" | "delegation_call" | "delegation_result";
  tool_name: string; // Name of the tool
  tool_response: string; // Full/complete response text
  tool_call_details: {
    summary?: string;
    [key: string]: any;
  };
  stream_part?: string; // Partial content for streaming updates (delegation only)
  is_complete: boolean; // Whether this is the final part
}
```

## Streaming Flow

### 1. Supervisor Delegates to Subagent

When the supervisor calls a delegation tool (e.g., `delegate_to_confluence_agent`), you'll receive:

```json
{
  "message": "",
  "citations": [],
  "tool_calls": [
    {
      "call_id": "call_abc123",
      "event_type": "delegation_call",
      "tool_name": "delegate_to_confluence_agent",
      "tool_response": "📄 Starting Confluence integration agent...",
      "tool_call_details": {
        "summary": "**Confluence Integration Agent**\n\n**Task:**\nFetch documentation...",
        "task": "Fetch documentation...",
        "context": "Project: nndn/PR_Reviewer..."
      },
      "is_complete": true
    }
  ]
}
```

### 2. Subagent Starts Working

You'll see a visual indicator:

```json
{
  "message": "\n\n---\n🤖 **Subagent Starting...**\n\n",
  "citations": [],
  "tool_calls": []
}
```

### 3. Subagent Text Streaming

As the subagent thinks and responds, you'll receive **text chunks only**:

```json
{
  "message": "Starting task: Fetch project documentation...",
  "citations": [],
  "tool_calls": []
}
```

```json
{
  "message": " I'll search for Confluence spaces...",
  "citations": [],
  "tool_calls": []
}
```

```json
{
  "message": " Found 3 spaces. Now fetching pages...",
  "citations": [],
  "tool_calls": []
}
```

**Important**: Subagent tool calls (like `get_confluence_spaces`, `get_confluence_page`) execute silently. You won't see tool call events for them - only the text responses.

### 4. Subagent Completes

When the subagent finishes, you'll see:

```json
{
  "message": "\n\n---\n✅ **Subagent Complete**\n\n",
  "citations": [],
  "tool_calls": []
}
```

### 5. Final Delegation Result

Finally, you'll receive the delegation result:

```json
{
  "message": "",
  "citations": [],
  "tool_calls": [
    {
      "call_id": "call_abc123",
      "event_type": "delegation_result",
      "tool_name": "delegate_to_confluence_agent",
      "tool_response": "## Task Result\n\nFound 3 Confluence spaces and 15 pages...",
      "tool_call_details": {},
      "is_complete": true
    }
  ]
}
```

## Frontend Implementation

### Simple React Hook

```typescript
import { useState, useEffect, useRef } from "react";

interface ChatMessageResponse {
  message: string;
  citations: string[];
  tool_calls: ToolCallResponse[];
}

interface ToolCallResponse {
  call_id: string;
  event_type: "call" | "result" | "delegation_call" | "delegation_result";
  tool_name: string;
  tool_response: string;
  tool_call_details: Record<string, any>;
  stream_part?: string;
  is_complete: boolean;
}

export function useSubagentStream(conversationId: string, runId: string) {
  const [message, setMessage] = useState("");
  const [toolCalls, setToolCalls] = useState<ToolCallResponse[]>([]);
  const [citations, setCitations] = useState<string[]>([]);
  const messageRef = useRef("");

  useEffect(() => {
    const eventSource = new EventSource(
      `/api/conversations/${conversationId}/stream?run_id=${runId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const response: ChatMessageResponse = JSON.parse(event.data);
        processChunk(response);
      } catch (error) {
        console.error("Error parsing stream chunk:", error);
      }
    };

    eventSource.onerror = (error) => {
      console.error("Stream error:", error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [conversationId, runId]);

  const processChunk = (response: ChatMessageResponse) => {
    // Process text message (this is what subagents stream)
    if (response.message) {
      messageRef.current += response.message;
      setMessage(messageRef.current);
    }

    // Process citations
    if (response.citations && response.citations.length > 0) {
      setCitations((prev) => [...prev, ...response.citations]);
    }

    // Process tool calls (only supervisor tool calls, not subagent tool calls)
    if (response.tool_calls && response.tool_calls.length > 0) {
      // Update tool calls state
      setToolCalls((prev) => {
        const updated = [...prev];
        response.tool_calls.forEach((toolCall) => {
          const existingIndex = updated.findIndex(
            (tc) => tc.call_id === toolCall.call_id
          );
          if (existingIndex >= 0) {
            // Update existing tool call
            updated[existingIndex] = toolCall;
          } else {
            // Add new tool call
            updated.push(toolCall);
          }
        });
        return updated;
      });
    }
  };

  return {
    message,
    toolCalls,
    citations,
  };
}
```

### Usage in Component

```typescript
function ConversationView({ conversationId, runId }: Props) {
  const { message, toolCalls, citations } = useSubagentStream(
    conversationId,
    runId
  );

  return (
    <div>
      {/* Display message content (includes subagent text streaming) */}
      <div className="message-content">{message}</div>

      {/* Display tool calls (only supervisor tool calls like delegation) */}
      {toolCalls.map((toolCall) => (
        <ToolCallCard key={toolCall.call_id} toolCall={toolCall} />
      ))}

      {/* Display citations */}
      {citations.map((citation, idx) => (
        <Citation key={idx} file={citation} />
      ))}
    </div>
  );
}

function ToolCallCard({ toolCall }: { toolCall: ToolCallResponse }) {
  const isDelegation =
    toolCall.event_type === "delegation_call" ||
    toolCall.event_type === "delegation_result";

  return (
    <div className={`tool-call ${isDelegation ? "delegation" : ""}`}>
      <div className="tool-header">
        <span className="tool-name">{toolCall.tool_name}</span>
        {toolCall.is_complete && <CheckIcon />}
      </div>
      {toolCall.event_type === "delegation_result" && (
        <div className="tool-content">{toolCall.tool_response}</div>
      )}
    </div>
  );
}
```

## Key Points

1. **Text-Only Streaming**: Subagents only stream text responses. Tool calls execute silently.

2. **Tool Calls in Response**:

   - Supervisor tool calls (like `delegate_to_confluence_agent`) appear in `tool_calls`
   - Subagent tool calls do NOT appear in `tool_calls`
   - Only text from subagents appears in `message` field

3. **Streaming Updates**:

   - Text chunks arrive incrementally in the `message` field
   - Accumulate `message` values to build the complete response
   - Delegation results appear in `tool_calls` with `event_type: "delegation_result"`

4. **Simple State Management**:
   - Track `message` as a string (accumulate chunks)
   - Track `tool_calls` as an array (update by `call_id`)
   - Track `citations` as an array

## Example: Complete Flow

```
1. User: "Fetch Confluence docs for PR_Reviewer"
   ↓
2. Supervisor: Delegates to Confluence agent
   → Frontend receives: delegation_call event
   → Frontend receives: "🤖 Subagent Starting..."
   ↓
3. Subagent: Executes get_confluence_spaces (silently)
   → (No event sent to frontend)
   ↓
4. Subagent: Streams text response
   → Frontend receives: "Found 3 spaces. Now fetching pages..."
   ↓
5. Subagent: Executes get_confluence_space_pages (silently)
   → (No event sent to frontend)
   ↓
6. Subagent: Streams text response
   → Frontend receives: "Found 15 pages in Engineering space..."
   ↓
7. Subagent: Completes
   → Frontend receives: "✅ Subagent Complete"
   → Frontend receives: delegation_result event with final summary
```

## Complete Example (React Hook)

```typescript
import { useState, useEffect, useRef } from "react";

interface ChatMessageResponse {
  message: string;
  citations: string[];
  tool_calls: ToolCallResponse[];
}

interface ToolCallResponse {
  call_id: string;
  event_type: "call" | "result" | "delegation_call" | "delegation_result";
  tool_name: string;
  tool_response: string;
  tool_call_details: Record<string, any>;
  stream_part?: string;
  is_complete: boolean;
}

export function useSubagentStream(conversationId: string, runId: string) {
  const [message, setMessage] = useState("");
  const [toolCalls, setToolCalls] = useState<Map<string, ToolCallResponse>>(
    new Map()
  );
  const [citations, setCitations] = useState<string[]>([]);
  const messageRef = useRef("");

  useEffect(() => {
    if (!conversationId || !runId) return;

    const eventSource = new EventSource(
      `/api/conversations/${conversationId}/stream?run_id=${runId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const response: ChatMessageResponse = JSON.parse(event.data);
        processChunk(response);
      } catch (error) {
        console.error("Error parsing stream chunk:", error);
      }
    };

    eventSource.onerror = (error) => {
      console.error("Stream error:", error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [conversationId, runId]);

  const processChunk = (response: ChatMessageResponse) => {
    // Process text message (accumulate subagent text responses)
    if (response.message) {
      messageRef.current += response.message;
      setMessage(messageRef.current);
    }

    // Process citations
    if (response.citations && response.citations.length > 0) {
      setCitations((prev) => {
        const newCitations = [...prev, ...response.citations];
        // Deduplicate
        return Array.from(new Set(newCitations));
      });
    }

    // Process tool calls (only supervisor tool calls)
    if (response.tool_calls && response.tool_calls.length > 0) {
      setToolCalls((prev) => {
        const updated = new Map(prev);
        response.tool_calls.forEach((toolCall) => {
          updated.set(toolCall.call_id, toolCall);
        });
        return updated;
      });
    }
  };

  return {
    message,
    toolCalls: Array.from(toolCalls.values()),
    citations,
  };
}
```

## Event Types

- `"call"`: Initial tool call event (supervisor only)
- `"result"`: Tool execution result (supervisor only)
- `"delegation_call"`: Supervisor delegating to a sub-agent
- `"delegation_result"`: Sub-agent completion result

## UI Considerations

1. **Text Streaming**: Append `message` content as it arrives to show subagent progress
2. **Tool Calls**: Only show supervisor tool calls (delegation events)
3. **Visual Indicators**: Show loading state while subagent is working
4. **Completion State**: When `delegation_result` arrives, show final result

## Troubleshooting

### Stream Stops Unexpectedly

- Check network connection
- Verify the backend is still processing (check logs)
- The backend has a 5-minute timeout - if it times out, you'll receive a timeout message

### No Text Appearing

- Ensure you're accumulating `message` field from each chunk
- Check that `messageRef.current` is being updated
- Verify the stream is still connected

### Tool Calls Not Appearing

- Only supervisor tool calls appear in `tool_calls`
- Subagent tool calls are intentionally hidden
- Check for `delegation_call` and `delegation_result` events

## Notes

- **Subagent tool calls are hidden**: They execute silently in the background
- **Only text streams**: Subagents stream text responses, not tool call events
- **Simple accumulation**: Just append `message` chunks to build the complete response
- **Tool calls are supervisor-only**: Only supervisor tool calls (like delegation) appear in `tool_calls`
