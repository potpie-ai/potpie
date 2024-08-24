# API Documentation for Conversations API

## Base URL: `/conversations/`

## Endpoints

### 1. Create a Conversation
- **Endpoint**: `/conversations/`
- **Method**: `POST`
- **Request Body**:
  - **Type**: `CreateConversationRequest`
  - **Description**: Contains the details required to create a conversation.
  - **Schema**:
    ```json
    {
      "user_id": "string",           // Unique identifier for the user creating the conversation
      "title": "string",             // Title of the conversation
      "status": "ConversationStatus", // Status of the conversation (active, archived, deleted)
      "project_ids": ["string"]      // List of project IDs associated with the conversation
    }
    ```
- **Response**:
  - **Type**: `CreateConversationResponse`
  - **Description**: Returns the details of the created conversation.
  - **Schema**:
    ```json
    {
      "message": "string",          // Confirmation message
      "conversation_id": "string"   // Unique identifier for the created conversation
    }
    ```
- **Example Request**:
    ```json
    {
      "user_id": "user123",
      "title": "New Conversation",
      "status": "active",
      "project_ids": ["project1", "project2"]
    }
    ```
- **Example Response**:
    ```json
    {
      "message": "Conversation created successfully.",
      "conversation_id": "123"
    }
    ```
- **Possible Status Codes**:
  - `201 Created`: Conversation created successfully.
  - `400 Bad Request`: Invalid input data.
  - `500 Internal Server Error`: Unexpected server error.

### 2. Get Conversation Info
- **Endpoint**: `/conversations/{conversation_id}/info/`
- **Method**: `GET`
- **Path Parameters**:
  - **conversation_id**: `string` - Unique identifier for the conversation.
- **Response**:
  - **Type**: `ConversationInfoResponse`
  - **Description**: Returns information about the specified conversation.
  - **Schema**:
    ```json
    {
      "id": "string",               // Unique identifier for the conversation
      "title": "string",            // Title of the conversation
      "status": "ConversationStatus", // Current status of the conversation
      "project_ids": ["string"],    // List of project IDs associated with the conversation
      "created_at": "datetime",     // Timestamp when the conversation was created
      "updated_at": "datetime",     // Timestamp when the conversation was last updated
      "total_messages": "int"       // Total number of messages in the conversation
    }
    ```
- **Example Response**:
    ```json
    {
      "id": "123",
      "title": "New Conversation",
      "status": "active",
      "project_ids": ["project1", "project2"],
      "created_at": "2024-08-24T12:00:00Z",
      "updated_at": "2024-08-24T12:05:00Z",
      "total_messages": 5
    }
    ```
- **Possible Status Codes**:
  - `200 OK`: Successfully retrieved conversation info.
  - `404 Not Found`: Conversation not found.
  - `500 Internal Server Error`: Unexpected server error.

### 3. Get Conversation Messages
- **Endpoint**: `/conversations/{conversation_id}/messages/`
- **Method**: `GET`
- **Path Parameters**:
  - **conversation_id**: `string` - Unique identifier for the conversation.
- **Query Parameters**:
  - **start**: `int` - The starting index for the messages. Default is `0`.
  - **limit**: `int` - The maximum number of messages to return. Default is `10`.
- **Response**:
  - **Type**: `List[MessageResponse]`
  - **Description**: Returns a list of messages for the specified conversation.
- **Example Response**:
    ```json
    [
      {
        "id": "msg1",
        "content": "Hello!",
        "sender": "user1",
        "timestamp": "2024-08-24T12:01:00Z"
      },
      {
        "id": "msg2",
        "content": "Hi there!",
        "sender": "user2",
        "timestamp": "2024-08-24T12:02:00Z"
      }
    ]
    ```
- **Possible Status Codes**:
  - `200 OK`: Successfully retrieved messages.
  - `404 Not Found`: Conversation not found.
  - `500 Internal Server Error`: Unexpected server error.

### 4. Post a Message
- **Endpoint**: `/conversations/{conversation_id}/message/`
- **Method**: `POST`
- **Path Parameters**:
  - **conversation_id**: `string` - Unique identifier for the conversation.
- **Request Body**:
  - **Type**: `MessageRequest`
  - **Description**: Contains the message content to be sent.
- **Response**:
  - **Type**: `StreamingResponse`
  - **Description**: Streams the response of the posted message.
- **Example Request**:
    ```json
    {
      "content": "This is a new message."
    }
    ```
- **Possible Status Codes**:
  - `200 OK`: Message posted successfully.
  - `400 Bad Request`: Invalid message content.
  - `404 Not Found`: Conversation not found.
  - `500 Internal Server Error`: Unexpected server error.

### 5. Regenerate Last Message
- **Endpoint**: `/conversations/{conversation_id}/regenerate/`
- **Method**: `POST`
- **Path Parameters**:
  - **conversation_id**: `string` - Unique identifier for the conversation.
- **Response**:
  - **Type**: `MessageResponse`
  - **Description**: Returns the regenerated last message.
- **Example Response**:
    ```json
    {
      "id": "msg1",
      "content": "This is the regenerated message.",
      "sender": "user1",
      "timestamp": "2024-08-24T12:03:00Z"
    }
    ```
- **Possible Status Codes**:
  - `200 OK`: Last message regenerated successfully.
  - `404 Not Found`: Conversation not found.
  - `500 Internal Server Error`: Unexpected server error.

### 6. Delete a Conversation
- **Endpoint**: `/conversations/{conversation_id}/`
- **Method**: `DELETE`
- **Path Parameters**:
  - **conversation_id**: `string` - Unique identifier for the conversation.
- **Response**:
  - **Type**: `dict`
  - **Description**: Confirms the deletion of the conversation.
- **Example Response**:
    ```json
    {
      "message": "Conversation deleted successfully."
    }
    ```
- **Possible Status Codes**:
  - `200 OK`: Conversation deleted successfully.
  - `404 Not Found`: Conversation not found.
  - `500 Internal Server Error`: Unexpected server error.

### 7. Stop Generation
- **Endpoint**: `/conversations/{conversation_id}/stop/`
- **Method**: `POST`
- **Path Parameters**:
  - **conversation_id**: `string` - Unique identifier for the conversation.
- **Response**:
  - **Type**: `dict`
  - **Description**: Confirms that the generation process has been stopped.
- **Example Response**:
    ```json
    {
      "message": "Generation stopped successfully."
    }
    ```
- **Possible Status Codes**:
  - `200 OK`: Generation stopped successfully.
  - `404 Not Found`: Conversation not found.
  - `500 Internal Server Error`: Unexpected server error.

## Schema Definitions

### CreateConversationRequest
- **Description**: Request body for creating a new conversation.
- **Fields**:
  - `user_id` (string): Unique identifier for the user creating the conversation.
  - `title` (string): Title of the conversation.
  - `status` (ConversationStatus): Status of the conversation (e.g., active, archived).
  - `project_ids` (List[string]): List of project IDs associated with the conversation.

### CreateConversationResponse
- **Description**: Response body for a created conversation.
- **Fields**:
  - `message` (string): Confirmation message indicating success.
  - `conversation_id` (string): Unique identifier for the created conversation.

### ConversationInfoResponse
- **Description**: Response body containing information about a conversation.
- **Fields**:
  - `id` (string): Unique identifier for the conversation.
  - `title` (string): Title of the conversation.
  - `status` (ConversationStatus): Current status of the conversation.
  - `project_ids` (List[string]): List of project IDs associated with the conversation.
  - `created_at` (datetime): Timestamp when the conversation was created.
  - `updated_at` (datetime): Timestamp when the conversation was last updated.
  - `total_messages` (int): Total number of messages in the conversation.
