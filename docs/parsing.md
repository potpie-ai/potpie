# API Documentation for Parsing API

## Base URL: `/parse/`

## Endpoints

### 1. Parse Directory
- **Endpoint**: `/parse`
- **Method**: `POST`
- **Request Body**:
  - **Type**: `ParsingRequest`
  - **Description**: Contains the details required to parse a directory.
  - **Schema**:
    ```json
    {
      "repo_name": "string",      // Name of the repository (optional if repo_path is provided)
      "repo_path": "string",      // Local path to the repository (optional if repo_name is provided)
      "branch_name": "string"     // Name of the branch to parse
    }
    ```
- **Response**:
  - **Type**: `ParsingResponse`
  - **Description**: Returns the result of the parsing operation.
  - **Schema**:
    ```json
    {
      "message": "string",        // Confirmation message
      "status": "string",         // Status of the parsing operation
      "project_id": "string"      // Unique identifier for the parsed project
    }
    ```
- **Example Request**:
    ```json
    {
      "repo_name": "user/repo",
      "branch_name": "main"
    }
    ```
- **Example Response**:
    ```json
    {
      "message": "The project has been parsed successfully.",
      "status": "READY",
      "project_id": "12345"
    }
    ```
- **Possible Status Codes**:
  - `200 OK`: Parsing completed successfully.
  - `400 Bad Request`: Invalid input data.
  - `403 Forbidden`: Access denied due to restrictions.
  - `500 Internal Server Error`: Unexpected server error.


## Error Handling

### Common Error Responses
- **400 Bad Request**
  - **Description**: Invalid input data.
  - **Example Response**:
    ```json
    {
      "detail": "Invalid input data."
    }
    ```

- **403 Forbidden**
  - **Description**: Access denied due to restrictions.
  - **Example Response**:
    ```json
    {
      "detail": "Cannot parse remote repository without auth token."
    }
    ```

- **500 Internal Server Error**
  - **Description**: Unexpected server error.
  - **Example Response**:
    ```json
    {
      "detail": "Unexpected server error."
    }
    ```

## Additional Notes
- Ensure that the environment variable `isDevelopmentMode` is set to "enabled" to parse local repositories.
- The `user_id` must not match the `defaultUsername` environment variable when parsing remote repositories.
