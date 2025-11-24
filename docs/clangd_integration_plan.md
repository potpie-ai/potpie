# Clangd Integration Revamp Plan

## Current State Analysis

### What's Working
- ✅ Basic clangd LSP integration via pygls
- ✅ Custom caching layer for LSP query results (definitions, references, hover, symbols)
- ✅ Background indexing flag enabled (`--background-index`)
- ✅ Cache directory configuration via symlink to `.cache/clangd/index`
- ✅ Environment variable configuration (XDG_CACHE_HOME, DARWIN_USER_CACHE_DIR)
- ✅ Timeout configuration (120 seconds)

### Current Issues
- ❌ No `compile_commands.json` detection/generation
- ❌ No `.clangd` configuration file support
- ❌ Native index not building (likely due to missing compile_commands.json)
- ❌ Indexing wait logic is complex and may not be optimal
- ❌ No graceful degradation when clangd fails
- ❌ Cache invalidation not handled for file changes
- ❌ No verification that clangd is actually working correctly

## Best Practices from Research

### 1. Compilation Database (compile_commands.json)
- **Critical**: clangd needs this to properly index C/C++ projects
- Should be in project root or build directory
- Can be generated via:
  - CMake: `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1`
  - Bear: `bear -- make`
  - compiledb: `compiledb make`
  - Manual creation for simple projects

### 2. .clangd Configuration File
- Project-specific configuration
- Can specify compiler flags, include paths, etc.
- Format: YAML
- Example:
  ```yaml
  CompileFlags:
    Add: [-std=c++20, -I/path/to/headers]
  ```

### 3. Indexing Strategy
- Background indexing happens automatically with `--background-index`
- Index is stored in `.cache/clangd/index/` relative to workspace root
- Index files are `.idx` files
- Indexing can take time, especially for large projects
- Without `compile_commands.json`, indexing is limited

### 4. Session Management
- Clangd should be long-lived (persistent across requests)
- One clangd instance per workspace
- Proper shutdown on cleanup

## Proposed Improvements

### Phase 1: Foundation (Critical)

#### 1.1 Compile Commands Detection & Generation
**Goal**: Ensure clangd has the information it needs to index properly

**Implementation**:
- Detect existing `compile_commands.json` in workspace root
- If missing, attempt to generate it:
  - Check for CMake build directory
  - Check for Makefile
  - Try `bear` if available
  - Create minimal compile_commands.json for simple projects
- Log warnings if not found and indexing may be limited

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/lsp_server_manager.py`
- New: `app/modules/intelligence/tools/code_query_tools/clangd_helpers.py`

#### 1.2 .clangd Configuration Support
**Goal**: Allow project-specific clangd configuration

**Implementation**:
- Detect `.clangd` file in workspace root
- Parse YAML configuration
- Apply configuration via initialization options or environment
- Document format for users

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/clangd_helpers.py` (new)
- `app/modules/intelligence/tools/code_query_tools/lsp_server_manager.py`

#### 1.3 Better Indexing Verification
**Goal**: More reliable detection of when indexing is complete

**Implementation**:
- Check for `compile_commands.json` presence
- Monitor `.cache/clangd/index/` for `.idx` files
- Use clangd's progress notifications if available
- Set realistic expectations (indexing may take time)
- Don't block on indexing if it's taking too long

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/lsp_server_manager.py` (warmup logic)

### Phase 2: Reliability (Important)

#### 2.1 Graceful Degradation
**Goal**: Handle clangd failures gracefully

**Implementation**:
- Detect if clangd process crashes
- Restart clangd on failure
- Fallback to basic file-based analysis if clangd unavailable
- Clear error messages to users

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/clangd_session.py`
- `app/modules/intelligence/tools/code_query_tools/pygls_client_session.py`

#### 2.2 Cache Invalidation
**Goal**: Handle file changes properly

**Implementation**:
- Invalidate cache when files change (check file modification time)
- Use file content hash in cache keys (already done)
- Clear cache on workspace changes

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/clangd_session.py`

#### 2.3 Health Checks
**Goal**: Verify clangd is working correctly

**Implementation**:
- Send test request on startup
- Verify clangd responds correctly
- Check for common issues (missing headers, etc.)
- Log diagnostics

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/clangd_session.py`
- `app/modules/intelligence/tools/code_query_tools/lsp_server_manager.py`

### Phase 3: Optimization (Nice to Have)

#### 3.1 Smarter Indexing Strategy
**Goal**: Optimize indexing for large projects

**Implementation**:
- Prioritize indexing of commonly used files
- Use clangd's progress notifications
- Allow incremental indexing
- Don't wait for full index if partial is sufficient

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/lsp_server_manager.py`

#### 3.2 Performance Monitoring
**Goal**: Track clangd performance

**Implementation**:
- Log indexing time
- Track query response times
- Monitor cache hit rates
- Alert on performance degradation

**Files to modify**:
- `app/modules/intelligence/tools/code_query_tools/clangd_session.py`
- `app/modules/intelligence/tools/code_query_tools/lsp_server_manager.py`

## Implementation Priority

### High Priority (Do First)
1. ✅ Compile commands detection
2. ✅ Better indexing verification
3. ✅ Graceful degradation

### Medium Priority
4. .clangd configuration support
5. Cache invalidation
6. Health checks

### Low Priority
7. Smarter indexing strategy
8. Performance monitoring

## File Structure

```
app/modules/intelligence/tools/code_query_tools/
├── clangd_session.py          # ClangdSession class (existing)
├── clangd_helpers.py          # NEW: Helper functions for compile_commands.json, .clangd
├── lsp_server_manager.py      # Modified: Better indexing logic
└── pygls_client_session.py    # Modified: Better error handling
```

## Testing Strategy

1. **Unit Tests**:
   - Compile commands detection
   - .clangd parsing
   - Cache invalidation logic

2. **Integration Tests**:
   - Full clangd session lifecycle
   - Indexing with/without compile_commands.json
   - Error recovery

3. **Manual Testing**:
   - Test with real C/C++ projects
   - Test with and without compile_commands.json
   - Test with large projects

## Migration Plan

1. Implement Phase 1 improvements (non-breaking)
2. Test thoroughly
3. Deploy and monitor
4. Implement Phase 2 improvements
5. Implement Phase 3 improvements

## Success Metrics

- ✅ Clangd native index builds successfully
- ✅ Indexing completes in reasonable time
- ✅ Query response times are fast (< 1s for cached, < 5s for uncached)
- ✅ No crashes or hangs
- ✅ Works with and without compile_commands.json
