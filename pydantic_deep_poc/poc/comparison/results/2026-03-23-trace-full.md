# Full Trace: celery-to-hatchet task
**Trace ID:** `019d19a171c34878ac522ad00ea80d09`
**Date:** 2026-03-23  07:38:24 → 07:57:25 UTC  (~19 min)
**Model:** minimax/minimax-m2.7  |  **Task:** Replace Celery with Hatchet across Potpie
**Outcome:** `UsageLimitExceeded: request_limit of 50` — both agent runs

## Agent runs
| span_id | role | duration | outcome |
|---|---|---|---|
| `fbc707288bd536ae` | supervisor (run 1) | 1141.7s | UsageLimitExceeded @ request 50 |
| `dc4f01a3631e7f0f` | THINK_EXECUTE subagent (run 2) | 636.4s | UsageLimitExceeded @ request 50 |

## Token usage
| Agent | LLM turns | Total input tokens | Total output tokens |
|---|---|---|---|
| supervisor (fbc…) | 50 | 4,252,909 | 21,392 |
| THINK_EXECUTE (dc4…) | 50 | 4,053,904 | 37,878 |
| **Total** | **100** | **8,306,813** | **59,270** |

## LLM turns detail
| # | time | agent | in_tokens | out_tokens | duration_s |
|---|---|---|---|---|---|
| 1 | 07:38:24 | supervisor | 8,466 | 189 | 6.0 |
| 2 | 07:38:30 | supervisor | 15,834 | 226 | 5.0 |
| 3 | 07:38:35 | supervisor | 28,464 | 223 | 6.7 |
| 4 | 07:38:42 | supervisor | 33,424 | 153 | 5.2 |
| 5 | 07:38:47 | supervisor | 39,526 | 160 | 6.0 |
| 6 | 07:38:53 | supervisor | 46,835 | 109 | 5.1 |
| 7 | 07:38:58 | supervisor | 51,676 | 185 | 6.6 |
| 8 | 07:39:05 | supervisor | 52,240 | 152 | 4.5 |
| 9 | 07:39:11 | supervisor | 53,516 | 136 | 6.3 |
| 10 | 07:39:17 | supervisor | 56,264 | 120 | 8.0 |
| 11 | 07:39:25 | supervisor | 80,406 | 81 | 6.2 |
| 12 | 07:39:32 | supervisor | 80,545 | 49 | 3.7 |
| 13 | 07:39:36 | supervisor | 80,654 | 58 | 3.8 |
| 14 | 07:39:40 | supervisor | 80,730 | 49 | 3.6 |
| 15 | 07:39:44 | supervisor | 81,863 | 66 | 3.7 |
| 16 | 07:39:49 | supervisor | 83,109 | 118 | 6.2 |
| 17 | 07:39:56 | supervisor | 84,728 | 62 | 5.1 |
| 18 | 07:40:02 | supervisor | 86,066 | 101 | 6.8 |
| 19 | 07:40:09 | supervisor | 86,288 | 98 | 4.8 |
| 20 | 07:40:15 | supervisor | 86,621 | 117 | 4.5 |
| 21 | 07:40:20 | supervisor | 86,885 | 118 | 5.2 |
| 22 | 07:40:25 | supervisor | 89,519 | 80 | 6.5 |
| 23 | 07:40:32 | supervisor | 89,671 | 58 | 3.7 |
| 24 | 07:40:36 | supervisor | 89,796 | 55 | 4.1 |
| 25 | 07:40:41 | supervisor | 89,921 | 59 | 4.2 |
| 26 | 07:40:45 | supervisor | 90,049 | 54 | 4.0 |
| 27 | 07:40:50 | supervisor | 90,172 | 60 | 3.8 |
| 28 | 07:40:54 | supervisor | 91,995 | 1,178 | 22.3 |
| 29 | 07:41:17 | supervisor | 93,274 | 83 | 5.8 |
| 30 | 07:41:23 | supervisor | 93,403 | 1,936 | 26.9 |
| 31 | 07:41:50 | supervisor | 95,495 | 1,324 | 22.6 |
| 32 | 07:42:12 | subagent | 7,764 | 237 | 5.6 |
| 33 | 07:42:18 | subagent | 14,077 | 71 | 3.2 |
| 34 | 07:42:21 | subagent | 14,172 | 66 | 2.3 |
| 35 | 07:42:23 | subagent | 14,291 | 62 | 2.2 |
| 36 | 07:42:26 | subagent | 14,399 | 52 | 1.8 |
| 37 | 07:42:27 | subagent | 14,491 | 43 | 1.8 |
| 38 | 07:42:29 | subagent | 14,567 | 39 | 1.9 |
| 39 | 07:42:31 | subagent | 14,636 | 50 | 2.4 |
| 40 | 07:42:34 | subagent | 15,448 | 223 | 4.4 |
| 41 | 07:42:38 | subagent | 33,113 | 142 | 10.1 |
| 42 | 07:42:48 | subagent | 78,410 | 69 | 5.8 |
| 43 | 07:42:54 | subagent | 78,506 | 68 | 4.0 |
| 44 | 07:42:58 | subagent | 78,631 | 67 | 3.1 |
| 45 | 07:43:01 | subagent | 78,755 | 60 | 4.4 |
| 46 | 07:43:05 | subagent | 79,623 | 752 | 12.0 |
| 47 | 07:43:17 | subagent | 80,408 | 684 | 12.7 |
| 48 | 07:43:30 | subagent | 81,137 | 637 | 12.0 |
| 49 | 07:43:42 | subagent | 81,807 | 858 | 13.6 |
| 50 | 07:43:56 | subagent | 82,713 | 653 | 10.0 |
| 51 | 07:44:06 | subagent | 83,399 | 6,764 | 77.3 |
| 52 | 07:45:23 | subagent | 90,195 | 639 | 13.9 |
| 53 | 07:45:37 | subagent | 90,867 | 789 | 13.1 |
| 54 | 07:45:50 | subagent | 91,688 | 639 | 12.4 |
| 55 | 07:46:03 | subagent | 92,360 | 1,124 | 17.3 |
| 56 | 07:46:20 | subagent | 93,515 | 639 | 11.1 |
| 57 | 07:46:31 | subagent | 94,187 | 745 | 13.1 |
| 58 | 07:46:44 | subagent | 94,963 | 639 | 11.8 |
| 59 | 07:46:56 | subagent | 95,635 | 1,833 | 28.0 |
| 60 | 07:47:24 | subagent | 97,497 | 640 | 13.8 |
| 61 | 07:47:38 | subagent | 98,170 | 215 | 6.1 |
| 62 | 07:47:44 | subagent | 98,416 | 645 | 11.5 |
| 63 | 07:47:56 | subagent | 99,094 | 2,974 | 42.1 |
| 64 | 07:48:38 | subagent | 102,102 | 641 | 12.1 |
| 65 | 07:48:50 | subagent | 102,776 | 146 | 4.9 |
| 66 | 07:48:55 | subagent | 102,955 | 69 | 4.8 |
| 67 | 07:49:00 | subagent | 103,057 | 77 | 4.5 |
| 68 | 07:49:04 | subagent | 103,194 | 52 | 4.9 |
| 69 | 07:49:09 | subagent | 104,008 | 47 | 4.2 |
| 70 | 07:49:13 | subagent | 104,285 | 72 | 4.7 |
| 71 | 07:49:18 | subagent | 104,909 | 63 | 4.4 |
| 72 | 07:49:22 | subagent | 105,004 | 75 | 4.7 |
| 73 | 07:49:27 | subagent | 105,106 | 665 | 12.3 |
| 74 | 07:49:39 | subagent | 105,876 | 930 | 15.2 |
| 75 | 07:49:55 | subagent | 106,914 | 6,785 | 86.0 |
| 76 | 07:51:21 | subagent | 113,761 | 1,963 | 28.5 |
| 77 | 07:51:49 | subagent | 115,832 | 2,853 | 39.6 |
| 78 | 07:52:29 | subagent | 118,836 | 147 | 6.4 |
| 79 | 07:52:35 | subagent | 119,047 | 47 | 3.7 |
| 80 | 07:52:39 | subagent | 119,231 | 54 | 4.5 |
| 81 | 07:52:44 | subagent | 120,077 | 74 | 5.0 |
| 82 | 07:52:49 | supervisor | 96,850 | 89 | 7.5 |
| 83 | 07:52:56 | supervisor | 96,966 | 190 | 6.7 |
| 84 | 07:53:03 | supervisor | 97,183 | 1,934 | 28.7 |
| 85 | 07:53:32 | supervisor | 99,144 | 193 | 6.1 |
| 86 | 07:53:38 | supervisor | 99,364 | 447 | 9.5 |
| 87 | 07:53:47 | supervisor | 99,838 | 4,379 | 66.6 |
| 88 | 07:54:54 | supervisor | 104,244 | 571 | 13.5 |
| 89 | 07:55:07 | supervisor | 104,842 | 693 | 12.4 |
| 90 | 07:55:20 | supervisor | 105,562 | 561 | 14.4 |
| 91 | 07:55:34 | supervisor | 106,150 | 804 | 15.5 |
| 92 | 07:55:50 | supervisor | 106,981 | 238 | 6.6 |
| 93 | 07:55:56 | supervisor | 107,246 | 86 | 4.4 |
| 94 | 07:56:01 | supervisor | 110,218 | 3,015 | 45.8 |
| 95 | 07:56:47 | supervisor | 113,260 | 110 | 5.8 |
| 96 | 07:56:52 | supervisor | 115,655 | 99 | 6.4 |
| 97 | 07:56:59 | supervisor | 117,534 | 142 | 8.5 |
| 98 | 07:57:07 | supervisor | 117,703 | 197 | 7.4 |
| 99 | 07:57:15 | supervisor | 117,927 | 102 | 5.1 |
| 100 | 07:57:20 | supervisor | 118,807 | 85 | 5.2 |

## Tool call sequence (154 calls)
| # | time | agent | tool | exception |
|---|---|---|---|---|
| 1 | 07:38:30 | supervisor | `webpage_extractor` | ✅ |
| 2 | 07:38:30 | supervisor | `get_code_file_structure` | ✅ |
| 3 | 07:38:35 | supervisor | `webpage_extractor` | ✅ |
| 4 | 07:38:35 | supervisor | `fetch_file` | ✅ |
| 5 | 07:38:35 | supervisor | `fetch_file` | ✅ |
| 6 | 07:38:35 | supervisor | `fetch_file` | ✅ |
| 7 | 07:38:35 | supervisor | `fetch_file` | ✅ |
| 8 | 07:38:35 | supervisor | `fetch_file` | ✅ |
| 9 | 07:38:42 | supervisor | `fetch_file` | ✅ |
| 10 | 07:38:42 | supervisor | `fetch_file` | ✅ |
| 11 | 07:38:42 | supervisor | `fetch_file` | ✅ |
| 12 | 07:38:42 | supervisor | `fetch_file` | ✅ |
| 13 | 07:38:42 | supervisor | `fetch_file` | ✅ |
| 14 | 07:38:42 | supervisor | `fetch_file` | ✅ |
| 15 | 07:38:47 | supervisor | `fetch_file` | ✅ |
| 16 | 07:38:47 | supervisor | `fetch_file` | ✅ |
| 17 | 07:38:47 | supervisor | `fetch_file` | ✅ |
| 18 | 07:38:47 | supervisor | `fetch_file` | ✅ |
| 19 | 07:38:53 | supervisor | `fetch_file` | ✅ |
| 20 | 07:38:53 | supervisor | `fetch_file` | ✅ |
| 21 | 07:38:53 | supervisor | `fetch_file` | ✅ |
| 22 | 07:38:53 | supervisor | `fetch_file` | ✅ |
| 23 | 07:38:58 | supervisor | `fetch_file` | ✅ |
| 24 | 07:38:58 | supervisor | `webpage_extractor` | ✅ |
| 25 | 07:38:58 | supervisor | `fetch_file` | ✅ |
| 26 | 07:39:05 | supervisor | `web_search_tool` | ✅ |
| 27 | 07:39:05 | supervisor | `bash_command` | ✅ |
| 28 | 07:39:05 | supervisor | `fetch_file` | ✅ |
| 29 | 07:39:10 | supervisor | `web_search_tool` | ✅ |
| 30 | 07:39:10 | supervisor | `bash_command` | ✅ |
| 31 | 07:39:17 | supervisor | `fetch_file` | ✅ |
| 32 | 07:39:17 | supervisor | `fetch_file` | ✅ |
| 33 | 07:39:17 | supervisor | `fetch_file` | ✅ |
| 34 | 07:39:17 | supervisor | `fetch_file` | ✅ |
| 35 | 07:39:25 | supervisor | `fetch_file` | ✅ |
| 36 | 07:39:25 | supervisor | `fetch_file` | ✅ |
| 37 | 07:39:25 | supervisor | `fetch_file` | ✅ |
| 38 | 07:39:32 | supervisor | `webpage_extractor` | ✅ |
| 39 | 07:39:35 | supervisor | `webpage_extractor` | ✅ |
| 40 | 07:39:39 | supervisor | `web_search_tool` | ✅ |
| 41 | 07:39:43 | supervisor | `webpage_extractor` | ✅ |
| 42 | 07:39:48 | supervisor | `webpage_extractor` | ✅ |
| 43 | 07:39:55 | supervisor | `webpage_extractor` | ✅ |
| 44 | 07:39:55 | supervisor | `webpage_extractor` | ✅ |
| 45 | 07:40:01 | supervisor | `webpage_extractor` | ✅ |
| 46 | 07:40:09 | supervisor | `webpage_extractor` | ✅ |
| 47 | 07:40:09 | supervisor | `webpage_extractor` | ✅ |
| 48 | 07:40:14 | supervisor | `webpage_extractor` | ✅ |
| 49 | 07:40:14 | supervisor | `webpage_extractor` | ✅ |
| 50 | 07:40:19 | supervisor | `webpage_extractor` | ✅ |
| 51 | 07:40:19 | supervisor | `webpage_extractor` | ✅ |
| 52 | 07:40:25 | supervisor | `webpage_extractor` | ✅ |
| 53 | 07:40:25 | supervisor | `webpage_extractor` | ✅ |
| 54 | 07:40:32 | supervisor | `webpage_extractor` | ✅ |
| 55 | 07:40:36 | supervisor | `webpage_extractor` | ✅ |
| 56 | 07:40:40 | supervisor | `webpage_extractor` | ✅ |
| 57 | 07:40:45 | supervisor | `webpage_extractor` | ✅ |
| 58 | 07:40:49 | supervisor | `webpage_extractor` | ✅ |
| 59 | 07:40:54 | supervisor | `webpage_extractor` | ✅ |
| 60 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 61 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 62 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 63 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 64 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 65 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 66 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 67 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 68 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 69 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 70 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 71 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 72 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 73 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 74 | 07:41:17 | supervisor | `add_todo` | ✅ |
| 75 | 07:41:23 | supervisor | `checkout_worktree_branch` | ✅ |
| 76 | 07:41:50 | supervisor | `add_file_to_changes` | ❌ Unknown tool name: 'add_file_to_changes'. Available tools: 'task', 'check_task', |
| 77 | 07:42:12 | supervisor | `task [→ THINK_EXECUTE]` | ✅ |
| 78 | 07:42:18 | subagent | `ls` | ✅ |
| 79 | 07:42:18 | subagent | `get_code_file_structure` | ✅ |
| 80 | 07:42:21 | subagent | `execute` | ✅ |
| 81 | 07:42:23 | subagent | `ls` | ✅ |
| 82 | 07:42:26 | subagent | `ls` | ✅ |
| 83 | 07:42:27 | subagent | `ls` | ✅ |
| 84 | 07:42:29 | subagent | `ls` | ✅ |
| 85 | 07:42:31 | subagent | `ls` | ✅ |
| 86 | 07:42:34 | subagent | `bash_command` | ✅ |
| 87 | 07:42:38 | subagent | `fetch_files_batch` | ✅ |
| 88 | 07:42:48 | subagent | `fetch_files_batch` | ✅ |
| 89 | 07:42:54 | subagent | `read_file` | ✅ |
| 90 | 07:42:58 | subagent | `read_file` | ✅ |
| 91 | 07:43:01 | subagent | `ls` | ✅ |
| 92 | 07:43:05 | subagent | `fetch_file` | ✅ |
| 93 | 07:43:17 | subagent | `write_todos` | ✅ |
| 94 | 07:43:30 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 95 | 07:43:30 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 96 | 07:43:42 | subagent | `write_todos` | ✅ |
| 97 | 07:43:56 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 98 | 07:43:56 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 99 | 07:44:06 | subagent | `write_todos` | ✅ |
| 100 | 07:45:23 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 101 | 07:45:37 | subagent | `write_todos` | ✅ |
| 102 | 07:45:50 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 103 | 07:46:03 | subagent | `write_todos` | ✅ |
| 104 | 07:46:20 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 105 | 07:46:31 | subagent | `write_todos` | ✅ |
| 106 | 07:46:44 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 107 | 07:46:56 | subagent | `write_todos` | ✅ |
| 108 | 07:47:24 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 109 | 07:47:38 | subagent | `write_todos` | ✅ |
| 110 | 07:47:44 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 111 | 07:47:56 | subagent | `write_todos` | ✅ |
| 112 | 07:48:38 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 113 | 07:48:50 | subagent | `write_todos` | ✅ |
| 114 | 07:48:55 | subagent | `hashline_edit` | ✅ |
| 115 | 07:49:00 | subagent | `read_file` | ✅ |
| 116 | 07:49:04 | subagent | `ls` | ✅ |
| 117 | 07:49:09 | subagent | `bash_command` | ✅ |
| 118 | 07:49:13 | subagent | `bash_command` | ✅ |
| 119 | 07:49:18 | subagent | `bash_command` | ✅ |
| 120 | 07:49:22 | subagent | `bash_command` | ✅ |
| 121 | 07:49:27 | subagent | `bash_command` | ✅ |
| 122 | 07:49:39 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 123 | 07:49:39 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 124 | 07:49:55 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 125 | 07:49:55 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 126 | 07:51:21 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 127 | 07:51:49 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 128 | 07:51:49 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 129 | 07:52:29 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 130 | 07:52:29 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 131 | 07:52:29 | subagent | `write_file [⚠ bypasses CCM]` | ✅ |
| 132 | 07:52:35 | subagent | `read_file` | ✅ |
| 133 | 07:52:39 | supervisor | `bash_command` | ✅ |
| 134 | 07:52:44 | supervisor | `bash_command` | ✅ |
| 135 | 07:52:49 | supervisor | `bash_command` | ✅ |
| 136 | 07:52:56 | supervisor | `bash_command` | ✅ |
| 137 | 07:53:03 | supervisor | `bash_command` | ✅ |
| 138 | 07:53:32 | supervisor | `bash_command` | ✅ |
| 139 | 07:53:38 | supervisor | `bash_command` | ✅ |
| 140 | 07:53:47 | supervisor | `bash_command` | ✅ |
| 141 | 07:54:54 | supervisor | `bash_command` | ✅ |
| 142 | 07:55:07 | supervisor | `bash_command` | ✅ |
| 143 | 07:55:20 | supervisor | `bash_command` | ✅ |
| 144 | 07:55:34 | supervisor | `bash_command` | ✅ |
| 145 | 07:55:50 | supervisor | `bash_command` | ✅ |
| 146 | 07:55:56 | supervisor | `bash_command` | ✅ |
| 147 | 07:56:01 | supervisor | `fetch_file` | ✅ |
| 148 | 07:56:47 | supervisor | `bash_command` | ✅ |
| 149 | 07:56:52 | supervisor | `fetch_file` | ✅ |
| 150 | 07:56:59 | supervisor | `fetch_file` | ✅ |
| 151 | 07:57:07 | supervisor | `bash_command` | ✅ |
| 152 | 07:57:15 | supervisor | `bash_command` | ✅ |
| 153 | 07:57:20 | supervisor | `fetch_file` | ✅ |
| 154 | 07:57:25 | supervisor | `bash_command` | ✅ |

## Exceptions
```
1. [07:41:50] supervisor called add_file_to_changes
   → ToolRetryError: Unknown tool name: 'add_file_to_changes'.
   Available: task, check_task, answer_subagent, list_active_tasks, wait_tasks,
             soft_cancel_task, hard_cancel_task, fetch_file, get_code_file_structure,
             web_search_tool, webpage_extractor, read_todos, write_todos, add_todo,
             update_todo_status, remove_todo, add_subtask, set_dependency,
             get_available_tasks, add_requirements, get_requirements,
             delete_requirements, bash_command, apply_changes, checkout_worktree_branch

2. [07:57:25] supervisor run 1 hit request_limit=50
   → UsageLimitExceeded: The next request would exceed the request_limit of 50

3. [end of subagent run] subagent run 2 hit request_limit=50
   → UsageLimitExceeded: The next request would exceed the request_limit of 50
```

## Key observations
- **Turns 1–59 (supervisor):** Pure exploration — no code written. 34×fetch_file + 24×webpage_extractor + 3×web_search + 2×bash. Context window grew from 8k → 92k tokens over these turns.
- **Turn 28 (supervisor input_tokens=91,995):** Big 22s turn generating 1,178 output tokens — the planning output (add_todo ×15 happened right after).
- **Turn 76:** First and only `add_file_to_changes` call → immediate ToolRetryError. Supervisor did NOT have CCM tools.
- **Turn 77:** Only delegation in entire run — `task` → THINK_EXECUTE.
- **Subagent turns 1–8 (07:42:12–07:42:34):** Re-explored codebase with `ls` ×6, `get_code_file_structure`, `execute` — duplicating supervisor's exploration.
- **Subagent turns 15–50 (07:43:05–07:52:44):** Used pydantic-deep built-in `write_file` ×21 (NOT CCM tools). Files written directly to disk; RunContext.code_changes stayed empty.
- **Subagent biggest turn:** 07:44:06, 77.3s, 6,764 output tokens — bulk of writing happened here.
- **Supervisor resumes (07:52:49):** Supervisor context has grown to 96k tokens. Ran bash_command ×14 for verification. apply_changes not called (CCM was empty).
- **Both runs hit limit 50** — neither committed anything via git.

## Context window growth (supervisor)
| time | input_tokens | notes |
|---|---|---|
| 07:38:24 | 8,466 | |
| 07:38:30 | 15,834 | |
| 07:38:35 | 28,464 | |
| 07:38:42 | 33,424 | |
| 07:38:47 | 39,526 | |
| 07:38:53 | 46,835 | |
| 07:38:58 | 51,676 | |
| 07:39:05 | 52,240 | |
| 07:39:11 | 53,516 | |
| 07:39:17 | 56,264 | |
| 07:39:25 | 80,406 | |
| 07:39:32 | 80,545 | |
| 07:39:36 | 80,654 | |
| 07:39:40 | 80,730 | |
| 07:39:44 | 81,863 | |
| 07:39:49 | 83,109 | |
| 07:39:56 | 84,728 | |
| 07:40:02 | 86,066 | |
| 07:40:09 | 86,288 | |
| 07:40:15 | 86,621 | |
| 07:40:20 | 86,885 | |
| 07:40:25 | 89,519 | |
| 07:40:32 | 89,671 | |
| 07:40:36 | 89,796 | |
| 07:40:41 | 89,921 | |
| 07:40:45 | 90,049 | |
| 07:40:50 | 90,172 | |
| 07:40:54 | 91,995 | |
| 07:41:17 | 93,274 | |
| 07:41:23 | 93,403 | |
| 07:41:50 | 95,495 | |
| 07:52:49 | 96,850 | |
| 07:52:56 | 96,966 | |
| 07:53:03 | 97,183 | |
| 07:53:32 | 99,144 | |
| 07:53:38 | 99,364 | |
| 07:53:47 | 99,838 | |
| 07:54:54 | 104,244 | |
| 07:55:07 | 104,842 | |
| 07:55:20 | 105,562 | |
| 07:55:34 | 106,150 | |
| 07:55:50 | 106,981 | |
| 07:55:56 | 107,246 | |
| 07:56:01 | 110,218 | |
| 07:56:47 | 113,260 | |
| 07:56:52 | 115,655 | |
| 07:56:59 | 117,534 | |
| 07:57:07 | 117,703 | |
| 07:57:15 | 117,927 | |
| 07:57:20 | 118,807 | |