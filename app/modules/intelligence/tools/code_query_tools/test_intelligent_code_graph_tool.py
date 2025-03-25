import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import asyncio
from app.modules.intelligence.tools.code_query_tools.intelligent_code_graph_tool import (
    IntelligentCodeGraphTool,
    NodeRelevance,
    get_intelligent_code_graph_tool
)
from app.modules.intelligence.provider.provider_service import ProviderService
from pydantic import BaseModel
from typing import List


class NodeRelevanceList(BaseModel):
    evaluations: List[NodeRelevance]


class TestIntelligentCodeGraphTool(unittest.TestCase):
    def setUp(self):
        self.sql_db = MagicMock()
        self.provider_service = MagicMock(spec=ProviderService)
        self.user_id = "test-user-id"
        
        # Set up tool with mocked dependencies
        self.tool = IntelligentCodeGraphTool(self.sql_db, self.provider_service, self.user_id)
        
        # Mock the dependencies
        self.tool.code_graph_tool = MagicMock()
        self.tool.code_from_node_tool = MagicMock()

    @patch('asyncio.run')
    def test_run_success(self, mock_asyncio_run):
        # Arrange
        project_id = "test-project-id"
        node_id = "test-node-id"
        
        # Mock code graph tool response
        self.tool.code_graph_tool.run.return_value = {
            "graph": {
                "name": "test-repo",
                "repo_name": "test-repo",
                "branch_name": "main",
                "root_node": {
                    "id": node_id,
                    "name": "TestClass",
                    "type": "CLASS",
                    "file_path": "path/to/TestClass.java",
                    "start_line": 1,
                    "end_line": 100,
                    "children": [
                        {
                            "id": "child-1",
                            "name": "method1",
                            "type": "METHOD",
                            "file_path": "path/to/TestClass.java",
                            "start_line": 10,
                            "end_line": 20,
                            "relationship": "CONTAINS"
                        }
                    ]
                }
            }
        }
        
        # Mock code content response
        self.tool.code_from_node_tool.run.return_value = {
            "code": "public class TestClass { public void method1() { } }"
        }
        
        # Mock the async process function
        filtered_node = {
            "id": node_id,
            "name": "TestClass",
            "type": "CLASS",
            "file_path": "path/to/TestClass.java",
            "start_line": 1,
            "end_line": 100,
            "relevance_score": 1.0,
            "reason": "Entry point for analysis",
            "children": [
                {
                    "id": "child-1",
                    "name": "method1",
                    "type": "METHOD",
                    "file_path": "path/to/TestClass.java",
                    "start_line": 10,
                    "end_line": 20,
                    "relevance_score": 0.9,
                    "reason": "Core business logic",
                    "children": []
                }
            ]
        }
        mock_asyncio_run.return_value = filtered_node
        
        # Act
        result = self.tool.run(project_id, node_id)
        
        # Assert
        self.tool.code_graph_tool.run.assert_called_once_with(project_id, node_id, max_depth=1)
        self.tool.code_from_node_tool.run.assert_called_once_with(project_id, node_id)
        mock_asyncio_run.assert_called_once()
        
        self.assertEqual(result["graph"]["name"], "test-repo")
        self.assertEqual(result["graph"]["root_node"], filtered_node)
        self.assertIn("nodes_evaluated", result["graph"])
        self.assertIn("nodes_included", result["graph"])

    @patch('asyncio.run')
    def test_run_error_in_code_graph(self, mock_asyncio_run):
        # Arrange
        project_id = "test-project-id"
        node_id = "test-node-id"
        
        # Mock error response
        self.tool.code_graph_tool.run.return_value = {"error": "Node not found"}
        
        # Act
        result = self.tool.run(project_id, node_id)
        
        # Assert
        self.assertEqual(result, {"error": "Node not found"})
        self.tool.code_graph_tool.run.assert_called_once_with(project_id, node_id, max_depth=1)
        self.tool.code_from_node_tool.run.assert_not_called()
        mock_asyncio_run.assert_not_called()

    @patch('asyncio.run', side_effect=Exception("Test exception"))
    def test_run_exception(self, mock_asyncio_run):
        # Arrange
        project_id = "test-project-id"
        node_id = "test-node-id"
        
        # Mock successful graph response
        self.tool.code_graph_tool.run.return_value = {
            "graph": {
                "name": "test-repo",
                "repo_name": "test-repo",
                "branch_name": "main",
                "root_node": {
                    "id": node_id,
                    "name": "TestClass",
                    "type": "CLASS",
                    "file_path": "path/to/TestClass.java",
                    "children": []
                }
            }
        }
        
        # Mock code content response
        self.tool.code_from_node_tool.run.return_value = {
            "code": "public class TestClass { }"
        }
        
        # Act
        result = self.tool.run(project_id, node_id)
        
        # Assert
        self.assertIn("error", result)
        self.assertIn("Test exception", result["error"])

    async def test_process_node_recursively(self):
        # Arrange
        project_id = "test-project-id"
        node = {
            "id": "test-node-id",
            "name": "TestClass",
            "type": "CLASS",
            "file_path": "path/to/TestClass.java",
            "start_line": 1,
            "end_line": 100,
            "children": [
                {
                    "id": "child-1",
                    "name": "method1",
                    "type": "METHOD",
                    "file_path": "path/to/TestClass.java",
                    "start_line": 10,
                    "end_line": 20,
                    "relationship": "CONTAINS"
                },
                {
                    "id": "child-2",
                    "name": "logger",
                    "type": "FIELD",
                    "file_path": "path/to/TestClass.java",
                    "start_line": 5,
                    "end_line": 5,
                    "relationship": "CONTAINS"
                }
            ]
        }
        node_code = "public class TestClass { private Logger logger; public void method1() { logger.info('test'); } }"
        
        # Mock evaluate_nodes_with_llm
        self.tool._evaluate_nodes_with_llm = AsyncMock()
        self.tool._evaluate_nodes_with_llm.return_value = [
            NodeRelevance(
                node_id="child-1",
                node_name="method1",
                is_relevant=True,
                relevance_score=0.9,
                reason="Core business logic"
            ),
            NodeRelevance(
                node_id="child-2",
                node_name="logger",
                is_relevant=False,
                relevance_score=0.2,
                reason="Logging utility"
            )
        ]
        
        # Mock code_from_node_tool for child-1
        self.tool.code_from_node_tool.run.return_value = {
            "code": "public void method1() { logger.info('test'); }"
        }
        
        # Mock recursive call for child-1 (by patching the method itself)
        with patch.object(self.tool, '_process_node_recursively') as mock_recursive:
            # Set up the mock to return a result for child-1
            child1_result = {
                "id": "child-1",
                "name": "method1",
                "type": "METHOD",
                "file_path": "path/to/TestClass.java",
                "start_line": 10,
                "end_line": 20,
                "relevance_score": 0.9,
                "reason": "Core business logic",
                "children": []
            }
            mock_recursive.return_value = child1_result
            
            # Act
            result = await self.tool._process_node_recursively(
                project_id=project_id,
                node=node,
                node_code=node_code,
                relevance_threshold=0.6,
                current_depth=0,
                max_depth=5
            )
            
            # Assert
            self.tool._evaluate_nodes_with_llm.assert_called_once()
            
            # Only child-1 should be processed recursively
            mock_recursive.assert_called_once()
            
            # Verify the result
            self.assertEqual(result["id"], "test-node-id")
            self.assertEqual(result["name"], "TestClass")
            self.assertEqual(len(result["children"]), 1)  # Only one child (child-1) should be included
            self.assertEqual(result["children"][0]["id"], "child-1")

    def test_create_evaluation_prompt(self):
        # Arrange
        parent_node = {
            "id": "parent-id",
            "name": "TestClass",
            "type": "CLASS"
        }
        parent_code = "public class TestClass { }"
        children = [
            {
                "id": "child-1",
                "name": "method1",
                "type": "METHOD",
                "file_path": "path/to/TestClass.java"
            }
        ]
        
        # Act
        messages = self.tool._create_evaluation_prompt(parent_node, parent_code, children)
        
        # Assert
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("TestClass", messages[1]["content"])
        self.assertIn("method1", messages[1]["content"])

    def test_format_children_for_prompt(self):
        # Arrange
        children = [
            {
                "id": "child-1",
                "name": "method1",
                "type": "METHOD",
                "file_path": "path/to/TestClass.java",
                "relationship": "CONTAINS"
            }
        ]
        
        # Act
        formatted = self.tool._format_children_for_prompt(children)
        
        # Assert
        self.assertIn("Child 1:", formatted)
        self.assertIn("ID: child-1", formatted)
        self.assertIn("Name: method1", formatted)
        self.assertIn("Type: METHOD", formatted)
        self.assertIn("File: path/to/TestClass.java", formatted)
        self.assertIn("Relationship to Parent: CONTAINS", formatted)

    async def test_evaluate_nodes_with_llm(self):
        # Arrange
        messages = [{"role": "system", "content": "test"}]
        children = [
            {"id": "child-1", "name": "method1", "type": "METHOD"},
            {"id": "child-2", "name": "logger", "type": "FIELD"}
        ]
        
        # Mock provider_service.call_llm_with_structured_output
        self.provider_service.call_llm_with_structured_output = AsyncMock()
        self.provider_service.call_llm_with_structured_output.return_value = NodeRelevanceList(
            evaluations=[
                NodeRelevance(
                    node_id="child-1",
                    node_name="method1",
                    is_relevant=True,
                    relevance_score=0.9,
                    reason="Core business logic"
                ),
                NodeRelevance(
                    node_id="child-2",
                    node_name="logger",
                    is_relevant=False,
                    relevance_score=0.2,
                    reason="Logging utility"
                )
            ]
        )
        
        # Act
        result = await self.tool._evaluate_nodes_with_llm(messages, children)
        
        # Assert
        self.provider_service.call_llm_with_structured_output.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].node_id, "child-1")
        self.assertEqual(result[0].relevance_score, 0.9)
        self.assertEqual(result[1].node_id, "child-2")
        self.assertEqual(result[1].relevance_score, 0.2)

    async def test_evaluate_nodes_with_llm_exception(self):
        # Arrange
        messages = [{"role": "system", "content": "test"}]
        children = [
            {"id": "child-1", "name": "method1", "type": "METHOD"},
            {"id": "child-2", "name": "logger", "type": "FIELD"}
        ]
        
        # Mock provider_service.call_llm_with_structured_output to raise exception
        self.provider_service.call_llm_with_structured_output = AsyncMock(
            side_effect=Exception("LLM error")
        )
        
        # Act
        result = await self.tool._evaluate_nodes_with_llm(messages, children)
        
        # Assert
        self.provider_service.call_llm_with_structured_output.assert_called_once()
        self.assertEqual(len(result), 2)
        # Should return default values
        self.assertEqual(result[0].node_id, "child-1")
        self.assertEqual(result[0].relevance_score, 0.8)  # Default score
        self.assertEqual(result[1].node_id, "child-2")
        self.assertEqual(result[1].relevance_score, 0.8)  # Default score

    def test_create_relevant_node(self):
        # Arrange
        node = {
            "id": "test-id",
            "name": "TestClass",
            "type": "CLASS",
            "file_path": "path/to/TestClass.java",
            "start_line": 1,
            "end_line": 100,
            "extra_field": "should not be copied"
        }
        
        # Act
        relevant_node = self.tool._create_relevant_node(node, 0.9, "Test reason")
        
        # Assert
        self.assertEqual(relevant_node["id"], "test-id")
        self.assertEqual(relevant_node["name"], "TestClass")
        self.assertEqual(relevant_node["type"], "CLASS")
        self.assertEqual(relevant_node["file_path"], "path/to/TestClass.java")
        self.assertEqual(relevant_node["start_line"], 1)
        self.assertEqual(relevant_node["end_line"], 100)
        self.assertEqual(relevant_node["relevance_score"], 0.9)
        self.assertEqual(relevant_node["reason"], "Test reason")
        self.assertEqual(relevant_node["children"], [])
        self.assertNotIn("extra_field", relevant_node)

    def test_count_nodes(self):
        # Arrange
        node = {
            "id": "test-id",
            "name": "TestClass",
            "type": "CLASS",
            "children": [
                {
                    "id": "child-1",
                    "name": "method1",
                    "type": "METHOD",
                    "children": []
                },
                {
                    "id": "child-2",
                    "name": "method2",
                    "type": "METHOD",
                    "children": [
                        {
                            "id": "grandchild-1",
                            "name": "variable",
                            "type": "VARIABLE",
                            "children": []
                        }
                    ]
                }
            ]
        }
        
        # Act
        count = self.tool._count_nodes(node)
        
        # Assert
        self.assertEqual(count, 4)  # 1 parent + 2 children + 1 grandchild

    def test_get_intelligent_code_graph_tool(self):
        # Arrange
        sql_db = MagicMock()
        provider_service = MagicMock()
        user_id = "test-user-id"
        
        # Act
        tool = get_intelligent_code_graph_tool(sql_db, provider_service, user_id)
        
        # Assert
        self.assertEqual(tool.name, "intelligent_code_graph")
        self.assertIn("Intelligently fetches", tool.description)


if __name__ == '__main__':
    unittest.main() 