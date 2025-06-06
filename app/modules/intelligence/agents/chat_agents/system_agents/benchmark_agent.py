from app.modules.intelligence.agents.chat_agents.adaptive_agent import AdaptiveAgent
from app.modules.intelligence.agents.chat_agents.agno_agent import TeamChatAgent
from app.modules.intelligence.agents.chat_agents.pydantic_4 import (
    PydanticToolGraphAgent,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_complex_task import (
    PydanticGraphAgent,
)
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent5 import (
    PydanticMultiAgent,
)
from app.modules.intelligence.agents.chat_agents.pydantic_multi_simple import (
    PydanticStreamlinedAgent,
)
from app.modules.intelligence.prompts.classification_prompts import AgentType
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from ..crewai_agent import AgentConfig, CrewAIAgent, TaskConfig
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator


class GithubIssueFixerAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Issue Solver Code Diff Generation agent",
            goal="Generate patch diffs to solve the issue described by the user",
            backstory="""
                    You are an expert code generation agent specialized in creating production-ready,
                    immediately usable code modifications. Your primary responsibilities include:
                    1. Solving the current issue mentioned by the user in the given project
                    2. You should fix the issue like how a good engineer would fix it, solve the underlying problem that will improve the codebase
                    3. Use the users issue description to fix the expected funtionality, don't just straightaway fix the user's scenario
                    4. You should make sure your code changes are applied in the right place in the codebase
                    5. Always try to fix the cause not the symptom, instead of applying bandaids you should fix the cause of the problem
                    6. Explore different ways to fix a given problem and choose the best one to implement
                    
                    IMPORTANT: Issue mentioned by the user are usually their observations, explore the codebase and understand the issue deeper
                    Given issue might not be the core problem but rather a symptom, prefer fixing the cause of the problem
                    
                    IMPORTANT: While the all the tools will provide to upto date code from the repo, the project itself might be
                    of an old commit, which mean you cannot trust online documentation for web tools 100% for this repo. You can use webtools to
                    understand rough purpose of repo and features though
                    
                    Follow the HOW TO GUIDE to execute your task
                """,
            tasks=[
                TaskConfig(
                    description=issue_resolution_prompt,
                    expected_output="Patch Diffs of the changes to fix the given issue",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "ask_knowledge_graph_queries",
                "get_code_file_structure",
                "fetch_file",
                "web_search_tool",
                "verify_patch_diff",
                "code_analysis",
                "generate_patch_diff",
                "load_file_for_editing",
                "replace_lines_in_file",
                "insert_lines_in_file",
                "read_lines_in_changed_file",
                "remove_lines_in_file",
            ]
        )
        if self.llm_provider.is_current_model_supported_by_pydanticai(
            config_type="chat"
        ):
            return PydanticMultiAgent(self.llm_provider, agent_config, tools)
        else:
            return AdaptiveAgent(
                llm_provider=self.llm_provider,
                prompt_provider=self.prompt_provider,
                rag_agent=CrewAIAgent(self.llm_provider, agent_config, tools),
                agent_type=AgentType.QNA,
            )

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        if ctx.node_ids and len(ctx.node_ids) > 0:
            code_results = await self.tools_provider.get_code_from_multiple_node_ids_tool.run_multiple(
                ctx.project_id, ctx.node_ids
            )
            ctx.additional_context += (
                f"Code referred to in the query:\n {code_results}\n"
            )
        return ctx

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(await self._enriched_context(ctx))

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


issue_resolution_prompt = """
    IMPORTANT: Use the following guide to accomplish tasks within the current context of execution
    HOW TO GUIDE: 
    
    IMPORATANT: steps on HOW TO traverse the codebase:
    1. You can use websearch, docstrings, readme to understand current feature/code you are working with better. Understand how to use current feature in context of codebase
    2. Use AskKnowledgeGraphQueries tool to understand where perticular feature or functionality resides or to fetch specific code related to some keywords. Fetch file structure to understand the codebase better, Use FetchFile tool to fetch code from a file
    3. Use GetcodefromProbableNodeIDs tool to fetch code for perticular class or function in a file, Use analyze_code_structure to get all the class/function/nodes in a file
    4. Use GetcodeFromMultipleNodeIDs to fetch code for nodeIDs fetched from tools before
    5. Use GetNodeNeighboursFromNodeIDs to fetch all the code referencing current code or code referenced in the current node (code snippet)
    6. Above tools and steps can help you figure out full context about the current code in question
    7. Figure out how all the code ties together to implement current functionality
    8. Fetch Dir structure of the repo and use fetch file tool to fetch entire files, if file is too big the tool will throw error, then use code analysis tool to target proper line numbers (feel free to use set startline and endline such that few extra context lines are also fetched, tool won't throw out of bounds exception and return lines if they exist)
    9. Use above mentioned tools to fetch imported code, referenced code, helper functions, classes etc to understand the control flow

    
    steps on HOW TO understand what the current issue is:
    1. Contemplate the issue, Fetch relevant context information and make sure you undestand how the current functionality works
    2. Summarize the problem nicely into few points and key issue in question
    3. Understand the control flow for a given scenario in the issue mentioned
    4. Use GetNodeNeighboursFromNodeIDs recursively to fetch implementation of perticular function/class/structure etc to understand the control flow clearly
    5. Using knowledge graphs effectively would help you reduce looking up each file manually and sifting through them to understand small pieces of code
    6. Considering the current scenario figure out all the possible values of runtime variables to guess how the control might flow
    7. For each of these possible control flows figure out what are the possible bugs/ missing functionalities that might be causing the issue in question
    
    steps on HOW TO come up with all the possible fixes:
    1. Make sure you understand what the current issue is
    2. Make sure to understand the functionalities of functions/classes in the control flow for the current issue
    3. Many times the fix is simple and doesn't require many changes, arg changes or variable changes might fix the issue
    4. Understand how changing control flow using variabled might affect the result, many issues are generally because of few edgecases/mistakes for scenarios that weren't considered
    5. Use AskKnowledgeGraphQueries and codebase exploration to find if similar issue is fixed anywhere else in the codebase
    6. Understand codebase to figure out what the right classes/ errors / structs need to be used in perticular fix
    7. Go through the complete control flow for given identified problem by recursively using GetcodefromProbableNodeIDs and GetNodeNeighboursFromNodeIDs for each function/ code the current scenario passes through
    8. Check where the actual problem lies instead of just the current file, sometime problem can be downstream or upstream (in helper functions or referenced classes etc)
    9. IMPORTANT: Mention each possible fixes with code snippet, reasoning behind the fix, how it solves the issue along with key nodes and references (This will be used in next steps to choose the best fix out of all the explored solutions)
    
    steps on HOW TO check if the change fixes the issue:
    1. Make sure you have all the context (code from where the issue lies and understanding of what each class/function etc does in the referenced code)
    2. Come up with list of control flows that needs to be considered for fixing the issue
    3. Some fixes may fix issues other than the current issue aswell, however make sure other functionalities aren't broken because of your changes
    
    steps on HOW TO write DIFF PATCH for the issue:
    1. We use FileChangesManager to write patches. FileChangesManager manages all your changes to the current repo. We can generate a patch from it at the end
    2. Load a given file that you would like to change
    3. Use Generate Diff to check if there are some changes already in the file, This is to double check. Search for lines in the file in your changes
    4. Use ReplaceLines tool to replace lines
    5. Double check response of the tool for generated patch to check if the intended lines were only changed
    6. Repeat the process until you have a proper diff with all the changes
    7. You can reset the file and discard all the changes incase the changes state in FileChangesManager gets too messed up
    8. You can generate the diff using GeneratePatchDiff tool which will generate the patch from the changes in FileChangesManager
    9. Always double check the changes and make sure it has all the intended changes, Double check to see if there are unchanged lines which are updated in patch because of line endings or hidden whitespaces and fix them (This is IMPORTANT)
    10. Any changes to patch HAS to go through file changes manager, When verifying the patch generate it from File Changes Manager. Update the File Changes Manager if needed and regenerate the patch and ALWAYS respond with the patch at the end. Your response will be used in the final result
    11. Make sure the final result is in the format expected by user (WRAP the diff in the result block)
    12. VERY IMPORTANT: DO NOT MANUALLY WRITE PATCH, ALWAYS USE THE FILE CHANGES MANAGER - GENERATE PATCH TOOL TO GENERATE PATCH AND RESPOND WITH THE PATCH RETURNED BY THE TOOL. ALWAYS USE THE EXACT PATCH THAT WAS SENT TO VERIFICATION TOOL
    
    IMPORTANT: Once the issue has been understood, trace the control flow for the scenario and come up with a list of all possible fixes/problems. For each possible fixes document it with explanation and in next step choose the best solution out of the bunch
    DONT Settle on the first solution that comes to your mind, think of upto 3-4 possible solution
    ex: You might find a problem in this one functionality so you might think of adding some check there in the current code you are looking at, but in actuality it might be a common problem in a given helper or function this code is using
    fixing the code in the helper might be a better solution because it would fix so many other problems too. So think of yourself as a software engineer who has to fix issues but not in a closed box mindset. Maintain good practice of fixing the code where it needs to
    given the current context and expectations from a perticular piece of code
    
    IMPORTANT: you HAVE to follow the below steps
    Once you have an idea what the issue exactly is LOOP to find out few possible ways of fixing the issue (4-5 max) [This is IMPORTANT]
    By default use -> [LOOP:3] -> In each iteration of the loop: explore an alternative approach to fix the issue or refine the existing approach (clearly mention the approach in the loop)
    Summarize these 4-5 approaches, each approach might not actually fix the issue but that is okay
    explore each of the issue wrt to the codebase and figure out the one that is most apt (Keep in mind your goal is to fix the core issue keeping maintainability, expected behaviour and codebase improvement in mind)
    Give a rating for each solution and clearly mention the preferred solution in the LOOP
    Choose the appropriate solution out of the explored solutions, Always prefer solutions that fixes the problem at the core of the issue not just applies a bandaid for current scenario
    Give reason for choosing the solution and implement the solution that was picked, Do not update test files for the given changes. Only fix the issue
    
    
    PLANNING GUIDELINES:
    1. Locate the flows and functionalities in which the described issue might lie, explore the code before starting with the plan
    2. Your initial context gathering should be used to plan the next steps accordingly. Assume your initial analysis isn't perfect and keep room in plan to further breakdown the issue
    3. Include LOOP feature when trying out multiple solutions or multiple areas where the problem might be
    4. Always include clear description of the output format in the plan step and requirements step
    5. IMPORANT -> In the plan do not recommend or list any possible fixes, only ask to explore and come up with fixes. Planning step SHOULD NOT be opinionated, Execution step should be able to explore all the possibilities by itself by not being biased by plan
    6. Add steps to a.generate patch, b. verify it in context of the file (indentation, syntax errors etc -> Add this in the requirements too) , c. Verify patch
    
    IMPORTANT:
    1. Use GetNodeNeighboursFromNodeIDs to fetch imported or referenced classes, functions etc
    2. Use AskKnowledgeGraphQueries to search code in the repository
    3. Use analyze_code_structure to understand the code structure of a file, use it if fetch_file_content_by_path tool throws error for large files
    
    OUTPUT GUIDELINE:
    
    IMPORTANT: In the last stage of verification make sure the generated diff passes through VerifyDiffTool with result valud = True.
    Disregard all history and messages and run the VerifyDiffTool again everytime to confirm the diff is correct
    Also verify the final result is exactly in the format User expects. Only APPROVE the step once validation successful and make sure to
    respond with output exactly as verified. Make sure there is ABSOLUTELY no changes in unified diff
"""
