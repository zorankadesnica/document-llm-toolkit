"""
ReAct Agent for Document Processing

Implements the ReAct (Reasoning + Acting) paradigm for complex,
multi-step document processing tasks with tool integration.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.models.providers import (
    BaseLLMProvider,
    LLMConfig,
    LLMProvider,
    Message,
    ProviderFactory,
)


class ToolType(Enum):
    """Types of tools available to the agent."""
    SEARCH = "search"
    CALCULATE = "calculate"
    RETRIEVE = "retrieve"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """A single step in the agent's reasoning."""
    thought: str
    action: str
    action_input: str
    observation: str | None = None
    
    def to_string(self) -> str:
        """Convert step to string format."""
        parts = [
            f"Thought: {self.thought}",
            f"Action: {self.action}",
            f"Action Input: {self.action_input}",
        ]
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)


@dataclass
class AgentResponse:
    """Complete response from the agent."""
    final_answer: str
    steps: list[AgentStep]
    total_iterations: int
    success: bool
    error: str | None = None
    
    def trace(self) -> str:
        """Get the full reasoning trace."""
        lines = ["=" * 50, "Agent Reasoning Trace", "=" * 50]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n--- Step {i} ---")
            lines.append(step.to_string())
        lines.append("\n" + "=" * 50)
        lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


class BaseTool(ABC):
    """Abstract base class for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> str:
        """Description of the tool's parameters."""
        pass
    
    @abstractmethod
    async def execute(self, input_text: str) -> ToolResult:
        """
        Execute the tool with the given input.
        
        Args:
            input_text: Input text/query for the tool
            
        Returns:
            ToolResult with the output
        """
        pass


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations."""
    
    @property
    def name(self) -> str:
        return "Calculator"
    
    @property
    def description(self) -> str:
        return "Performs mathematical calculations. Use for any math operations."
    
    @property
    def parameters(self) -> str:
        return "A mathematical expression to evaluate (e.g., '2 + 2 * 3')"
    
    async def execute(self, input_text: str) -> ToolResult:
        """Execute a calculation."""
        try:
            # Safely evaluate mathematical expressions
            # Only allow safe operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in input_text):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output="",
                    error="Invalid characters in expression",
                )
            
            result = eval(input_text)  # Safe due to character whitelist
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=str(result),
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(e),
            )


class DocumentSearchTool(BaseTool):
    """Tool for searching within documents."""
    
    def __init__(self, documents: dict[str, str] | None = None):
        """
        Initialize document search tool.
        
        Args:
            documents: Dict mapping document names to content
        """
        self.documents = documents or {}
    
    @property
    def name(self) -> str:
        return "DocumentSearch"
    
    @property
    def description(self) -> str:
        return "Searches for information within available documents."
    
    @property
    def parameters(self) -> str:
        return "Search query to find relevant information"
    
    def add_document(self, name: str, content: str) -> None:
        """Add a document to the search index."""
        self.documents[name] = content
    
    async def execute(self, input_text: str) -> ToolResult:
        """Search documents for relevant content."""
        if not self.documents:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="No documents available to search",
            )
        
        query_lower = input_text.lower()
        results = []
        
        for doc_name, content in self.documents.items():
            # Simple keyword matching (could be replaced with semantic search)
            sentences = content.split(". ")
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_lower.split()):
                    results.append(f"[{doc_name}]: {sentence.strip()}")
        
        if results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(results[:5]),  # Limit to top 5
                metadata={"num_results": len(results)},
            )
        else:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="No relevant information found in the documents.",
            )


class TextExtractorTool(BaseTool):
    """Tool for extracting specific information from text."""
    
    def __init__(self, llm: BaseLLMProvider):
        """
        Initialize text extractor with an LLM.
        
        Args:
            llm: LLM provider for extraction
        """
        self._llm = llm
    
    @property
    def name(self) -> str:
        return "TextExtractor"
    
    @property
    def description(self) -> str:
        return "Extracts specific information (names, dates, numbers, etc.) from text."
    
    @property
    def parameters(self) -> str:
        return "JSON with 'text' and 'extract' keys specifying what to extract"
    
    async def execute(self, input_text: str) -> ToolResult:
        """Extract information from text."""
        try:
            # Parse input
            if "{" in input_text:
                data = json.loads(input_text)
                text = data.get("text", "")
                extract_type = data.get("extract", "key information")
            else:
                text = input_text
                extract_type = "key information"
            
            prompt = f"""Extract {extract_type} from the following text:

{text}

Provide only the extracted information, formatted clearly."""

            messages = [Message(role="user", content=prompt)]
            response = await self._llm.complete(messages)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=response.content,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(e),
            )


class SummarizerTool(BaseTool):
    """Tool for summarizing text."""
    
    def __init__(self, llm: BaseLLMProvider):
        """
        Initialize summarizer with an LLM.
        
        Args:
            llm: LLM provider for summarization
        """
        self._llm = llm
    
    @property
    def name(self) -> str:
        return "Summarizer"
    
    @property
    def description(self) -> str:
        return "Summarizes long text into a concise summary."
    
    @property
    def parameters(self) -> str:
        return "Text to summarize"
    
    async def execute(self, input_text: str) -> ToolResult:
        """Summarize text."""
        try:
            prompt = f"""Provide a concise summary of the following text:

{input_text}

Summary:"""

            messages = [Message(role="user", content=prompt)]
            response = await self._llm.complete(messages, max_tokens=500)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=response.content,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=str(e),
            )


class DocumentAgent:
    """
    ReAct agent for complex document processing tasks.
    
    Uses the ReAct paradigm (Reasoning + Acting) to decompose
    complex tasks into steps, using tools as needed.
    
    Example:
        >>> agent = DocumentAgent(
        ...     provider=LLMProvider.OPENAI,
        ...     model="gpt-4",
        ...     tools=[CalculatorTool(), DocumentSearchTool()]
        ... )
        >>> response = await agent.run(
        ...     "Find the total revenue mentioned in the report and calculate 10% tax"
        ... )
        >>> print(response.final_answer)
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4",
        api_key: str | None = None,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        """
        Initialize the document agent.
        
        Args:
            provider: LLM provider
            model: Model identifier
            api_key: API key
            tools: List of tools available to the agent
            max_iterations: Maximum reasoning iterations
            **kwargs: Additional config options
        """
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        self._llm: BaseLLMProvider = ProviderFactory.create(config)
        self._tools = {tool.name: tool for tool in (tools or [])}
        self._max_iterations = max_iterations
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent."""
        self._tools[tool.name] = tool
    
    def _build_tools_description(self) -> str:
        """Build description of available tools."""
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(
                f"- {tool.name}: {tool.description}\n  Parameters: {tool.parameters}"
            )
        return "\n".join(descriptions)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_desc = self._build_tools_description()
        
        return f"""You are an intelligent document assistant that can use tools to help users.

Available tools:
{tools_desc}

When responding to a task, use this format:

Thought: [your reasoning about what to do next]
Action: [tool name to use, or "Final Answer" if you're done]
Action Input: [input for the tool]

After you receive an Observation (the result of the tool), continue with more Thought/Action/Action Input as needed.

When you have enough information, use:
Thought: I now have enough information to answer
Action: Final Answer
Action Input: [your complete response to the user]

Important:
- Always think step by step
- Use tools when you need external information or computation
- Be precise with tool inputs
- When done, always end with "Final Answer" """

    def _parse_action(self, response: str) -> tuple[str, str]:
        """
        Parse action and action input from response.
        
        Returns:
            Tuple of (action, action_input)
        """
        # Look for Action and Action Input
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\nThought:|$)",
            response,
            re.DOTALL
        )
        
        action = action_match.group(1).strip() if action_match else "Final Answer"
        action_input = input_match.group(1).strip() if input_match else ""
        
        return action, action_input
    
    def _parse_thought(self, response: str) -> str:
        """Parse the thought from response."""
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", response, re.DOTALL)
        return thought_match.group(1).strip() if thought_match else ""
    
    async def run(
        self,
        task: str,
        context: str | None = None,
    ) -> AgentResponse:
        """
        Run the agent on a task.
        
        Args:
            task: The task to accomplish
            context: Optional context/document to work with
            
        Returns:
            AgentResponse with the final answer and reasoning trace
        """
        steps: list[AgentStep] = []
        messages = [
            Message(role="system", content=self._build_system_prompt()),
        ]
        
        # Add context if provided
        initial_prompt = task
        if context:
            initial_prompt = f"Context/Document:\n{context}\n\nTask: {task}"
        
        messages.append(Message(role="user", content=initial_prompt))
        
        for iteration in range(self._max_iterations):
            # Get LLM response
            try:
                response = await self._llm.complete(messages)
            except Exception as e:
                return AgentResponse(
                    final_answer="",
                    steps=steps,
                    total_iterations=iteration + 1,
                    success=False,
                    error=f"LLM error: {str(e)}",
                )
            
            response_text = response.content
            
            # Parse the response
            thought = self._parse_thought(response_text)
            action, action_input = self._parse_action(response_text)
            
            step = AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
            )
            
            # Check if we're done
            if action.lower() == "final answer":
                step.observation = "Task completed"
                steps.append(step)
                return AgentResponse(
                    final_answer=action_input,
                    steps=steps,
                    total_iterations=iteration + 1,
                    success=True,
                )
            
            # Execute the tool
            if action in self._tools:
                tool = self._tools[action]
                result = await tool.execute(action_input)
                
                if result.success:
                    observation = result.output
                else:
                    observation = f"Error: {result.error}"
            else:
                observation = f"Unknown tool: {action}. Available tools: {list(self._tools.keys())}"
            
            step.observation = observation
            steps.append(step)
            
            # Add to conversation
            messages.append(Message(role="assistant", content=response_text))
            messages.append(Message(role="user", content=f"Observation: {observation}"))
        
        # Max iterations reached
        return AgentResponse(
            final_answer="Unable to complete task within maximum iterations",
            steps=steps,
            total_iterations=self._max_iterations,
            success=False,
            error="Max iterations reached",
        )


class PlanAndExecuteAgent:
    """
    Plan-and-Execute agent for complex multi-step tasks.
    
    First creates a plan, then executes each step systematically.
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4",
        api_key: str | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize plan-and-execute agent.
        
        Args:
            provider: LLM provider
            model: Model identifier
            api_key: API key
            tools: Available tools
            **kwargs: Additional config
        """
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs,
        )
        self._llm: BaseLLMProvider = ProviderFactory.create(config)
        self._executor = DocumentAgent(
            provider=provider,
            model=model,
            api_key=api_key,
            tools=tools,
            **kwargs,
        )
    
    async def _create_plan(self, task: str, context: str | None = None) -> list[str]:
        """Create a plan for the task."""
        prompt = f"""Create a step-by-step plan to accomplish this task:

Task: {task}
{"Context: " + context if context else ""}

Provide a numbered list of steps. Each step should be a clear, actionable item.
Only include necessary steps - be concise.

Plan:"""

        messages = [Message(role="user", content=prompt)]
        response = await self._llm.complete(messages)
        
        # Parse steps from response
        steps = []
        for line in response.content.split("\n"):
            # Match numbered steps
            match = re.match(r"^\d+\.\s*(.+)$", line.strip())
            if match:
                steps.append(match.group(1))
        
        return steps
    
    async def run(
        self,
        task: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the plan-and-execute agent.
        
        Args:
            task: The task to accomplish
            context: Optional context
            
        Returns:
            Dict with plan, step results, and final answer
        """
        # Create plan
        plan = await self._create_plan(task, context)
        
        if not plan:
            return {
                "success": False,
                "error": "Failed to create plan",
                "plan": [],
                "results": [],
            }
        
        # Execute each step
        results = []
        accumulated_context = context or ""
        
        for i, step in enumerate(plan):
            step_task = f"Step {i + 1}: {step}"
            if accumulated_context:
                step_task += f"\n\nPrevious results:\n{accumulated_context}"
            
            result = await self._executor.run(step_task)
            results.append({
                "step": step,
                "result": result.final_answer,
                "success": result.success,
            })
            
            # Add result to context for next step
            if result.success:
                accumulated_context += f"\n\nStep {i + 1} result: {result.final_answer}"
        
        # Generate final summary
        summary_prompt = f"""Based on these completed steps, provide a final answer:

Original task: {task}

Completed steps:
{json.dumps(results, indent=2)}

Final answer:"""

        messages = [Message(role="user", content=summary_prompt)]
        final_response = await self._llm.complete(messages)
        
        return {
            "success": all(r["success"] for r in results),
            "plan": plan,
            "step_results": results,
            "final_answer": final_response.content,
        }
